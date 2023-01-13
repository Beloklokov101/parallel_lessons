/*
 * Author: Nikolay Khokhlov <k_h@inbox.ru>, 2016
 */

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <mpi.h>

#define ind(i, j) (((i + l->nx) % l->nx) + ((j + l->ny) % l->ny) * (l->nx))

typedef struct {
	int nx, ny;
	int *u0;
	int *u1;
	int steps;
	int save_steps;

	int rank;
	int size;
    int *dims;
    int *coords;
	int start_col, end_col;
	int start_row, end_row;
	MPI_Datatype type_col, type_row, type_block, type_block_right, type_block_down;
    MPI_Comm decart_comm;
} life_t;

void life_init(const char *path, life_t *l);
void life_free(life_t *l);
void life_exchange(life_t *l);
void life_step(life_t *l);
void life_save_vtk(const char *path, life_t *l);
void life_gather(life_t *l);
void life_gather2(life_t *l);

int main(int argc, char **argv)
{
	// if (argc != 2) {
	// 	printf("Usage: %s input file.\n", argv[0]);
	// 	return 0;
	// }
	MPI_Init(&argc, &argv);
	life_t l;
	life_init(argv[1], &l);
	
	int i;
	char buf[100];

	double time_start = MPI_Wtime();
	for (i = 0; i < l.steps; i++) {
		if (i % l.save_steps == 0) {
			sprintf(buf, "outMix/life_%06d.vtk", i);
			life_gather2(&l);
			if ((l.coords[0] == l.dims[0] - 1) && (l.coords[1] == l.dims[1] - 1)){
                printf("Saving step %d to '%s'.\n", i, buf);
				// life_save_vtk(buf, &l);
			}
		}
		life_step(&l);
	}
	double time_finish = MPI_Wtime();

	FILE *fout = fopen(argv[2], "a");
	if (l.rank == 0){
		fprintf(fout, "%i %f\n", l.size, time_finish - time_start);
	}
	fclose(fout);
	
	printf(fout, "%i %f\n", l.size, time_finish - time_start);

	life_free(&l);
	MPI_Finalize();
	return 0;
}

/**
 * Загрузить входную конфигурацию.
 * Формат файла, число шагов, как часто сохранять, размер поля, затем идут координаты заполненых клеток:
 * steps
 * save_steps
 * nx ny
 * i1 j2
 * i2 j2
 */
void life_init(const char *path, life_t *l)
{
	FILE *fd = fopen(path, "r");
	assert(fd);
	assert(fscanf(fd, "%d\n", &l->steps));
	assert(fscanf(fd, "%d\n", &l->save_steps));
	printf("Steps %d, save every %d step.\n", l->steps, l->save_steps);
	assert(fscanf(fd, "%d %d\n", &l->nx, &l->ny));
	printf("Field size: %dx%d\n", l->nx, l->ny);

	l->u0 = (int*)calloc(l->nx * l->ny, sizeof(int));
	l->u1 = (int*)calloc(l->nx * l->ny, sizeof(int));
	
	int i, j, r, cnt;
	cnt = 0;
	while ((r = fscanf(fd, "%d %d\n", &i, &j)) != EOF) {
		l->u0[ind(i, j)] = 1;
		cnt++;
	}
	printf("Loaded %d life cells.\n", cnt);
	fclose(fd);

	/* Decompozition. */
    MPI_Comm_size(MPI_COMM_WORLD, &(l->size));
	// MPI_Comm_rank(MPI_COMM_WORLD, &(l->rank));

    int ndims, reorder, periods[2];
    ndims = 2;
    l->dims = (int*)calloc(2, sizeof(int));
    l->coords = (int*)calloc(2, sizeof(int));
    l->dims[0] = 0;
    l->dims[1] = 0;
    MPI_Dims_create(l->size, 2, l->dims);
    // l->dims = &dims;
    printf("dims = %d, %d\n", l->dims[0], l->dims[1]);
    periods[0] = 0;
    periods[1] = 0;
    reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, l->dims, periods, reorder, &(l->decart_comm));
    MPI_Comm_rank(l->decart_comm, &(l->rank));
    MPI_Comm_size(l->decart_comm, &(l->size));
    MPI_Cart_coords(l->decart_comm, l->rank, ndims, l->coords);
    // l->coords = &coords;

	l->start_col = l->coords[0] * (l->nx / l->dims[0]);
	l->end_col = (l->coords[0] + 1) * (l->nx / l->dims[0]);
	if (l->coords[0] == l->dims[0] - 1) l->end_col = l->nx;

	l->start_row = l->coords[1] * (l->ny / l->dims[1]);
	l->end_row = (l->coords[1] + 1) * (l->ny / l->dims[1]);
	if (l->coords[1] == l->dims[1] - 1) l->end_row = l->ny;

    printf("\n#%d: coord[0] = %d, coord[1] = %d\n", l->rank, l->coords[0], l->coords[1]);
	printf("#%d: start_col = %d, end_col = %d\n", l->rank, l->start_col, l->end_col);
	printf("#%d: start_row = %d, end_row = %d\n\n", l->rank, l->start_row, l->end_row);

    if (l->coords[1] != l->dims[1] - 1)
	    MPI_Type_vector(l->ny / l->dims[1], 1, l->nx, MPI_INT, &(l->type_col));
    else
	    MPI_Type_vector(l->ny - (l->dims[1] - 1) * (l->ny / l->dims[1]), 1, l->nx, MPI_INT, &(l->type_col));
	MPI_Type_commit(&(l->type_col));

    int row_length = 0;
    if (l->coords[0] != l->dims[0] - 1)
        row_length = l->nx / l->dims[0];
    else 
        row_length = l->nx - (l->dims[0] - 1) * (l->nx / l->dims[0]);

    if (l->coords[0] == 0){
        int count = 2;
        int bl[2] = {1, row_length + 1};
        int disp[2] = {0, - l->nx + 1};
        MPI_Type_indexed(count, bl, disp, MPI_INT, &(l->type_row));
    }
    else if (l->coords[0] == l->dims[0] - 1){
        int count = 2;
        int bl[2] = {row_length + 1, 1};
        int disp[2] = {0, - l->nx + (row_length + 1)};
        MPI_Type_indexed(count, bl, disp, MPI_INT, &(l->type_row));
    }
    else {
        int count = 1;
        int bl[1] = {row_length + 2};
        int disp[1] = {0};
        MPI_Type_indexed(count, bl, disp, MPI_INT, &(l->type_row));
    }
    MPI_Type_commit(&(l->type_row));

    MPI_Type_vector(l->ny / l->dims[1], l->nx / l->dims[0], l->nx, MPI_INT, &(l->type_block));
    MPI_Type_vector(l->ny / l->dims[1], l->nx - (l->dims[0] - 1) * (l->nx / l->dims[0]), l->nx, MPI_INT, &(l->type_block_right));
    MPI_Type_vector(l->ny - (l->dims[1] - 1) * (l->ny / l->dims[1]), l->nx / l->dims[0], l->nx, MPI_INT, &(l->type_block_down));
	
	MPI_Type_commit(&(l->type_block));
	MPI_Type_commit(&(l->type_block_right));
	MPI_Type_commit(&(l->type_block_down));
}

void life_free(life_t *l)
{
	free(l->u0);
	free(l->u1);
	free(l->dims);
	free(l->coords);
	l->nx = l->ny = 0;
    MPI_Comm_free(&(l->decart_comm));
	MPI_Type_free(&(l->type_col));
	MPI_Type_free(&(l->type_row));
	MPI_Type_free(&(l->type_block));
	MPI_Type_free(&(l->type_block_right));
	MPI_Type_free(&(l->type_block_down));
}

void life_save_vtk(const char *path, life_t *l)
{
	FILE *f;
	int i1, i2, j;
	f = fopen(path, "w");
	assert(f);
	fprintf(f, "# vtk DataFile Version 3.0\n");
	fprintf(f, "Created by write_to_vtk2d\n");
	fprintf(f, "ASCII\n");
	fprintf(f, "DATASET STRUCTURED_POINTS\n");
	fprintf(f, "DIMENSIONS %d %d 1\n", l->nx+1, l->ny+1);
	fprintf(f, "SPACING %d %d 0.0\n", 1, 1);
	fprintf(f, "ORIGIN %d %d 0.0\n", 0, 0);
	fprintf(f, "CELL_DATA %d\n", l->nx * l->ny);
	
	fprintf(f, "SCALARS life int 1\n");
	fprintf(f, "LOOKUP_TABLE life_table\n");
	for (i2 = 0; i2 < l->ny; i2++) {
		for (i1 = 0; i1 < l->nx; i1++) {
			fprintf(f, "%d\n", l->u0[ind(i1, i2)]);
		}
	}
	fclose(f);
}

void life_exchange(life_t *l)
{
    int rank_right, rank_left, rank_up, rank_down;
    int coords_right[2], coords_left[2], coords_up[2], coords_down[2];
    coords_right[0] = (l->coords[0] + 1) % l->dims[0];
    coords_right[1] = l->coords[1];
    coords_left[0] = (l->coords[0] - 1 + l->dims[0]) % l->dims[0];
    coords_left[1] = l->coords[1];
    coords_up[0] = l->coords[0];
    coords_up[1] = (l->coords[1] - 1 + l->dims[1]) % l->dims[1];
    coords_down[0] = l->coords[0];
    coords_down[1] = (l->coords[1] + 1) % l->dims[1];
    MPI_Cart_rank(l->decart_comm, coords_right, &rank_right);
    MPI_Cart_rank(l->decart_comm, coords_left, &rank_left);
    MPI_Cart_rank(l->decart_comm, coords_up, &rank_up);
    MPI_Cart_rank(l->decart_comm, coords_down, &rank_down);

    if (l->coords[0] % 2 == 0){
		MPI_Send(l->u0 + ind(l->end_col - 1, l->start_row), 1, l->type_col, rank_right, 0, l->decart_comm);
		MPI_Recv(l->u0 + ind(l->start_col - 1, l->start_row), 1, l->type_col, rank_left, 0, l->decart_comm, MPI_STATUS_IGNORE);

		MPI_Send(l->u0 + ind(l->start_col, l->start_row), 1, l->type_col, rank_left, 0, l->decart_comm);
		MPI_Recv(l->u0 + ind(l->end_col, l->start_row), 1, l->type_col, rank_right, 0, l->decart_comm, MPI_STATUS_IGNORE);
	} else {
		MPI_Recv(l->u0 + ind(l->start_col - 1, l->start_row), 1, l->type_col, rank_left, 0, l->decart_comm, MPI_STATUS_IGNORE);
		MPI_Send(l->u0 + ind(l->end_col - 1, l->start_row), 1, l->type_col, rank_right, 0, l->decart_comm);

		MPI_Recv(l->u0 + ind(l->end_col, l->start_row), 1, l->type_col, rank_right, 0, l->decart_comm, MPI_STATUS_IGNORE);
		MPI_Send(l->u0 + ind(l->start_col, l->start_row), 1, l->type_col, rank_left, 0, l->decart_comm);
	}

    if (l->dims[1] != 1) {
        if (l->coords[1] % 2 == 0){
            MPI_Send(l->u0 + ind(l->start_col - 1, l->start_row), 1, l->type_row, rank_up, 0, l->decart_comm);
            MPI_Recv(l->u0 + ind(l->start_col - 1, l->end_row), 1, l->type_row, rank_down, 0, l->decart_comm, MPI_STATUS_IGNORE);

            MPI_Send(l->u0 + ind(l->start_col - 1, l->end_row - 1), 1, l->type_row, rank_down, 0, l->decart_comm);
            MPI_Recv(l->u0 + ind(l->start_col - 1, l->start_row - 1), 1, l->type_row, rank_up, 0, l->decart_comm, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(l->u0 + ind(l->start_col - 1, l->end_row), 1, l->type_row, rank_down, 0, l->decart_comm, MPI_STATUS_IGNORE);
            MPI_Send(l->u0 + ind(l->start_col - 1, l->start_row), 1, l->type_row, rank_up, 0, l->decart_comm);

            MPI_Recv(l->u0 + ind(l->start_col - 1, l->start_row - 1), 1, l->type_row, rank_up, 0, l->decart_comm, MPI_STATUS_IGNORE);
            MPI_Send(l->u0 + ind(l->start_col - 1, l->end_row - 1), 1, l->type_row, rank_down, 0, l->decart_comm);
        }
    }
}

void life_step(life_t *l)
{
    life_exchange(l);

	int i, j;
	for (j = l->start_row; j < l->end_row; j++) {
		for (i = l->start_col; i < l->end_col; i++) {
			int n = 0;
			n += l->u0[ind(i+1, j)];
			n += l->u0[ind(i+1, j+1)];
			n += l->u0[ind(i,   j+1)];
			n += l->u0[ind(i-1, j)];
			n += l->u0[ind(i-1, j-1)];
			n += l->u0[ind(i,   j-1)];
			n += l->u0[ind(i-1, j+1)];
			n += l->u0[ind(i+1, j-1)];
			l->u1[ind(i,j)] = 0;
			if (n == 3 && l->u0[ind(i,j)] == 0) {
				l->u1[ind(i,j)] = 1;
			}
			if ((n == 3 || n == 2) && l->u0[ind(i,j)] == 1) {
				l->u1[ind(i,j)] = 1;
			}
            // l->u1[ind(i,j)] = l->rank;    //coloring different slices 
		}
	}
	int *tmp;
	tmp = l->u0;
	l->u0 = l->u1;
	l->u1 = tmp;
}

// void life_gather(life_t *l)
// {
// 	if (l->rank == 0) {
// 		int i;
// 		for (i = 1; i < l->size; i++) {
// 			int start = (i * l->ny) / l->size;
// 			int end = ((i + 1) * l->ny) / l->size;
// 			MPI_Recv(l->u0 + ind(0, start), (end-start) * l->nx,
// 			MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
// 		}
// 	} else {
// 		// MPI_Send(l->u0 + ind(0, l->start),
// 		//  (l->end - l->start) * l->nx, MPI_INT, 0, 0, MPI_COMM_WORLD);
// 	}
// }

void life_gather2(life_t *l)
{
	if ((l->coords[0] == l->dims[0] - 1) && (l->coords[1] == l->dims[1] - 1)) {
		int i;
		for (i = 0; i < l->size-1; i++) {
            int rank = 0;
            int coords[] = {i % l->dims[0], i / l->dims[0]};
            MPI_Cart_rank(l->decart_comm, coords, &rank);
            
			int start_col = coords[0] * (l->nx / l->dims[0]);
			int start_row = coords[1] * (l->ny / l->dims[1]);

            if (coords[0] == l->dims[0] - 1){
                MPI_Recv(l->u0 + ind(start_col, start_row), 1,
                l->type_block_right, rank, 0, l->decart_comm, MPI_STATUS_IGNORE);
		    }
            else if (coords[1] == l->dims[1] - 1){
                MPI_Recv(l->u0 + ind(start_col, start_row), 1,
                l->type_block_down, rank, 0, l->decart_comm, MPI_STATUS_IGNORE);
		    }
            else {
                MPI_Recv(l->u0 + ind(start_col, start_row), 1,
                l->type_block, rank, 0, l->decart_comm, MPI_STATUS_IGNORE);
		    }
        }
	} else {
        int last_rank = 0;
        int last_coords[] = {l->dims[0] - 1, l->dims[1] - 1};
        MPI_Cart_rank(l->decart_comm, last_coords, &last_rank);

        if (l->coords[0] == l->dims[0] - 1){
            MPI_Send(l->u0 + ind(l->start_col, l->start_row), 1, 
            l->type_block_right, last_rank, 0, l->decart_comm);
	    }
        else if (l->coords[1] == l->dims[1] - 1){
            MPI_Send(l->u0 + ind(l->start_col, l->start_row), 1, 
            l->type_block_down, last_rank, 0, l->decart_comm);
	    }
        else {
            MPI_Send(l->u0 + ind(l->start_col, l->start_row), 1, 
            l->type_block, last_rank, 0, l->decart_comm);
        }
    }
}
