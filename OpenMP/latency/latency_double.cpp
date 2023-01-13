#include <iostream>
#include <fstream>
// #include <stdio.h>
//#include <string>
#include <vector>
#include <mpi.h> // Заголовочный файл MPI.
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
    char lett = 'A';
    int msg_len = atoi(argv[1]);
    int it_num = 1024 * 1024;
    cout << "iterations per cycle = " << it_num << endl;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    // MPI_Send gets non-const
    string filename(argv[2]);
    ofstream fout0;
    fout0.open(filename.c_str(), ios_base::app);
    if (fout0.is_open() /*&& fout1.is_open()*/){
        vector<double> vec_in(msg_len);
        vector<double> vec_out(msg_len);
        //cout << "The size of vec_in is " << vec_in.size() << " doubles.\n";
        // cout << "N = " << msg_len << endl;
        double time_start = MPI_Wtime();
        for (int j = 0; j <= it_num; j++){
            if (rank == 0){
                MPI_Send(&vec_in[0], vec_in.size(), MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&vec_out[0], msg_len, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank == size - 1){
                MPI_Recv(&vec_out[0], msg_len, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&vec_in[0], vec_in.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
        double time_finish = MPI_Wtime();

        if (rank == 0){
            fout0 << msg_len * 8 << " " << (time_finish - time_start) / (2 * it_num) << endl;
        }
    }
    MPI_Finalize(); // Завершение работы с MPI.
    return 0;
}
