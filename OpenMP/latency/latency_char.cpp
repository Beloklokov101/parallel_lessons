#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h> 

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
    ofstream fout0(filename, ios::app);
    if (fout0.is_open()){
        string msg;
        msg.append(msg_len, lett);
        char msgl[msg_len + 1];

        double time_start = MPI_Wtime();
        for (int j = 0; j <= it_num; j++){
            if (rank == 0){
                MPI_Send(&msg[0], msg.size() + 1, MPI_CHAR, size - 1, 0, MPI_COMM_WORLD);
                MPI_Recv(msgl, msg_len + 1, MPI_CHAR, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank == size - 1){
                MPI_Recv(msgl, msg_len + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&msg[0], msg.size() + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
        double time_finish = MPI_Wtime();
        if (rank == 0){
            fout0 << msg_len << " " << (time_finish - time_start) / (2 * it_num) << endl;
            // cout << msg_len << " " << (time_finish - time_start) / (2 * it_num) << endl;
        }
    }
    fout0.close();
    MPI_Finalize(); // Завершение работы с MPI.
    return 0;
}
