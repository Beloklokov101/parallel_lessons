#include <iostream>
#include <fstream>
// #include <stdio.h>
// #include <string>
#include <vector>
#include <mpi.h> // Заголовочный файл MPI.

using namespace std;

int main(int argc, char *argv[])
{
    char lett = 'A';
    int msg_len_first = 100;
    int msg_len_last = 100;
    string msg;

    int it_num = 1;
    // cin >> it_num; !!!!!!! ??????
    cout << "iterations per cycle = " << it_num << endl;

    // int numtasks, rank; // Номер и число процессов.
    int rank, size;
    MPI_Init(&argc, &argv); // Инициализация MPI.
    // MPI_Comm_size(MPI_COMM_WORLD, &numtasks); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    // string msg = "hello from " + to_string(rank);
    // msg.data();
    // MPI_Send gets non-const
    // bool flag = true; 
    // vector<ofstream> fout_vec;
    // for (int i = 0; i <= size / 2; i++){
    //     ofstream fout_i("nodes_1/test" + to_string(i * 2) + ".txt");
    //     fout_vec.emplace_back(fout_i);
    //     if (!fout_i.is_open()) flag = false;
    // }
    ofstream fout0("test.txt");

    if (fout0.is_open()/*&& fout1.is_open()*/){
        for (int msg_len = msg_len_first; msg_len <= msg_len_last; msg_len+=1){
            // string msg;
            // msg.append(msg_len, lett);
            // char msgl[msg_len + 1];

            vector<double> vec_in(msg_len);
            vector<double> vec_out(msg_len);
            cout << "The size of vec_in is " << vec_in.size() << " doubles.\n";
            cout << "rank = " << rank << endl;
            double time_start = MPI_Wtime();
            for (int j = 0; j <= it_num; j++){
                if (rank % 2 == 0){
                    MPI_Send(&vec_in[0], vec_in.size(), MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD);
                    MPI_Recv(&vec_out[0], msg_len, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cout << "to && from rank = " << (rank + 1 + size) % size << endl;
                }
                else {
                    MPI_Send(&vec_in[0], vec_in.size(), MPI_DOUBLE, (rank - 1 + size) % size, 0, MPI_COMM_WORLD);
                    MPI_Recv(&vec_out[0], msg_len, MPI_DOUBLE, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cout << "to && from rank = " << (rank - 1 + size) % size << endl;
                }
            }
            double time_finish = MPI_Wtime();
            cout << "The size of vec_out is " << vec_out.size() << " doubles.\n";
            cout << "start = " << time_start << " finish = " << time_finish << endl;
            // cout << "#" << rank << ": " << msgl << endl;
            // cout << "time = " << (time_finish - time_start) / (2 * it_num) << endl;
            // for (int k = 0; k < size / 2; k++){
            //     fout_vec[k] << msg_len * 8 << " " << (time_finish - time_start) / (2 * it_num) << endl;
            // }
            if (rank == 0){
                fout0 << msg_len << " " << (time_finish - time_start) / (2 * it_num) << endl;
            }
        }
    }
    // for (int i = 0; i < size / 2; i++){
    //     fout_vec[i].close();
    // }

    MPI_Finalize(); // Завершение работы с MPI.
    return 0;
}