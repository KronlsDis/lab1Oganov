#!/usr/bin/env python3
"""
Параллельная версия умножения матрицы на вектор с использованием Send/Recv.
Основана на лекционном материале (стр. 7-12).
"""
from mpi4py import MPI
from numpy import empty, int32, float64, dot
import time

def read_params(filename="data/in.dat"):

    textStream = open(filename, 'r')
    N = int(textStream.readline ())
    M = int(textStream.readline ())
    textStream.close ()

    return N, M

def auxiliary_arrays_determination(M, numprocs):
    """Расчет rcounts и displs для разбиения данных"""

    ave, res = divmod(M, numprocs - 1)
    rcounts = empty(numprocs , dtype=int32)
    displs = empty(numprocs , dtype=int32)

    rcounts[0] = 0
    displs[0] = 0

    for k in range (1, numprocs):
        if k <= res:
            rcounts[k] = ave +1
        else:
            rcounts[k] = ave

        displs[k] = displs[k -1] + rcounts [k -1]

    return rcounts, displs

def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        start_time = time.time()
    numprocs = comm.Get_size()

    if rank == 0:
        N, M = read_params()
    else:
        N = 0
        M = 0

    N = comm.bcast(N, root=0)
    M = comm.bcast(M, root=0)

    rcounts, displs = auxiliary_arrays_determination(M, numprocs)

    A_part = empty((rcounts[rank], N), dtype=float64)

# Распределение матрицы
    if rank == 0:
        textStream = open("data/AData.dat", 'r')

        for k in range(1, numprocs):

            A_part_for_send = empty((rcounts[k], N), dtype = float64)
            
            for j in range(rcounts[k]):
                for i in range(N):
                    A_part_for_send[j, i] = float(textStream.readline())

            comm.Send([A_part_for_send, rcounts[k] * N, MPI.DOUBLE], dest=k, tag=0)

        textStream.close()

    else:
        comm.Recv([A_part, rcounts[rank] * N, MPI.DOUBLE], source=0, tag=0)
    
# Распределение вектора
    x = empty(N, dtype = float64)

    if rank == 0:

        textStream = open("data/xData.dat", 'r')

        for i in range(N):
            x[i] = float(textStream.readline())

        textStream.close()

    comm.Bcast([x, N, MPI.DOUBLE], root=0)

# Локальное вычисление
    b_part = empty(rcounts[rank], dtype=float64)

    if rank != 0:
        # b_part = dot(A_part , x)
        for j in range(rcounts[rank]):
            b_part[j] = 0
            for i in range(N):
                b_part[j] += A_part[j, i] * x[i]

# Сбор результатов
    status = MPI.Status()

    if rank == 0:

        b = empty(M, dtype=float64)
 
        for k in range(1, numprocs):
            comm.Probe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
            comm.Recv([b[displs[status.Get_source()]:], rcounts[status.Get_source()], MPI.DOUBLE], 
                      source = status.Get_source(), tag = MPI.ANY_TAG, status = None)
    else:
        comm.Send([b_part, rcounts[rank], MPI.DOUBLE], dest = 0, tag = 0)

# Запись результата
    if rank == 0:

        textStream = open("Results_parallel.dat", 'w')

        for j in range(M):
            print(b[j], file=textStream)

        textStream.close()

        # print("Result:")
        # print(b)
        
        elapsed_time = time.time() - start_time
        print(f"Execution time: {elapsed_time:.4f} seconds")

  

if __name__ == "__main__":
    main()