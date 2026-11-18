#!/usr/bin/env python3
"""
Параллельная версия умножения матрицы на вектор с использованием Scatterv/Gatherv.
Основана на лекционном материале (стр. 13-17).
"""
from mpi4py import MPI
import numpy as np
import time

def read_params(filename="data/in.dat"):

    textStream = open(filename, 'r')
    N = int(textStream.readline())
    M = int(textStream.readline())
    textStream.close()

    return N, M

def auxiliary_arrays_determination(M, numprocs):

    ave, res = divmod(M, numprocs - 1)
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)

    rcounts[0] = 0
    displs[0] = 0

    for k in range(1, numprocs):

        if k <= res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave

        displs[k] = displs[k - 1] + rcounts[k - 1]

    return rcounts, displs

def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()

    if numprocs < 2:
        if rank == 0:
            print("Для запуска необходимо не менее 2 процессов.")
        return

    #Чтение параметров
    if rank == 0:
        start_time = time.time()

        N, M = read_params()
    else:
        N = 0
        M = 0

    #Рассылка параметров
    N = comm.bcast(N, root=0)
    M = comm.bcast(M, root=0)

    rcounts, displs = auxiliary_arrays_determination(
        M, numprocs
    )

    M_part = rcounts[rank]

    A_part = np.empty((M_part, N), dtype=np.float64)

    #Распределение матрицы
    if rank == 0:
        A = np.empty((M, N), dtype=np.float64)

        textStream = open("data/AData.dat", 'r')

        for j in range(M):
            for i in range(N):
                A[j, i] = float(textStream.readline())

        textStream.close()

        sendcounts = rcounts * N
        senddispls = displs * N

        comm.Scatterv([A, sendcounts, senddispls, MPI.DOUBLE],[A_part, M_part * N, MPI.DOUBLE],root=0)

    else:
        comm.Scatterv(None,[A_part, M_part * N, MPI.DOUBLE],root=0)

    x = np.empty(N, dtype=np.float64)

    if rank == 0:

        textStream = open("data/xData.dat", 'r')

        for i in range(N):
            x[i] = float(textStream.readline())

        textStream.close()

    comm.Bcast([x, N, MPI.DOUBLE], root=0)

    #Локальное вычисление
    b_part = np.dot(A_part, x)

    #Сбор результатов
    if rank == 0:
        b = np.empty(M, dtype=np.float64)
        recvbuf = [b, rcounts, displs, MPI.DOUBLE]
    else:
        b = None
        recvbuf = None

    comm.Gatherv([b_part, M_part, MPI.DOUBLE],recvbuf,root=0)

    #Запись результата
    if rank == 0:

        textStream = open("Results_parallel.dat", 'w')

        for j in range(M):
            print(b[j], file=textStream)

        textStream.close()

        elapsed_time = time.time() - start_time

        print("Result:")
        print(b)
        print(f"Execution time: {elapsed_time:.4f} seconds")
    
    # TODO: Реализуйте:
    # 1. Чтение параметров на процессе 0
    # 2. Рассылку параметров через Bcast
    # 3. Распределение матрицы через Scatterv
    # 4. Распределение вектора через Bcast
    # 5. Локальное вычисление
    # 6. Сбор результатов через Gathervs
    # 7. Запись результата

if __name__ == "__main__":
    main()