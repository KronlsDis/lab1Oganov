#!/usr/bin/env python3
"""
Последовательная версия умножения матрицы на вектор.
Основана на лекционном материале (стр. 5-6).
"""

from numpy import empty
import time

def read_params(filename="data/in.dat"):
    """Чтение размеров матрицы из файла"""

    textStream = open(filename, 'r')
    N = int(textStream.readline ())
    M = int(textStream.readline ())
    textStream.close ()

    return N, M

def read_matrix(filename="data/AData.dat", M=None, N=None):
    """Чтение матрицы из файла"""

    A = empty((M, N))

    textStream = open(filename, 'r')
    for j in range(M):
        for i in range(N):
            A[j, i] = float(textStream.readline())

    textStream.close ()

    return A

def read_vector(filename="data/xData.dat"):
    """Чтение вектора из файла"""

    with open(filename, 'r') as textStream:
        x = [float(line) for line in textStream]

    return x

def sequential_mat_vec_mult(A, x):
    """Последовательное умножение матрицы на вектор"""
    M, N = A.shape
    b = empty(M)

    for j in range(M):
        b[j] = 0
        for i in range(N):
            b[j] += A[j, i]*x[i]

    return b

def main():
    """Основная функция"""

    start_time = time.time()

    N, M = read_params()
    x = read_vector()

    A = read_matrix(M=M, N=N)

    b = sequential_mat_vec_mult(A,x)

    textStream = open('Results_sequential.dat', 'w')

    for j in range(M):
        print(b[j], file=textStream)

    textStream.close ()

    elapsed_time = time.time() - start_time

    # print("Result:")
    # print(b)
    print(f"Execution time: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    main()