#!/usr/bin/env python3
"""
Генерация тестовых данных для ЛР №1
"""

import numpy as np
import os


def generate_test_data(M, N, seed, output_dir):
    """Генерация тестовых данных"""

    np.random.seed(seed)

    A = np.random.rand(M, N)
    x = np.random.rand(N)

    os.makedirs(output_dir, exist_ok=True)

    # Запись матрицы: одно число на строку
    textStream = open(os.path.join(output_dir, "AData.dat"), "w")

    for j in range(M):
        for i in range(N):
            print(A[j, i], file=textStream)

    textStream.close()

    # Запись вектора: одно число на строку
    textStream = open(os.path.join(output_dir, "xData.dat"), "w")

    for i in range(N):
        print(x[i], file=textStream)

    textStream.close()

    # Запись размеров матрицы
    textStream = open(os.path.join(output_dir, "in.dat"), "w")

    print(N, file=textStream)
    print(M, file=textStream)

    textStream.close()


if __name__ == "__main__":
    generate_test_data(2000, 1000, 42, "data")