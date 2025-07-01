import torch
import numpy as np

# Задание 1.1

# # Тензор 3x4 со случайными числами:
# tensor1 = torch.rand(3, 4)
# print("Тензор 3x4 со случайными числами:")
# print(tensor1)

# # Тензор размером 2x3x4, заполненный нулями
# tensor2 = torch.zeros(2, 3, 4)
# print("Тензор 2x3x4 с нулями:")
# print(tensor2)

# # Тензор размером 5x5, заполненный единицами
# tensor3 = torch.ones(5, 5)
# print("Тензор 5x5 с единицами:")
# print(tensor3)

# # Тензор размером 4x4 с числами от 0 до 15 (используя reshape)
#tensor4 = torch.arange(16).reshape(4, 4)
#print("Тензор 4x4 с числами от 0 до 15:")
#print(tensor4)


# Задание 1.2

# # Создаем тензоры A (3x4) и B (4x3):
# A = torch.tensor([[1, 2, 3, 4],
#                   [5, 6, 7, 8],
#                   [9, 10, 11, 12]])
#
# B = torch.tensor([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9],
#                   [10, 11, 12]])


# # Если использовать случайные тензоры вместо заданных значений:
# A = torch.rand(3, 4)
# B = torch.rand(4, 3)

# print("Тензор A:")
# print(A)
# print("\nТензор B:")
# print(B)

# # Транспонирование тензора A:
# A_transposed = A.T
# print("\nТранспонированный тензор A:")
# print(A_transposed)

# # Матричное умножение A и B
# matrix_product = torch.matmul(A, B)  # или A @ B
# print("\nМатричное произведение A и B:")
# print(matrix_product)

# # Поэлементное умножение A и транспонированного B
# # Сначала убедимся, что размерности совпадают (3x4 и 3x4)
# B_transposed = B.T
# elementwise_product = A * B_transposed
# print("\nПоэлементное произведение A и B^T:")
# print(elementwise_product)
#
# # Сумма всех элементов тензора A
# sum_A = torch.sum(A)
# print("\nСумма всех элементов тензора A:")
# print(sum_A)

# Задание 1.3

# # Создаем тензор 5x5x5 со значениями от 0 до 124 (5*5*5=125)
# tensor = torch.arange(125).reshape(5, 5, 5)
# print("Исходный тензор 5x5x5:")
# print(tensor)
# print(f"Размерность: {tensor.shape}\n")
#
# # Извлечение первой строки каждого слоя (матрицы)
# first_rows = tensor[:, 0, :]  # Все слои, первая строка, все столбцы
# print("Первая строка каждого слоя (размер 5x5):")
# print(first_rows)
# print(f"Размерность: {first_rows.shape}\n")
#
# # Извлечение последнего столбца каждого слоя
# last_columns = tensor[:, :, -1]  # Все слои, все строки, последний столбец
# print("Последний столбец каждого слоя (размер 5x5):")
# print(last_columns)
# print(f"Размерность: {last_columns.shape}\n")
#
# # Подматрица 2x2 из центра каждого слоя
# center_submatrix = tensor[:, 1:3, 1:3]  # Все слои, строки 1-2, столбцы 1-2
# print("Подматрица 2x2 из центра каждого слоя (размер 5x2x2):")
# print(center_submatrix)
# print(f"Размерность: {center_submatrix.shape}\n")
#
# # Все элементы с четными индексами по каждому измерению
# even_indices = tensor[::2, ::2, ::2]  # Четные индексы по всем трем осям
# print("Элементы с четными индексами по всем измерениям (размер 3x3x3):")
# print(even_indices)
# print(f"Размерность: {even_indices.shape}")

# Задание 1.4

# Создаем тензор из 24 элементов (можно использовать любые значения)
# Здесь используем числа от 0 до 23:
tensor = torch.arange(24)
print("Исходный тензор:")
print(tensor)
print(f"Размерность: {tensor.shape}\n")

# Преобразование в различные формы
# Форма 2x12
tensor_2x12 = tensor.view(2, 12)
print("Форма 2x12:")
print(tensor_2x12)
print(f"Размерность: {tensor_2x12.shape}\n")

# Форма 3x8
tensor_3x8 = tensor.reshape(3, 8)
print("Форма 3x8:")
print(tensor_3x8)
print(f"Размерность: {tensor_3x8.shape}\n")

# Форма 4x6
tensor_4x6 = tensor.view(4, 6)
print("Форма 4x6:")
print(tensor_4x6)
print(f"Размерность: {tensor_4x6.shape}\n")

# Форма 2x3x4
tensor_2x3x4 = tensor.reshape(2, 3, 4)
print("Форма 2x3x4:")
print(tensor_2x3x4)
print(f"Размерность: {tensor_2x3x4.shape}\n")

# Форма 2x2x2x3
tensor_2x2x2x3 = tensor.view(2, 2, 2, 3)
print("Форма 2x2x2x3:")
print(tensor_2x2x2x3)
print(f"Размерность: {tensor_2x2x2x3.shape}")