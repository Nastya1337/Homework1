from typing import List, Union, Any

import torch
import time


# Задание 3.1

# Создание тензоров с разными размерами
tensor1 = torch.randn(64, 1024, 1024)  # 64 x 1024 x 1024
tensor2 = torch.randn(128, 512, 512)   # 128 x 512 x 512
tensor3 = torch.randn(256, 256, 256)   # 256 x 256 x 256

# Вывод информации о тензорах
print("Тензор 1 размер:", tensor1.shape, "тип:", tensor1.dtype)
print("Тензор 2 размер:", tensor2.shape, "тип:", tensor2.dtype)
print("Тензор 3 размер:", tensor3.shape, "тип:", tensor3.dtype)
print(tensor1)
print(tensor2)
print(tensor3)

# Задание 3.2

def measure_time(device='cuda', warmup=3, repeats=10):

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Подготовка для GPU (если доступен)
            if max == 'cuda' and torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()  # Синхронизация перед началом
            else:
                device = 'cpu'

            # Прогревочные запуски
            for _ in range(warmup):
                func(*args, **kwargs)

            # Измерение времени
            times = []
            for _ in range(repeats):
                if device == 'cuda':
                    torch.cuda.synchronize()
                    start_event.record()
                else:
                    start_time = time.time()

                func(*args, **kwargs)

                if device == 'cuda':
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event) / 1000  # переводим в секунды
                else:
                    elapsed_time = time.time() - start_time

                times.append(elapsed_time)

            # Вывод результатов
            avg_time = sum(times) / repeats
            min_time = min(times)
            max_time = max(times)
            print(f"\nРезультаты измерений ({device.upper()}):")
            print(f"Среднее время: {avg_time:.6f} сек")
            print(f"Минимальное время: {min_time:.6f} сек")
            print(f"Максимальное время: {max_time:.6f} сек")
            print(f"Количество замеров: {repeats} (прогрев: {warmup})")

            return avg_time

        return wrapper

    return decorator


# Пример использования
if __name__ == "__main__":
    # Создаем большой тензор
    x = torch.randn(10000, 10000)

    if torch.cuda.is_available():
        x = x.cuda()  # Переносим на GPU


    # Тестируем функцию matmul
    @measure_time(device='cuda' if torch.cuda.is_available() else 'cpu')
    def test_matmul(x):
        return x @ x


    result: object = test_matmul(x)

# # Задание 3.3
#
# # Проверка доступности CUDA
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Используемое устройство: {device}\n")
#
#
#
# # Создание больших матриц
# matrix_sizes = [
#     (64, 1024, 1024),
#     (128, 512, 512),
#     (256, 256, 256)
# ]
#
# matrices_cpu = [torch.randn(size) for size in matrix_sizes]
# matrices_gpu = [mat.to(device) for mat in matrices_cpu]
#
#
# def print_table(headers, rows):
#     """
#     Функция, позволяюшая вывести формат строки
#     :param headers: головные параметры
#     :param rows: строка
#     :return: формат строки
#     """
#     # Определяем ширину колонок
#     col_widths = {
#         max(len(str(row[i])) for row in [headers] + rows
#             for i in range(len(headers)))
#     }
#
#     # Создаем формат строки
#     row_format = " | ".join(["{:<" + str(width) + "}" for width in col_widths])
#
#     # Печатаем заголовки
#     print(row_format.format(*headers))
#     print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
#
#     # Печатаем строки
#     for row in rows:
#         print(row_format.format(*row))
#
#
# def measure_time(operation, description):
#     results = []
#
#     for i, (mat_cpu, mat_gpu) in enumerate(zip(matrices_cpu, matrices_gpu)):
#         size_str = f"{matrix_sizes[i][0]}x{matrix_sizes[i][1]}x{matrix_sizes[i][2]}"
#
#         # Измерение на CPU
#         start_time = time.time()
#         result_cpu = operation(mat_cpu)
#         cpu_time = (time.time() - start_time) * 1000  # мс
#
#         # Измерение на GPU (если доступен)
#         if torch.cuda.is_available():
#             start_event = torch.cuda.Event(enable_timing=True)
#             end_event = torch.cuda.Event(enable_timing=True)
#
#             torch.cuda.synchronize()
#             start_event.record()
#             result_gpu = operation(mat_gpu)
#             end_event.record()
#             torch.cuda.synchronize()
#
#             gpu_time = start_event.elapsed_time(end_event)  # уже в мс
#             speedup = cpu_time / gpu_time
#             gpu_time_str = f"{gpu_time:.1f}"
#             speedup_str = f"{speedup:.1f}x"
#         else:
#             gpu_time_str = "N/A"
#             speedup_str = "N/A"
#
#         results.append([
#             description if i == 0 else "",
#             size_str,
#             f"{cpu_time:.1f}",
#             gpu_time_str,
#             speedup_str
#         ])
#
#     return results
#
#
# # Определение операций
# operations = [
#     (lambda x: torch.matmul(x, x), "Матричное умножение"),
#     (lambda x: x + x, "Поэлементное сложение"),
#     (lambda x: x * x, "Поэлементное умножение"),
#     (lambda x: x.transpose(-2, -1), "Транспонирование"),
#     (lambda x: torch.sum(x), "Сумма всех элементов")
# ]
#
# # Измерение времени для всех операций
# all_results = []
# for op, desc in operations:
#     all_results.extend(measure_time(op, desc))
#
# # Вывод результатов в табличном виде
# headers = ["Операция", "Размер матрицы", "CPU (мс)", "GPU (мс)", "Ускорение"]
# print_table(headers, all_results)



