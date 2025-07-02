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

# Установим размер матриц
N = 1000  # Размер матриц

# Создаем случайные тензоры для операций
a = torch.randn(N, N)
b = torch.randn(N, N)


# Функция для измерения времени выполнения
def measure_time(func, *args, device='cpu', warmup=10, repeats=100):
    # Переносим данные на нужное устройство
    if device == 'cuda':
        for arg in args:
            arg = arg.to('cuda')

    # Прогрев
    for _ in range(warmup):
        func(*args)

    # Измерение времени
    times = []
    for _ in range(repeats):
        start_time = time.time()
        func(*args)
        times.append(time.time() - start_time)

    return sum(times) / len(times) * 1000  # Возвращаем среднее время в миллисекундах


# Список операций для тестирования
operations = {
    "Матричное умножение": lambda: torch.matmul(a, b),
    "Поэлементное сложение": lambda: a + b,
    "Поэлементное умножение": lambda: a * b,
    "Транспонирование": lambda: a.t(),
    "Сумма всех элементов": lambda: a.sum()
}

# Сбор результатов
results = []

for op_name, op_func in operations.items():
    # Измеряем время на CPU
    cpu_time = measure_time(op_func, device='cpu')

    # Измеряем время на GPU, если доступен
    gpu_time = None
    if torch.cuda.is_available():
        gpu_time = measure_time(op_func, device='cuda')

    # Рассчитываем ускорение
    speedup = None
    if gpu_time is not None:
        speedup = cpu_time / gpu_time

    results.append((op_name, cpu_time, gpu_time, speedup))

# Вывод результатов в табличном виде
print(f"{'Операция':<25} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")
print("-" * 70)
for op_name, cpu_time, gpu_time, speedup in results:
    gpu_time_str = f"{gpu_time:.2f}" if gpu_time is not None else "N/A"
    speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
    print(f"{op_name:<25} | {cpu_time:<10.2f} | {gpu_time_str:<10} | {speedup_str:<10}")

