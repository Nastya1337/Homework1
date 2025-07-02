# Домашнее задание к уроку 1: Основы PyTorch

## Задание 1: Создание и манипуляции с тензорами

### 1.1 Создание тензоров

- Тензор 3x4 со случайными числами:
```Python
tensor1 = torch.rand(3, 4)
print("Тензор 3x4 со случайными числами:")
print(tensor1)
```
- Тензор размером 2x3x4, заполненный нулями
```Python
tensor2 = torch.zeros(2, 3, 4)
print("Тензор 2x3x4 с нулями:")
print(tensor2)
```
- Тензор размером 5x5, заполненный единицами
```Python
tensor3 = torch.ones(5, 5)
print("Тензор 5x5 с единицами:")
print(tensor3)
```
- Тензор размером 4x4 с числами от 0 до 15 (используя reshape)
```Python
tensor4 = torch.arange(16).reshape(4, 4)
print("Тензор 4x4 с числами от 0 до 15:")
print(tensor4)
```

### 1.2 Операции с тензорами

- Создаем тензоры A (3x4) и B (4x3):
```Python
A = torch.rand(3, 4)
B = torch.rand(4, 3)
```
- Транспонирование тензора A:
```Python
A_transposed = A.T
print("\nТранспонированный тензор A:")
print(A_transposed)
```
- Матричное умножение A и B
```Python
matrix_product = torch.matmul(A, B)  # или A @ B
print("\nМатричное произведение A и B:")
print(matrix_product)
```
- Поэлементное умножение A и транспонированного B
  
Сначала убедимся, что размерности совпадают (3x4 и 3x4)
```Python
B_transposed = B.T
elementwise_product = A * B_transposed
print("\nПоэлементное произведение A и B^T:")
print(elementwise_product)
```
- Сумма всех элементов тензора A
```Python
sum_A = torch.sum(A)
print("\nСумма всех элементов тензора A:")
print(sum_A)
```
### 1.3 Индексация и срезы

- Создаем тензор 5x5x5 со значениями от 0 до 124 (5*5*5=125)
```Python
tensor = torch.arange(125).reshape(5, 5, 5)
print("Исходный тензор 5x5x5:")
print(tensor)
print(f"Размерность: {tensor.shape}\n")
```
- Извлечение первой строки каждого слоя (матрицы)
```Python
first_rows = tensor[:, 0, :]  # Все слои, первая строка, все столбцы
print("Первая строка каждого слоя (размер 5x5):")
print(first_rows)
print(f"Размерность: {first_rows.shape}\n")
```
- Извлечение последнего столбца каждого слоя
```Python
last_columns = tensor[:, :, -1]  # Все слои, все строки, последний столбец
print("Последний столбец каждого слоя (размер 5x5):")
print(last_columns)
print(f"Размерность: {last_columns.shape}\n")
```
- Подматрица 2x2 из центра каждого слоя
```Python
center_submatrix = tensor[:, 1:3, 1:3]  # Все слои, строки 1-2, столбцы 1-2
print("Подматрица 2x2 из центра каждого слоя (размер 5x2x2):")
print(center_submatrix)
print(f"Размерность: {center_submatrix.shape}\n")
```
- Все элементы с четными индексами по каждому измерению
```Python
even_indices = tensor[::2, ::2, ::2]  # Четные индексы по всем трем осям
print("Элементы с четными индексами по всем измерениям (размер 3x3x3):")
print(even_indices)
print(f"Размерность: {even_indices.shape}")
```
### 1.4 Работа с формами

- Создаем тензор из 24 элементов (можно использовать любые значения)
  
Здесь используем числа от 0 до 23:
```Python
tensor = torch.arange(24)
print("Исходный тензор:")
print(tensor)
print(f"Размерность: {tensor.shape}\n")
```
- Преобразование в различные формы

Форма 2x12
```Python
tensor_2x12 = tensor.view(2, 12)
print("Форма 2x12:")
print(tensor_2x12)
print(f"Размерность: {tensor_2x12.shape}\n")
```
Форма 3x8
```Python
tensor_3x8 = tensor.reshape(3, 8)
print("Форма 3x8:")
print(tensor_3x8)
print(f"Размерность: {tensor_3x8.shape}\n")
```
Форма 4x6
```Python
tensor_4x6 = tensor.view(4, 6)
print("Форма 4x6:")
print(tensor_4x6)
print(f"Размерность: {tensor_4x6.shape}\n")
```
Форма 2x3x4
```Python
tensor_2x3x4 = tensor.reshape(2, 3, 4)
print("Форма 2x3x4:")
print(tensor_2x3x4)
print(f"Размерность: {tensor_2x3x4.shape}\n")
```
Форма 2x2x2x3
```Python
tensor_2x2x2x3 = tensor.view(2, 2, 2, 3)
print("Форма 2x2x2x3:")
print(tensor_2x2x2x3)
print(f"Размерность: {tensor_2x2x2x3.shape}")
```
## Задание 2: Автоматическое дифференцирование 

### 2.1 Простые вычисления с градиентами

- Создаем тензоры с requires_grad=True для автоматического вычисления градиентов
```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)
```

- Вычисляем функцию f(x,y,z) = x² + y² + z² + 2xyz
```python
f = x**2 + y**2 + z**2 + 2 * x * y * z
print(f'f (no grad) = {f}, f.requires_grad = {f.requires_grad}')
```

- Вычисляем градиенты
```python
f.backward()
```

- Получаем градиенты
```python
grad_x = x.grad
grad_y = y.grad
grad_z = z.grad

print("Вычисленные градиенты:")
print(f"df/dx = {grad_x.item()}")
print(f"df/dy = {grad_y.item()}")
print(f"df/dz = {grad_z.item()}\n")
```
- Аналитическое вычисление градиентов
```python
def analytical_gradients(x_val, y_val, z_val):
    df_dx = 2*x_val + 2*y_val*z_val
    df_dy = 2*y_val + 2*x_val*z_val
    df_dz = 2*z_val + 2*x_val*y_val
    return df_dx, df_dy, df_dz
```

- Вычисляем аналитические градиенты
```python
a_grad_x, a_grad_y, a_grad_z = analytical_gradients(x.item(), y.item(), z.item())

print("Аналитические градиенты:")
print(f"df/dx = {a_grad_x}")
print(f"df/dy = {a_grad_y}")
print(f"df/dz = {a_grad_z}\n")
```

- Проверка совпадения результатов
```python
assert torch.allclose(grad_x, torch.tensor(a_grad_x))
assert torch.allclose(grad_y, torch.tensor(a_grad_y))
assert torch.allclose(grad_z, torch.tensor(a_grad_z))
```

### 2.2 Градиент функции потерь

- Создаем данны:
```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=False)
y_true = torch.tensor([2.5, 4.3, 6.1, 7.8], requires_grad=False)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
```

- Реализация функции:
```python
def linear_regression(x, w, b):
    return w * x + b
```
- Реализация MSE loss:
```python
def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)
```
- Вычисляем:
```python
y_pred = linear_regression(x, w, b)
loss = mse_loss(y_pred, y_true)
```
- Вычисляем градиенты:
```python
loss.backward()
```
- Получаем градиенты
```python
print(f"Градиент по w: {w.grad.item():.4f}")
print(f"Градиент по b: {b.grad.item():.4f}")
```
- Аналитическая проверка градиентов:
```python
n = len(x)
analytic_grad_w = (2/n) * torch.sum((y_pred - y_true) * x)
analytic_grad_b = (2/n) * torch.sum(y_pred - y_true)
```
- Проверка совпадения:
```python
assert torch.allclose(w.grad, analytic_grad_w, atol=1e-5)
assert torch.allclose(b.grad, analytic_grad_b, atol=1e-5)
```

### 2.3 Цепное правило

- Определяем переменную x как тензор с необходимыми свойствами для автоградирующих функций
```python
x = torch.tensor(2.0, requires_grad=True)
```
- Определяем составную функцию f(x)
```python
def f(x):
    return torch.sin(x**2 + 1)
```
- Вычисляем значение функции
```python
y = f(x)
```
- Вычисляем градиент вручную
Применяем цепное правило
- df/dx = cos(u) * du/dx, где u = x^2 + 1
- du/dx = 2x
```python
cu = torch.cos(x**2 + 1)  # cos(u)
du_dx = 2 * x            # du/dx
df_dx_manual = cu * du_dx
```
- Вычисляем автоматический градиент с помощью PyTorch
```python
y.backward()  # Это вычисляет градиент для y, так как x имеет requires_grad=True
df_dx_autograd = x.grad
```
- Выводим результаты
```python
print(f"Ручной расчет градиента df/dx: {df_dx_manual.item()}")
print(f"Градиент с использованием torch.autograd: df/dx = {df_dx_autograd.item()}")
```

## Задание 3: Сравнение производительности CPU vs CUDA

### 3.1 Подготовка данных
- Создаем 3 тензора:
  - `tensor1`: Тензор размером 64 × 1024 × 1024, заполненный случайными значениями из нормального распределения
  - `tensor2`: Тензор размером 128 × 512 × 512, также заполненный случайными значениями
  - `tensor3`: Тензор размером 256 × 256 × 256, заполненный случайными значениями

```python
tensor1 = torch.randn(64, 1024, 1024)  # 64 x 1024 x 1024
tensor2 = torch.randn(128, 512, 512)   # 128 x 512 x 512
tensor3 = torch.randn(256, 256, 256)   # 256 x 256 x 256
```

### 3.2 Функция измерения времени
- Параметры функции `measure_time`:
  
  - `func`: функция, время выполнения которой нужно измерить
  - `*args`: аргументы, передаваемые в функцию func
  - `device`: устройство, на котором будет выполняться код ('cpu' или 'cuda' для GPU)
  - `warmup`: количество прогревочных запусков функции перед измерением времени (по умолчанию 10)
  - `repeats`: количество повторений измерения времени (по умолчанию 100)
    
- Поддержка CPU и GPU: автоматически определяет доступность CUDA

Точные измерения:

 - Для GPU использует torch.cuda.Event (более точное измерение)
 - Для CPU использует time.time()
 - Прогрев (warmup): исключает влияние начальной инициализации
 - Многократные замеры: для получения стабильных результатов
 - Статистика: выводит среднее, минимальное и максимальное время

```python
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
```

### 3.3 Сравнение операций
```python
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
```
- Вывод
```Python
Операция                  | CPU (мс)   | GPU (мс)   | Ускорение 
----------------------------------------------------------------------
Матричное умножение       | 4.08       | 4.35       | 0.94x
Поэлементное сложение     | 0.14       | 0.14       | 1.00x
Поэлементное умножение    | 0.14       | 0.14       | 1.01x
Транспонирование          | 0.00       | 0.00       | 1.08x
Сумма всех элементов      | 0.09       | 0.09       | 0.96x
```

### 3.4 Анализ результатов
- Какие операции получают наибольшее ускорение на GPU?

Из таблицы видно, что никакая операция не получила значительного ускорения на GPU. Напротив:
1. Матричное умножение даже замедлилось на GPU (0.94x).
2. Остальные операции (кроме суммы) показали почти одинаковое время на CPU и GPU (ускорение ~1x).

В данном тесте GPU не дал преимущества.

- Почему некоторые операции могут быть медленнее на GPU?

Некоторые операции могут быть медленнее на GPU по следующим причинам:

1. Задержки передачи: Время, затрачиваемое на передачу данных и инициализацию контекста GPU, может превышать время вычислений для малых задач.
2. Накладные расходы: Для работы на GPU данные нужно скопировать из CPU в GPU (и обратно), что требует времени. Для маленьких матриц это может перевесить выгоду от ускорения вычислений.
3. Параллелизация: GPU эффективен, когда задача может быть разбита на тысячи потоков. Для маленьких матриц или простых операций (поэлементное сложение) CPU может справиться быстрее за счет более высокой тактовой частоты.
4. Оптимизация: Библиотеки для CPU очень хорошо оптимизированы для небольших матриц.

- Как размер матриц влияет на ускорение?

Маленькие матрицы (например, 100x100):

1. CPU часто быстрее, так как GPU не успевает "раскрутиться".
2. Накладные расходы на передачу данных значительны.

Большие матрицы (например, 10000x10000):

1. GPU начинает выигрывать за счет массового параллелизма.
2. Время копирования данных становится незначительным по сравнению с временем вычислений.
3. Матричное умножение (тяжелая операция) ускоряется в десятки/сотни раз.

- Что происходит при передаче данных между CPU и GPU?

При передаче данных между CPU и GPU происходят следующие процессы:
1. Копирование данных: При вызове операции на GPU (например, tensor.cuda()) данные копируются из оперативной памяти (CPU) в память GPU. Это занимает время.
2. Обратная передача: Если результат нужен на CPU (например, для вывода или сохранения), данные копируются обратно.
3. Накладные расходы: Для маленьких данных это может быть дольше, чем само вычисление.

### Вывод
- GPU не всегда быстрее CPU – для маленьких данных или простых операций CPU может выигрывать.
- Матричное умножение – главный кандидат на ускорение, но только для больших матриц.
- Поэлементные операции (сложение, умножение) почти не ускоряются на GPU, так как они слишком "легкие" для параллелизации.
- Размер имеет значение – GPU показывает преимущество только на больших матрицах (от тысяч элементов).
