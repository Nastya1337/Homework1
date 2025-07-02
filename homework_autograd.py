from typing import Union

import torch

# Задание 2.1

# Создаем тензоры с requires_grad=True для автоматического вычисления градиентов
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

print(f'x.requires_grad = {x.requires_grad}')
print(f'y.requires_grad = {y.requires_grad}')
print(f'z.requires_grad = {z.requires_grad}')

# Вычисляем функцию f(x,y,z) = x² + y² + z² + 2xyz
f = x**2 + y**2 + z**2 + 2 * x * y * z
print(f'f (no grad) = {f}, f.requires_grad = {f.requires_grad}')

# Вычисляем градиенты
f.backward()

# Получаем градиенты
grad_x = x.grad
grad_y = y.grad
grad_z = z.grad

print("Вычисленные градиенты:")
print(f"df/dx = {grad_x.item()}")
print(f"df/dy = {grad_y.item()}")
print(f"df/dz = {grad_z.item()}\n")

# Аналитическое вычисление градиентов
def analytical_gradients(x_val, y_val, z_val):
    df_dx = 2*x_val + 2*y_val*z_val
    df_dy = 2*y_val + 2*x_val*z_val
    df_dz = 2*z_val + 2*x_val*y_val
    return df_dx, df_dy, df_dz

# Вычисляем аналитические градиенты
a_grad_x, a_grad_y, a_grad_z = analytical_gradients(x.item(), y.item(), z.item())

print("Аналитические градиенты:")
print(f"df/dx = {a_grad_x}")
print(f"df/dy = {a_grad_y}")
print(f"df/dz = {a_grad_z}\n")

# Проверка совпадения результатов
assert torch.allclose(grad_x, torch.tensor(a_grad_x))
assert torch.allclose(grad_y, torch.tensor(a_grad_y))
assert torch.allclose(grad_z, torch.tensor(a_grad_z))

print("Градиенты совпадают с аналитическими вычислениями!")

# Задание 2.2

# Создаем данны:
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=False)
y_true = torch.tensor([2.5, 4.3, 6.1, 7.8], requires_grad=False)

w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Реализация функции:
def linear_regression(x, w, b):
    return w * x + b

# Реализация MSE loss:
def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

# Вычисляем:
y_pred = linear_regression(x, w, b)
loss = mse_loss(y_pred, y_true)

# Вычисляем градиенты:
loss.backward()

# Получаем градиенты
print(f"Градиент по w: {w.grad.item():.4f}")
print(f"Градиент по b: {b.grad.item():.4f}")

# Аналитическая проверка градиентов:
n = len(x)
analytic_grad_w = (2/n) * torch.sum((y_pred - y_true) * x)
analytic_grad_b = (2/n) * torch.sum(y_pred - y_true)

print("\nАналитические градиенты:")
print(f"∂MSE/∂w: {analytic_grad_w.item():.4f}")
print(f"∂MSE/∂b: {analytic_grad_b.item():.4f}")

# Проверка совпадения:
assert torch.allclose(w.grad, analytic_grad_w, atol=1e-5)
assert torch.allclose(b.grad, analytic_grad_b, atol=1e-5)

print("\nПроверка пройдена: градиенты совпадают!")

# Задание 2.3

# Определяем переменную x как тензор с необходимыми свойствами для автоградирующих функций
x = torch.tensor(2.0, requires_grad=True)

# Определяем составную функцию f(x)
def f(x):
    return torch.sin(x**2 + 1)

# Вычисляем значение функции
y = f(x)

# Вычисляем градиент вручную
# Применяем цепное правило
# df/dx = cos(u) * du/dx, где u = x^2 + 1
# du/dx = 2x
cu = torch.cos(x**2 + 1)  # cos(u)
du_dx = 2 * x            # du/dx
df_dx_manual = cu * du_dx

# Вычисляем автоматический градиент с помощью PyTorch
y.backward()  # Это вычисляет градиент для y, так как x имеет requires_grad=True
df_dx_autograd = x.grad

# Выводим результаты
print(f"Ручной расчет градиента df/dx: {df_dx_manual.item()}")
print(f"Градиент с использованием torch.autograd: df/dx = {df_dx_autograd.item()}")