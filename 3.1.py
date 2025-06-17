# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:30:32 2025

@author: AM4
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Загружаем данные
df = pd.read_csv('data.csv')

# выводим первые строки таблицы
print(df.head())

# Выделяем признаки (берем 3 признака) и целевую переменную
X = df.iloc[:, [0, 2, 3]].values
y1 = df.iloc[:, 4].values

# Переводим целевую переменную в числа: 1 — Iris-setosa, -1 — другие
y1 = np.where(y1 == "Iris-setosa", 1, -1)

# Преобразуем в тензоры
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y1, dtype=torch.float32)
y_tensor = y_tensor.view(-1, 1)  # преобразуем в вектор-столбец

linear = nn.Linear(3, 1)

# Выводим начальные веса и смещения
print('w: ', linear.weight)
print('b: ', linear.bias)

lossFn = nn.MSELoss()

optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Обучающий цикл
for i in range(0, 10):
    # прямой проход
    pred = linear(X_tensor)
    
    # вычисление ошибки
    loss = lossFn(pred, y_tensor)
    print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    # обнуление градиентов
    optimizer.zero_grad()

    # обратное распространение
    loss.backward()

    # обновление весов
    optimizer.step()
