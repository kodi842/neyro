import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Загрузка данных
df = pd.read_csv('dataset_simple.csv')
X = torch.Tensor(df.iloc[:, 0:2].values)  # Возраст и доход
y = df.iloc[:, 2].values  # Купит/не купит
y = torch.Tensor(np.where(y == "купит", 1, -1).reshape(-1, 1))

class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Tanh()
        )
    
    def forward(self, X):
        return self.layers(X)

inputSize = X.shape[1]
hiddenSizes = 3
outputSize = 1

net = NNet(inputSize, hiddenSizes, outputSize)

# Обучение
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

epochs = 100
for i in range(epochs):
    pred = net.forward(X)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f'Ошибка на {i+1} итерации: {loss.item()}')

with torch.no_grad():
    pred = net.forward(X)
pred = torch.Tensor(np.where(pred >= 0, 1, -1).reshape(-1, 1))
err = sum(abs(y - pred)) / 2
print("Ошибка классификации:", err)