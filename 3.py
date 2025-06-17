# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:18:55 2025

@author: user
"""

import torch

x = torch.randint(1, 100, (1,), dtype=torch.int32)
x = x.to(torch.float32)
x.requires_grad = True

n = 2
a = x ** n
b = torch.randint(1, 11, (1,), dtype=torch.float32)
c = a * b
y = torch.exp(c)

y.backward()
print(x.item())
print(b.item())
print(y.item())
print(x.grad.item())
