# -*- coding: utf-8 -*-
"""
Created on Wed May 21 13:01:08 2025

@author: user
"""

import random
a=int(input())
rnd=[random.randint (0,100)for _ in range(a)]
b=0
for num in rnd:
    if num%2==0:
        b+=num
print("Массив:", rnd)
print('Сумма четных чисел:',b)
        
        

