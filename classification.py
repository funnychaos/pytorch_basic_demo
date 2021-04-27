import torch
import matplotlib.pyplot as plt

#数据分类

#step 1 构建数据集
n_data = torch.ones(100, 2)         # 数据的基本形态 构建了100*2的矩阵
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)

print(n_data)
print(x0)
print(y0)
print(x1)
print(y1)