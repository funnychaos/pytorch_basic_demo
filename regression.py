import torch
import matplotlib.pyplot as plt
import torch.nn.functional as fun       #激励函数


#函数回归问题 通过导入相应的数据，让网路学习如果回归到某函数上

#step 1 构建带噪声的数据集
#此函数表示生成[-1,1]之间共一百个数据的平均分布 torch.linspace(-1, 1, 100)
#此函数表示将之前生成的tensor转换为1维(参数为dim的值)的矩阵

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)         # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)     datas have noise

# 画图
plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()

#step 2 构建网络结构
class Net(torch.nn.Module):

    def __init__(self,n_feature, n_hidden,n_output):
        super(Net, self).__init__()     #继承 __init__ 功能，规定格式

        #自定义网络    隐藏层 神经元个数
        self.hidden = torch.nn.Linear(n_feature,n_hidden)       #相当于定义隐藏层的输入以及输出个数
        self.predict = torch.nn.Linear(n_hidden,n_output)


    #正向传播输入值
    def forward(self,x):
        x_activate = fun.relu(self.hidden(x))   #激励函数为relu函数
        x_pre = self.predict(x_activate)        #输出值
        return x_pre


net = Net(n_feature=1,n_hidden=15,n_output=1)

print(net)  # net 的结构
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

#step 3 定义优化器
optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)  #传入参数为net的所有参数 定义学习率为0.1
loss_func = torch.nn.MSELoss()      #预测值与真实值的误差计算 这里定义的是均方差


plt.ion()   # 画图
plt.show()


for i in range(300):
    prediction = net(x)     #训练网络 输入x值
    loss = loss_func(prediction,y)     #根据预测值和y真实值计算loss值
    optimizer.zero_grad()       #梯度赋值为0
    loss.backward()             #反向传播 更新参数值
    optimizer.step()            #将更新net上的参数

    # 可视化操作
    if i % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.close()

print()





