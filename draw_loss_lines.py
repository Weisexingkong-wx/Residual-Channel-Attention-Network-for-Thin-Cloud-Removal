import matplotlib.pyplot as plt
import numpy as np

iteration1 = []
Loss1 = []
with open('txt文档的绝对路径', 'r') as file:  # 训练中间过程生成的记录验证集损失函数的文档
    for line in file.readlines():
        line = line.strip().split(" ")
        # print(line)
        itera, loss = line[0], line[1]
        itera = int(itera.split(':')[1])
        iteration1.append(itera)
        loss = float(loss.split(':')[1])
        Loss1.append(loss)
        # print(itera,'\n',loss)

# 绘制损失函数曲线图
plt.title('Loss')
# plt.plot(x,y)

plt.plot(iteration1, Loss1, color='cyan', label='eval_loss (N=4)')

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')

# plt.ylim(-1,1)
plt.savefig('保存损失函数曲线.png的绝对路径')
plt.show()
