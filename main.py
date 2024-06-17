# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt

# 生成一组随机数据，用于模拟真实的数据
np.random.seed(0)  # 设置随机种子，保证每次运行结果一致
x = np.linspace(0, 10, 100)  # 生成100个等间隔的数，作为自变量
y = 3 * x + 5 + np.random.randn(100) * 0.5  # 生成100个服从正态分布的数，作为因变量，其中3和5是真实的斜率和截距，2是噪声的标准差


# 定义线性回归模型，即y = w * x + b，其中w是斜率，b是截距
def linear_model(x, w, b):
    return w * x + b


# 定义损失函数，即平方误差，即(y_pred - y_true)^2的均值，其中y_pred是模型的预测值，y_true是真实值
def loss_function(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# 定义梯度下降函数，即用当前的参数减去学习率乘以损失函数对参数的偏导数，其中lr是学习率
def gradient_descent(w, b, x, y, lr):
    # 计算当前的预测值
    y_pred = linear_model(x, w, b)
    # 计算当前的损失值
    loss = loss_function(y_pred, y)
    # 计算损失函数对w的偏导数，即2 * x * (y_pred - y_true)的均值
    dw = np.mean(2 * x * (y_pred - y))
    # 计算损失函数对b的偏导数，即2 * (y_pred - y_true)的均值
    db = np.mean(2 * (y_pred - y))
    # 更新w和b，沿着负梯度方向，用当前的参数减去学习率乘以偏导数
    w = w - lr * dw
    b = b - lr * db
    # 返回更新后的w和b，以及当前的损失值
    return w, b, loss


# 初始化w和b，随机生成一个0到10之间的数
w = np.random.uniform(0, 5)
b = np.random.uniform(0, 5)
# 设置学习率，可以根据实际情况调整
lr = 0.005
# 设置迭代次数，可以根据实际情况调整
epochs = 10000
# 创建一个空列表，用于存储每次迭代的损失值
losses = []

# 进行迭代，每次调用梯度下降函数，更新w和b，记录损失值
for i in range(epochs):
    w, b, loss = gradient_descent(w, b, x, y, lr)
    losses.append(loss)
    print(f"Epoch {i + 1}, w = {w:.4f}, b = {b:.4f}, loss = {loss:.4f}")

# 绘制数据点和拟合的直线
plt.scatter(x, y, label="Data")
plt.plot(x, linear_model(x, w, b), color="red", label="Linear Model")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# 绘制损失值随迭代次数的变化
plt.plot(range(1, epochs + 1), losses, color="green", label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()