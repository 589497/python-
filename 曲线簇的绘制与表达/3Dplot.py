import matplotlib.pyplot as plt
import numpy as np

# 为了兼容性，最好显式导入
from mpl_toolkits.mplot3d import Axes3D

# 1. 生成数据
t = np.linspace(-4 * np.pi, 4 * np.pi, 200)
z = np.linspace(-2, 2, 200)
r = z**2 + 1
x = r * np.sin(t)
y = r * np.cos(t)

# 2. 创建图窗和3D坐标轴
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection='3d')

# 3. 绘图
ax.plot(x, y, z, label='可交互3D螺旋线')

# 4. 设置标签和标题
ax.set_xlabel('X aexs')
ax.set_ylabel('Y aexs')
ax.set_zlabel('Z aexs')
ax.set_title('用鼠标拖动我来旋转！')
ax.legend()

# 5. 显示图形 (这是关键!)
# 在脚本模式下，plt.show()会启动交互式窗口并等待其关闭
plt.show()