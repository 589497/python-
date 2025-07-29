import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def plot_3d_bode(data_dict, param_name, plot_type, p_indices, colors):
    """
    绘制指定参数的3D伯德图（幅值或相位）。

    Args:
        data_dict (dict): 包含所有P值数据帧的字典。
        param_name (str): 要绘制的参数名 ('Y11', 'Y21', etc.)。
        plot_type (str): 'Magnitude' 或 'Phase'。
        p_indices (dict): 将P名称映射到整数索引的字典。
        colors (list): 用于绘图的颜色列表。
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 根据绘图类型确定列名和Y轴标签
    if plot_type == 'Magnitude':
        col_name_part1 = '幅值(S)'
        y_label = 'Magnitude (dB)'
    else: # Phase
        col_name_part1 = '相位(deg)'
        y_label = 'Phase (deg)'

    # 确定完整的列名
    # 在pandas MultiIndex中，列名是一个元组
    freq_col = ('频率(Hz)', 'Unnamed: 0_level_1')
    value_col = (param_name, col_name_part1)

    for i, p_name in enumerate(data_dict.keys()):
        df = data_dict[p_name]
        p_index = p_indices[p_name]

        # 提取频率和数值数据
        freq = df[freq_col].values
        values = df[value_col].values

        # 计算幅值dB
        if plot_type == 'Magnitude':
            # 避免log(0)错误，用一个很小的数替换0或负值
            values[values <= 0] = 1e-9
            plot_values = 20 * np.log10(values)
        else:
            plot_values = values
        
        # Z轴数据，为当前P值的索引
        z_data = np.full_like(freq, p_index)

        # 绘制3D曲线
        ax.plot(np.log10(freq), plot_values, z_data, label=p_name, color=colors[i])

    # 设置坐标轴
    ax.set_xlabel('Log10(Frequency (Hz))')
    ax.set_ylabel(y_label)
    ax.set_zlabel('P-Value Index')
    ax.set_title(f'3D Bode Plot - {param_name} {plot_type}', pad=20)
    ax.legend(title="Sheet Name")

    # 自动调整视角
    ax.view_init(elev=20, azim=-65)
    plt.tight_layout()
    
    # 保存图像文件
    plt.savefig(f'bode_3d_{param_name}_{plot_type}.png')


# --- 主程序 ---
if __name__ == "__main__":
    try:
        # 定义Excel文件名和Sheet名
        excel_file = 'demo.xls'
        sheet_names = [f'P_{i}' for i in range(1, 11)]

        # 读取所有sheet的数据到字典中
        # header=[0, 1] 用于读取多级标题行
        all_data = pd.read_excel(excel_file, sheet_name=sheet_names, header=[0, 1])

        # 定义要处理的阻抗参数
        impedance_params = ['Y11', 'Y21', 'Y22', 'Y12']
        
        # 定义P值名称到整数索引的映射
        p_indices = {name: i for i, name in enumerate(sheet_names, 1)}

        # 创建一个颜色映射，为10条曲线分配不同颜色
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(sheet_names))]

        # 循环生成8个图
        for param in impedance_params:
            # 绘制幅值图
            plot_3d_bode(all_data, param, 'Magnitude', p_indices, colors)
            # 绘制相位图
            plot_3d_bode(all_data, param, 'Phase', p_indices, colors)
        
        print("所有8个三维伯德图已生成并保存为PNG文件。")
        print("现在将以可交互模式显示所有图窗...")
        
        # plt.show()会显示所有已经创建的figure，并进入交互模式
        plt.show()

    except FileNotFoundError:
        print(f"错误: 文件 '{excel_file}' 未在当前目录中找到。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")