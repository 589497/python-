import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 注意: 运行此代码前，您可能需要安装 'xlrd' 和 'matplotlib' 库。
# 您可以使用命令: pip install xlrd matplotlib

# --- 1. 数据加载与准备 ---

# 定义Excel文件名和工作表名
excel_file_path = 'demo.xls'
sheet_names = [f'P_{i}' for i in range(1, 11)]

# 创建一个列表来存储所有数据帧
all_data = []

print(f"开始从Excel文件 '{excel_file_path}' 读取数据...")

# 检查文件是否存在
if not os.path.exists(excel_file_path):
    print(f"错误: Excel文件 '{excel_file_path}' 未找到。请确保文件已上传并且名称正确。")
else:
    # 循环读取10个工作表
    for i, sheet in enumerate(sheet_names):
        try:
            # 使用两行作为多级表头来读取指定的工作表
            df = pd.read_excel(excel_file_path, sheet_name=sheet, header=[0, 1])

            # 清理和扁平化多级列名
            new_cols = []
            for col in df.columns:
                if 'Unnamed' in col[1]:
                    new_cols.append(f'{col[0]}')
                else:
                    new_cols.append(f'{col[0]}_{col[1]}')
            df.columns = new_cols

            # 重命名频率列以便于访问
            df = df.rename(columns={'频率(Hz)': 'Frequency'})

            # 添加一个'Sheet'列来标识数据来源
            df['Sheet'] = i + 1
            all_data.append(df)
            print(f"成功加载并处理工作表: {sheet}")

        except Exception as e:
            print(f"处理工作表 {sheet} 时发生错误: {e}")
            continue

    # 将所有数据帧合并成一个
    if not all_data:
        print("未能从Excel文件中加载任何数据，程序将退出。")
    else:
        full_df = pd.concat(all_data, ignore_index=True)

        # --- 2. 数据计算 ---

        y_params = ['Y11', 'Y21', 'Y22', 'Y12']

        print("\n开始计算幅值的dB值...")
        for param in y_params:
            magnitude_col_s = f'{param}_幅值(S)'
            db_col = f'{param}_幅值(dB)'
            if magnitude_col_s in full_df.columns:
                numeric_magnitudes = pd.to_numeric(full_df[magnitude_col_s], errors='coerce')
                positive_magnitudes = numeric_magnitudes[numeric_magnitudes > 0]
                full_df[db_col] = np.nan
                full_df.loc[positive_magnitudes.index, db_col] = 20 * np.log10(positive_magnitudes)
        print("计算完成。")


        # --- 3. 三维绘图 (改进版) ---

        print("\n开始生成改进版的三维伯德图...")
        # 获取一个颜色映射, 'viridis', 'plasma', 'inferno' 都是不错的选择
        cmap = plt.get_cmap('viridis')
        # 创建一个归一化实例，将表格编号 (1-10) 映射到 colormap 的范围 (0-1)
        norm = plt.Normalize(1, 10)

        for param in y_params:
            print(f"正在绘制 {param} 的图像...")

            # --- 幅值图 ---
            fig_mag = plt.figure(figsize=(14, 10))
            ax_mag = fig_mag.add_subplot(111, projection='3d')

            for sheet_num in range(1, 11):
                sheet_data = full_df[full_df['Sheet'] == sheet_num]
                if not sheet_data.empty:
                    # 根据表格编号从颜色映射中获取颜色
                    color = cmap(norm(sheet_num))
                    ax_mag.plot(sheet_data['Frequency'],
                                np.full_like(sheet_data['Frequency'], sheet_num),
                                sheet_data[f'{param}_幅值(dB)'],
                                color=color,
                                linewidth=2.5) # 增加线宽

            ax_mag.set_xlabel('频率 (Hz)', fontsize=12, labelpad=15)
            ax_mag.set_ylabel('表格编号 (顺序)', fontsize=12, labelpad=15)
            ax_mag.set_zlabel('幅值 (dB)', fontsize=12, labelpad=15)
            ax_mag.set_xscale('log')
            ax_mag.set_title(f'三维伯德图 - {param} 幅值', fontsize=18, pad=20)
            # 调整视角以获得更好的可见性, (仰角, 方位角)
            ax_mag.view_init(elev=25, azim=-135)

            # 添加颜色条作为图例
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig_mag.colorbar(sm, ax=ax_mag, shrink=0.6, aspect=12, pad=0.1)
            cbar.set_label('表格编号', rotation=270, labelpad=20, fontsize=12)

            mag_filename = f'{param}_magnitude_3d_plot_v2.png'
            plt.savefig(mag_filename)
            plt.close(fig_mag)
            print(f"  已保存图像: {mag_filename}")


            # --- 相位图 ---
            fig_phase = plt.figure(figsize=(14, 10))
            ax_phase = fig_phase.add_subplot(111, projection='3d')

            for sheet_num in range(1, 11):
                sheet_data = full_df[full_df['Sheet'] == sheet_num]
                if not sheet_data.empty:
                    color = cmap(norm(sheet_num))
                    ax_phase.plot(sheet_data['Frequency'],
                                  np.full_like(sheet_data['Frequency'], sheet_num),
                                  sheet_data[f'{param}_相位(deg)'],
                                  color=color,
                                  linewidth=2.5)

            ax_phase.set_xlabel('频率 (Hz)', fontsize=12, labelpad=15)
            ax_phase.set_ylabel('表格编号 (顺序)', fontsize=12, labelpad=15)
            ax_phase.set_zlabel('相位 (deg)', fontsize=12, labelpad=15)
            ax_phase.set_xscale('log')
            ax_phase.set_title(f'三维伯德图 - {param} 相位', fontsize=18, pad=20)
            ax_phase.view_init(elev=25, azim=-135)

            sm_phase = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm_phase.set_array([])
            cbar_phase = fig_phase.colorbar(sm_phase, ax=ax_phase, shrink=0.6, aspect=12, pad=0.1)
            cbar_phase.set_label('表格编号', rotation=270, labelpad=20, fontsize=12)

            phase_filename = f'{param}_phase_3d_plot_v2.png'
            plt.savefig(phase_filename)
            plt.close(fig_phase)
            print(f"  已保存图像: {phase_filename}")

        print("\n所有改进版图像均已生成！")
