# -*- coding: utf-8 -*-
"""
@File:   generate_and_plot_frequency_response.py
@Time:   2025/07/28
@Author: Gemini AI
@Description: 
1.  生成10组代表电力系统的二阶传递函数模型。
2.  在其中3组模型中，使用低阻尼比(zeta)来模拟宽频振荡特性。
3.  为10个系统绘制对比的伯德图(Bode Plots)和奈奎斯特图(Nyquist Plots)。
4.  此版本修正了原代码的关注点，从时域信号仿真转向频域响应分析。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import pandas as pd

# --- 1. 参数定义 ---
N_SYSTEMS = 10
N_OSC_SYSTEMS = 3
# 定义分析的频率范围 (0.1 rad/s to 1000 rad/s)
FREQUENCIES = np.logspace(-1, 3, 500)

# --- 2. 系统模型生成 ---
print("开始生成系统传递函数模型...")

systems = []
system_properties = []

# 随机选择3个系统作为“振荡”系统
osc_system_indices = np.random.choice(range(N_SYSTEMS), N_OSC_SYSTEMS, replace=False)
print(f"指定的振荡系统编号: {sorted(osc_system_indices)}")

for i in range(N_SYSTEMS):
    is_oscillatory = i in osc_system_indices
    
    # 为每个系统设置不同的参数以供对比
    # 随机化自然频率 wn (rad/s)
    wn = np.random.uniform(10, 50) 
    
    if is_oscillatory:
        # 振荡系统具有很低的阻尼比
        zeta = np.random.uniform(0.05, 0.15)
        status = "Poorly Damped (Oscillatory)"
    else:
        # 正常系统具有较好的阻尼
        zeta = np.random.uniform(0.5, 0.8)
        status = "Well Damped (Stable)"
        
    # 定义二阶传递函数: G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
    numerator = [wn**2]
    denominator = [1, 2 * zeta * wn, wn**2]
    system = signal.TransferFunction(numerator, denominator)
    
    systems.append(system)
    system_properties.append({'id': i, 'status': status, 'zeta': zeta, 'wn': wn})

print("系统模型生成完毕。")

# --- 3. 计算频率响应 ---
print("\n正在计算各系统的频率响应...")

all_responses = []
for i, system in enumerate(systems):
    # 计算频率响应
    w_out, H = signal.freqresp(system, w=FREQUENCIES)
    
    # 存储结果
    for j, freq in enumerate(w_out):
        response_data = {
            'system_id': i,
            'status': system_properties[i]['status'],
            'frequency': freq,
            'magnitude_db': 20 * np.log10(np.abs(H[j])),
            'phase_deg': np.angle(H[j], deg=True),
            'real': np.real(H[j]),
            'imag': np.imag(H[j])
        }
        all_responses.append(response_data)

df_response = pd.DataFrame(all_responses)
print("频率响应计算完毕。")


# --- 4. 可视化 ---
print("开始绘图...")
sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 4.1 绘制伯德图 (Bode Plots) ---
print("正在绘制伯德图...")
fig_bode, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig_bode.suptitle('系统频率响应对比 - 伯德图', fontsize=20, y=0.96)

# 绘制幅频响应图
sns.lineplot(data=df_response, x='frequency', y='magnitude_db', hue='system_id', style='status', palette='viridis', ax=axes[0])
axes[0].set_title('Magnitude Response')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_xscale('log')
axes[0].legend(title='System ID / Status', bbox_to_anchor=(1.05, 1), loc='upper left')

# 绘制相频响应图
sns.lineplot(data=df_response, x='frequency', y='phase_deg', hue='system_id', style='status', palette='viridis', legend=False, ax=axes[1])
axes[1].set_title('Phase Response')
axes[1].set_xlabel('Frequency (rad/s)')
axes[1].set_ylabel('Phase (degrees)')
axes[1].set_xscale('log')

plt.tight_layout(rect=[0, 0, 0.85, 0.95])
plt.savefig("bode_plot_comparison.png", dpi=300)
print("伯德图已保存为 bode_plot_comparison.png")


# --- 4.2 绘制奈奎斯特图 (Nyquist Plots) ---
print("\n正在绘制奈奎斯特图...")

g_nyquist = sns.relplot(
    data=df_response,
    x='real', y='imag',
    hue='status',
    col='system_id',
    kind='line',
    col_wrap=5,
    height=3, aspect=1,
    palette=sns.color_palette("colorblind", 2)
)
g_nyquist.fig.suptitle('系统频率响应对比 - 奈奎斯特图', fontsize=20, y=1.03)
g_nyquist.set_titles("System {col_name}")
g_nyquist.set_axis_labels("Real Part", "Imaginary Part")

# 在每个子图上标记关键点 (-1,0)
for i, ax in enumerate(g_nyquist.axes.flatten()):
    ax.grid(True)
    ax.axvline(0, color='gray', lw=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    # 标记临界点 (-1, 0)
    ax.plot([-1], [0], 'r+', markersize=10, markeredgewidth=2)
    # 添加系统参数文本
    props = system_properties[i]
    text = f"$\\zeta={props['zeta']:.2f}$\n$\\omega_n={props['wn']:.1f}$"
    ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("nyquist_plot_comparison.png", dpi=300)
print("奈奎斯特图已保存为 nyquist_plot_comparison.png")

plt.show()