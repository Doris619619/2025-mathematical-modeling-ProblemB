import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import math
import sympy as sp
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
def load_data(file_path):
    df = pd.read_excel(file_path)
    wavenumber = df['波数 (cm-1)'].values
    reflectance = df['反射率 (%)'].values
    return wavenumber, reflectance

# 数据预处理函数
def preprocess_reflectance(wavenumber, reflectance, visualize=False):
    """
    对反射率数据进行清洗和预处理
    """
    # 1. 中值滤波去除尖峰噪声
    reflectance_clean = median_filter(reflectance, size=5)
    
    # 2. Savitzky-Golay平滑
    reflectance_smooth = savgol_filter(reflectance_clean, window_length=11, polyorder=3)
    
    # 3. 简单的基线校正（减去最小值）
    reflectance_baseline = reflectance_smooth - np.min(reflectance_smooth)
    
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(wavenumber, reflectance, 'purple', alpha=0.6, label='原始数据')
        plt.plot(wavenumber, reflectance_clean, 'orange', alpha=0.7, label='中值滤波')
        plt.plot(wavenumber, reflectance_smooth, 'green', alpha=0.7, label='平滑后')
        plt.plot(wavenumber, reflectance_baseline, 'red', linewidth=1.5, label='最终处理')
        
        # 使用AMPD算法检测峰值
        peaks = AMPD(reflectance_baseline)
        
        # 在图像上标注峰值点
        plt.scatter(wavenumber[peaks], reflectance_baseline[peaks], 
                   color='blue', marker='x', s=50, zorder=5, 
                   label=f'检测到的峰值 ({len(peaks)}个)')
        
        # 为每个峰值添加数值标注
        for i, peak in enumerate(peaks):
            plt.annotate(f'{wavenumber[peak]:.1f}', 
                        xy=(wavenumber[peak], reflectance_baseline[peak]),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.xlabel('波数 (1/cm)')
        plt.ylabel('反射率 (%)')
        plt.title('光谱数据预处理过程及峰值检测')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return reflectance_baseline

# AMPD波峰检测函数
def AMPD(data):
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]

# Sellmeier方程计算折射率
def calculate_refractive_index(wavelength):
    """
    计算碳化硅在特定波长下的折射率
    wavelength: 波长 (cm)
    """
    λ = sp.symbols('lambda')
    term1 = (0.20075 * λ**2) / (λ**2 + 12.07224)
    term2 = (5.54861 * λ**2) / (λ**2 - 0.02641)
    term3 = (35.65066 * λ**2) / (λ**2 - 1268.24708)
    n_squared_minus_1 = term1 + term2 + term3
    n_squared = n_squared_minus_1 + 1
    n_expression = sp.sqrt(n_squared)
    
    # 将波长从cm转换为μm
    wavelength_um = wavelength * 10000
    n_value = n_expression.subs(λ, wavelength_um)
    return float(n_value)

# 提高计算效率，创建折射率计算的向量化版本
calculate_refractive_index_vec = np.vectorize(calculate_refractive_index)

# 干涉模型函数
def interference_model(v, a0, a1, a2, b0, b1, b2, p0, p1, p2, d, theta_deg):
    """
    干涉反射率模型
    v: 波数 (cm⁻¹)
    d: 薄膜厚度 (cm)
    theta_deg: 入射角 (度)
    """
    # 将角度转换为弧度
    theta_rad = np.deg2rad(theta_deg)
    
    # 缓变函数 A(v), B(v), Psi(v)
    A_v = a0 + a1 * v + a2 * (v**2)
    B_v = b0 + b1 * v + b2 * (v**2)
    Psi_v = p0 + p1 * v + p2 * (v**2)
    
    wavelength = 1 / v
    
    # 计算折射率 n(v) 
    n_v = calculate_refractive_index_vec(wavelength)
    
    # 计算主相位差 Δ(v)
    Delta_v = 4 * np.pi * n_v * d * np.cos(theta_rad) * v
    
    # 计算总相位并返回反射率 R(v)
    total_phase = Delta_v + Psi_v
    R_v = B_v + A_v * np.cos(total_phase)
    
    return R_v

# 包装函数用于curve_fit
def model_wrapper(v, a0, a1, a2, b0, b1, b2, p0, p1, p2):
    return interference_model(v, a0, a1, a2, b0, b1, b2, p0, p1, p2, d_fixed, theta_fixed)

def main():
    global d_fixed, theta_fixed
    
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 构建文件路径
    file_path1 = current_dir / 'test1.xlsx'
    file_path2 = current_dir / 'test2.xlsx'
    
    # 已知厚度和入射角
    d1 = 0.00080074  # cm (文件1的厚度)
    d2 = 0.00080396  # cm (文件2的厚度)
    theta1 = 10      # 度 (文件1的入射角)
    theta2 = 15      # 度 (文件2的入射角)
    
    # 处理文件1
    print("处理文件1 (入射角 10°)...")
    wavenumber1, reflectance1 = load_data(file_path1)
    process_file(wavenumber1, reflectance1, d1, theta1, "文件1")
    
    # 处理文件2
    print("\n处理文件2 (入射角 15°)...")
    wavenumber2, reflectance2 = load_data(file_path2)
    process_file(wavenumber2, reflectance2, d2, theta2, "文件2")

def process_file(wavenumber, reflectance, d, theta, file_name):
    global d_fixed, theta_fixed
    d_fixed = d
    theta_fixed = theta
    reflectance_processed = preprocess_reflectance(wavenumber, reflectance, visualize=True)
    peaks = AMPD(reflectance_processed)
    
    # 绘制原始光谱和检测到的波峰
    plot_spectrum_with_peaks(wavenumber, reflectance_processed, peaks, file_name, theta)
    
    if len(peaks) >= 4:
        fourth_peak_idx = peaks[3]
        start_idx = fourth_peak_idx
        print(f"找到第4个波峰在索引 {fourth_peak_idx}, 波数 {wavenumber[fourth_peak_idx]:.1f} cm⁻¹")
    else:
        start_idx = int(len(wavenumber) * 0.3)  # 如果没有找到足够波峰，使用30%作为起始点
        print(f"未找到足够波峰，从索引 {start_idx} 开始")
    
    # 截取数据（从第四个波峰开始）
    wavenumber_cut = wavenumber[start_idx:]
    reflectance_cut = reflectance_processed[start_idx:]  # 使用预处理后的数据
    
    #随机划分训练集(80%)和测试集(20%)
    wavenumber_train, wavenumber_test, reflectance_train, reflectance_test = train_test_split(
        wavenumber_cut, reflectance_cut, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {len(wavenumber_train)}, 测试集大小: {len(wavenumber_test)}")
    
    # 初始参数猜测
    initial_guess = [
        (np.max(reflectance_train) - np.min(reflectance_train)) / 2, 0, 0,  # A(v)系数: a0, a1, a2
        np.mean(reflectance_train), 0, 0,  # B(v)系数: b0, b1, b2
        0, 0, 0  # Psi(v)系数: p0, p1, p2
    ]
    
    print("开始拟合模型...")
    try:
        # 执行拟合
        popt, pcov = curve_fit(model_wrapper, wavenumber_train, reflectance_train, 
                              p0=initial_guess, maxfev=5000)
        
        # 提取拟合参数
        a0, a1, a2, b0, b1, b2, p0, p1, p2 = popt
        print("拟合成功!")
        print(f"A(v)参数: a0={a0:.4f}, a1={a1:.6f}, a2={a2:.9f}")
        print(f"B(v)参数: b0={b0:.4f}, b1={b1:.6f}, b2={b2:.9f}")
        print(f"Psi(v)参数: p0={p0:.4f}, p1={p1:.6f}, p2={p2:.9f}")
        
        #使用测试集验证模型
        reflectance_pred = model_wrapper(wavenumber_test, *popt)
        
        # 计算评估指标
        r2 = r2_score(reflectance_test, reflectance_pred)
        rmse = np.sqrt(mean_squared_error(reflectance_test, reflectance_pred))
        mae = np.mean(np.abs(reflectance_test - reflectance_pred))
        
        print(f"\n模型验证结果:")
        print(f"R²分数: {r2:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        #可视化结果
        plot_results(wavenumber, reflectance_processed, wavenumber_train, reflectance_train,
                    wavenumber_test, reflectance_test, popt, d, theta, file_name, 
                    start_idx, r2, rmse)
        
    except Exception as e:
        print(f"拟合失败: {e}")
        import traceback
        traceback.print_exc()

def plot_spectrum_with_peaks(wavenumber, reflectance, peaks, file_name, theta):
    """
    绘制原始光谱和检测到的波峰位置
    """
    plt.figure(figsize=(12, 6))
    plt.plot(wavenumber, reflectance, 'b-', linewidth=1, label='反射率光谱')
    
    # 标记波峰位置
    peak_wavenumbers = wavenumber[peaks]
    peak_reflectance = reflectance[peaks]
    plt.plot(peak_wavenumbers, peak_reflectance, 'ro', markersize=8, 
             label=f'检测到的波峰 (共{len(peaks)}个)')
    
    # 在波峰位置添加波数值标注
    for i, (w, r) in enumerate(zip(peak_wavenumbers, peak_reflectance)):
        plt.text(w, r + 0.02, f'{w:.1f}', ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.xlabel('波数 (1/cm)')
    plt.ylabel('反射率 (%)')
    plt.title(f'{file_name} - 入射角 {theta}° 光谱波峰检测结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_results(wavenumber_full, reflectance_full, wavenumber_train, reflectance_train,
                wavenumber_test, reflectance_test, popt, d, theta, file_name, 
                start_idx, r2, rmse):
    """
    绘制拟合和验证结果
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 图1: 完整光谱和截取位置
    ax1.plot(wavenumber_full, reflectance_full, 'b-', alpha=0.7, label='完整光谱')
    ax1.axvline(x=wavenumber_full[start_idx], color='r', linestyle='--', 
               label=f'截取起始点: {wavenumber_full[start_idx]:.1f} cm⁻¹')
    ax1.set_xlabel('波数 (1/cm)')
    ax1.set_ylabel('反射率 (%)')
    ax1.set_title(f'{file_name} - 完整光谱和截取位置\n厚度: {d:.6f} cm, 入射角: {theta}°')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 训练集拟合结果
    wavenumber_sorted = np.sort(wavenumber_train)
    reflectance_train_sorted = reflectance_train[np.argsort(wavenumber_train)]
    reflectance_pred_train = model_wrapper(wavenumber_sorted, *popt)
    
    ax2.plot(wavenumber_sorted, reflectance_train_sorted, 'bo', alpha=0.6, 
            markersize=3, label='训练数据')
    ax2.plot(wavenumber_sorted, reflectance_pred_train, 'r-', linewidth=1.5, 
            label='拟合曲线')
    ax2.set_xlabel('波数 (1/cm)')
    ax2.set_ylabel('反射率 (%)')
    ax2.set_title('训练集拟合结果')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 测试集验证结果
    wavenumber_test_sorted = np.sort(wavenumber_test)
    reflectance_test_sorted = reflectance_test[np.argsort(wavenumber_test)]
    reflectance_pred_test = model_wrapper(wavenumber_test_sorted, *popt)
    
    ax3.plot(wavenumber_test_sorted, reflectance_test_sorted, 'go', alpha=0.6, 
            markersize=4, label='测试数据')
    ax3.plot(wavenumber_test_sorted, reflectance_pred_test, 'r-', linewidth=1.5, 
            label='预测曲线')
    ax3.set_xlabel('波数 (1/cm)')
    ax3.set_ylabel('反射率 (%)')
    ax3.set_title(f'测试集验证结果\nR*R = {r2:.4f}, RMSE = {rmse:.4f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 残差分析
    residuals = reflectance_test - model_wrapper(wavenumber_test, *popt)
    ax4.hist(residuals, bins=30, alpha=0.7, color='purple')
    ax4.axvline(x=0, color='r', linestyle='--')
    ax4.set_xlabel('残差')
    ax4.set_ylabel('频数')
    ax4.set_title('测试集残差分布')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印拟合的参数函数
    a0, a1, a2, b0, b1, b2, p0, p1, p2 = popt
    print(f"\n拟合的函数表达式:")
    print(f"A(ν) = {a0:.6f} + {a1:.9f}·ν + {a2:.12f}·ν²")
    print(f"B(ν) = {b0:.6f} + {b1:.9f}·ν + {b2:.12f}·ν²")
    print(f"ψ(ν) = {p0:.6f} + {p1:.9f}·ν + {p2:.12f}·ν²")

if __name__ == '__main__':
    main()

