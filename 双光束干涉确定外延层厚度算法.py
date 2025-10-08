import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
import sympy as sp
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import os
from pathlib import Path


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
def load_data(file_path):
    df = pd.read_excel(file_path)
    wavenumber = df['波数 (cm-1)'].values
    reflectance = df['反射率 (%)'].values
    return wavenumber, reflectance

def preprocess_reflectance(wavenumber, reflectance, visualize=False):
    """
    对反射率数据进行清洗和预处理
    """
    # 1. 中值滤波去除尖峰噪声
    reflectance_clean = median_filter(reflectance, size=5)
    
    # 2. Savitzky-Golay平滑
    reflectance_smooth = savgol_filter(reflectance_clean, window_length=11, polyorder=3)
    
    # 3. 简单的基线校正
    reflectance_baseline = reflectance_smooth - np.min(reflectance_smooth)
    
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(wavenumber, reflectance, 'purple', alpha=0.6, label='原始数据')
        plt.plot(wavenumber, reflectance_clean, 'orange', alpha=0.7, label='中值滤波')
        plt.plot(wavenumber, reflectance_smooth, 'green', alpha=0.7, label='平滑后')
        plt.plot(wavenumber, reflectance_baseline, 'red', linewidth=1.5, label='最终处理')
        
        # AMPD算法检测峰值
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

# 斯涅尔定律计算折射角
def calculate_refraction_angle(n0, n1, theta):
    theta_rad = math.radians(theta)
    phi_rad = math.asin(n0 * math.sin(theta_rad) / n1)
    return phi_rad

# 定义Sellmeier方程的折射率计算
def calculate_refractive_index(wavelength):
    λ = sp.symbols('lambda')
    term1 = (0.20075 * λ**2) / (λ**2 + 12.07224)
    term2 = (5.54861 * λ**2) / (λ**2 - 0.02641)
    term3 = (35.65066 * λ**2) / (λ**2 - 1268.24708)
    n_squared_minus_1 = term1 + term2 + term3
    n_squared = n_squared_minus_1 + 1
    n_expression = sp.sqrt(n_squared)
    n_value = n_expression.subs(λ, wavelength*10000)
    return float(n_value)

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

# 计算厚度（剔除异常值的平均厚度）
def calculate_thickness(wavenumbers, reflectance, n0=1, theta=10):
    # 数据预处理 
    reflectance_processed = preprocess_reflectance(wavenumbers, reflectance, visualize=True)
    peaks = AMPD(reflectance_processed)
    peak_wavenumbers = wavenumbers[peaks]
    print("波峰位置的波数：", peak_wavenumbers)
    peak_wavenumbers = peak_wavenumbers[1:]

    thicknesses = []
    thickness_details = []
    for i in range(1, len(peak_wavenumbers)):
        wavelength = 1 / peak_wavenumbers[i]
        n1 = calculate_refractive_index(wavelength)
        print(f"波长 {wavelength:.6f} cm 处的折射率: {n1:.6f}")
        phi = calculate_refraction_angle(n0, n1, theta)
        delta_v = peak_wavenumbers[i] - peak_wavenumbers[i - 1]
        thickness = 1 / (2 * n1 * math.cos(phi) * delta_v)
        thicknesses.append(thickness)
        thickness_details.append({
            'peak_pair': (i-1, i),
            'wavenumber1': peak_wavenumbers[i-1],
            'wavenumber2': peak_wavenumbers[i],
            'delta_v': delta_v,
            'wavelength': wavelength,
            'refractive_index': n1,
            'refraction_angle': math.degrees(phi),
            'thickness': thickness
        })

    if thicknesses:
        avg_thickness = np.mean(thicknesses)#箱线图法剔除异常值 
        Q1 = np.percentile(thicknesses, 25)
        Q3 = np.percentile(thicknesses, 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filtered = [t for t in thicknesses if lower <= t <= upper]
        if filtered:
            avg_thickness_filtered = np.mean(filtered)
        else:
            avg_thickness_filtered = avg_thickness
        return avg_thickness, avg_thickness_filtered, thickness_details, lower, upper  # 新增返回边界值
    else:
        return None, None, None, None, None

#绘制原始光谱和检测到的波峰 
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

# 可视化厚度结果（异常值显示红色,其他显示蓝色）
def plot_thickness_results(thickness_details, file_name, avg_raw, avg_filtered, lower, upper):
    if not thickness_details:
        return
    
    peak_pairs = [f"{i+1}" for i in range(len(thickness_details))]
    thickness_values = [detail['thickness'] for detail in thickness_details]
    
    # 判断哪些是异常值（红色），哪些是正常值（天蓝色）
    colors = ['red' if not (lower <= t <= upper) else 'skyblue' for t in thickness_values]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    bars = ax1.bar(peak_pairs, thickness_values, alpha=0.7, color=colors)  # 使用不同颜色
    ax1.axhline(y=avg_raw, color='red', linestyle='--', label=f'原始均值: {avg_raw:.6f} cm')
    ax1.axhline(y=avg_filtered, color='green', linestyle='--', label=f'去异常均值: {avg_filtered:.6f} cm')
    ax1.axhline(y=lower, color='orange', linestyle=':', alpha=0.7, label=f'下边界: {lower:.6f} cm')
    ax1.axhline(y=upper, color='purple', linestyle=':', alpha=0.7, label=f'上边界: {upper:.6f} cm')
    
    ax1.set_xlabel('相邻波峰对编号')
    ax1.set_ylabel('厚度 (cm)')
    ax1.set_title(f'{file_name} - 各相邻波峰对厚度值 (红色为异常值)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for bar, value in zip(bars, thickness_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(thickness_values)*0.01,
                f'{value:.6f}', ha='center', va='bottom', fontsize=8)

    ax2.boxplot(thickness_values, vert=True)
    ax2.scatter([1] * len(thickness_values), thickness_values, alpha=0.6, color='red')
    ax2.set_ylabel('厚度 (cm)')
    ax2.set_title(f'{file_name} - 厚度值分布 (箱线图)')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['厚度值'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 主函数
def main():
     # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 构建文件路径
    file_path1 = current_dir / 'test1.xlsx'
    file_path2 = current_dir / 'test2.xlsx'

    wavenumber1, reflectance1 = load_data(file_path1)
    wavenumber2, reflectance2 = load_data(file_path2)

    print("\n文件1的厚度计算 (入射角 10°):")
    reflectance1_processed = preprocess_reflectance(wavenumber1, reflectance1, visualize=False)
    peaks1 = AMPD(reflectance1_processed)
    # 绘制原始光谱和波峰
    plot_spectrum_with_peaks(wavenumber1, reflectance1, peaks1, "test1.xlsx", 10)
    avg1_raw, avg1_filtered, details1, lower1, upper1 = calculate_thickness(wavenumber1, reflectance1, theta=10)
    print(f"文件1原始平均厚度 = {avg1_raw:.8f} cm, 去异常平均厚度 = {avg1_filtered:.8f} cm")
    plot_thickness_results(details1, "test1.xlsx (入射角 10°)", avg1_raw, avg1_filtered, lower1, upper1)

    print("\n文件2的厚度计算 (入射角 15°):")
    reflectance2_processed = preprocess_reflectance(wavenumber2, reflectance2, visualize=False)
    peaks2 = AMPD(reflectance2_processed)
    # 绘制原始光谱和波峰
    plot_spectrum_with_peaks(wavenumber2, reflectance2, peaks2, "test2.xlsx", 15)
    
    
    avg2_raw, avg2_filtered, details2, lower2, upper2 = calculate_thickness(wavenumber2, reflectance2, theta=15)
    print(f"文件2原始平均厚度 = {avg2_raw:.8f} cm, 去异常平均厚度 = {avg2_filtered:.8f} cm")
    plot_thickness_results(details2, "test2.xlsx (入射角 15°)", avg2_raw, avg2_filtered, lower2, upper2)

    if details1 and details2:
        print("\n两个入射角结果的比较:")
        print("=" * 50)
        print(f"入射角 10° 的原始平均厚度: {avg1_raw:.8f} cm, 去异常: {avg1_filtered:.8f} cm")
        print(f"入射角 15° 的原始平均厚度: {avg2_raw:.8f} cm, 去异常: {avg2_filtered:.8f} cm")

if __name__ == '__main__':
    main()