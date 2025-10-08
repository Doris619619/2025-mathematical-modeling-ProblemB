import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from scipy.optimize import curve_fit
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据预处理函数
def preprocess_reflectance(wavenumber, reflectance, visualize=False):
    """
    对反射率数据进行清洗和预处理
    """
    # 1. 中值滤波去除尖峰噪声
    reflectance_clean = medfilt(reflectance, kernel_size=5)
    
    # 2. Savitzky-Golay平滑
    reflectance_smooth = savgol_filter(reflectance_clean, window_length=11, polyorder=3)
    
    # 3. 简单的基线校正（减去最小值）
    reflectance_baseline = reflectance_smooth - np.min(reflectance_smooth)
    
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(wavenumber, reflectance, 'b-', alpha=0.5, label='原始数据', linewidth=1)
        plt.plot(wavenumber, reflectance_smooth, 'r-', linewidth=2, label='预处理后数据')
        plt.xlabel('波数 (cm$^{-1}$)')
        plt.ylabel('反射率')
        plt.legend()
        plt.title('反射率数据预处理')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return reflectance_baseline

# 计算折射角函数
def calculate_refraction_angle(n0, n1, theta_deg):
    """
    计算折射角
    n0: 入射介质折射率 (空气≈1)
    n1: 硅折射率
    theta_deg: 入射角 (度)
    """
    theta_rad = math.radians(theta_deg)
    phi_rad = math.asin(n0 * math.sin(theta_rad) / n1)
    return phi_rad

# 反射率模型函数 R1和R2为波数的二次函数
def reflectance_model(wavenumber, d, R1_a, R1_b, R1_c, R2_a, R2_b, R2_c, delta_a, delta_b, delta_c, n0, n1, theta_deg):
    """
    反射率模型函数（波数版本）
    wavenumber: 波数 (cm⁻¹)
    d: 厚度 (μm)
    R1_a, R1_b, R1_c: R1 = R1_a + R1_b*v + R1_c*v² 的系数
    R2_a, R2_b, R2_c: R2 = R2_a + R2_b*v + R2_c*v² 的系数
    delta_a, delta_b, delta_c: δ = delta_a + delta_b*v + delta_c*v² 的系数
    n0: 空气折射率
    n1: 硅折射率
    theta_deg: 入射角 (度)
    """
    # 转换为SI单位 (m⁻¹)
    wavenumber_si = wavenumber * 100  
    # 计算R1和R2的二次函数形式，并确保在合理范围内
    R1 = R1_a + R1_b * wavenumber_si + R1_c * wavenumber_si**2
    R2 = R2_a + R2_b * wavenumber_si + R2_c * wavenumber_si**2
    
    # 确保R1和R2在合理范围内
    R1 = np.maximum(0.01, np.minimum(0.9, R1))
    R2 = np.maximum(0.01, np.minimum(0.9, R2))
    
    # 计算折射角
    phi_rad = calculate_refraction_angle(n0, n1, theta_deg)
    cos_phi = math.cos(phi_rad)
    
    # 计算几何相位差（使用波数）
    # Δφ = 4π * n * d * cosθ * ν
    geometric_phase = 4 * np.pi * n1 * (d * 1e-6) * cos_phi * wavenumber_si
    
    # 计算附加相位项 δ = delta_a + delta_b*v + delta_c*v²
    delta_phase = delta_a + delta_b * wavenumber_si + delta_c * wavenumber_si**2
    
    # 总相位差 = 几何相位差 + 附加相位项
    delta_phi = geometric_phase + delta_phase
    
    # 计算反射率
    sqrt_R1R2 = np.sqrt(R1 * R2)
    numerator = R1 + R2 + 2 * sqrt_R1R2 * np.cos(delta_phi)
    denominator = 1 + R1 * R2 + 2 * sqrt_R1R2 * np.cos(delta_phi)
    
    return numerator / denominator

def create_fit_function(theta_deg):
    """
    创建带有固定theta_deg的拟合函数
    """
    def fit_function(wavenumber, d, R1_a, R1_b, R1_c, R2_a, R2_b, R2_c, delta_a, delta_b, delta_c):
        n0 = 1.0  # 空气折射率
        n1 = 3.469  # 硅折射率
        return reflectance_model(wavenumber, d, R1_a, R1_b, R1_c, R2_a, R2_b, R2_c, 
                                delta_a, delta_b, delta_c, n0, n1, theta_deg)
    
    return fit_function

# 改进的主程序（包含附加相位项δ）
def main_improved(file_path, theta_deg, visualize=True):
    """
    改进的主拟合程序，包含附加相位项δ
    """
    # 读取数据
    data = pd.read_excel(file_path)
    wavenumber = data.iloc[:, 0].values  # 波数数据，单位cm⁻¹
    reflectance_raw = data.iloc[:, 1].values / 100  # 转换为0-1范围
    
    # 数据预处理
    reflectance = preprocess_reflectance(wavenumber, reflectance_raw, visualize=visualize)
    
    # 8:2随机分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        wavenumber, reflectance, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 创建拟合函数
    fit_func = create_fit_function(theta_deg)
    
    # 初始参数 - 包含附加相位项δ的系数
    initial_guess = [
        4.5,       # d: 厚度 (μm)
        0.3,       # R1_a: R1常数项
        -1e-6,     # R1_b: R1一次项系数
        1e-13,     # R1_c: R1二次项系数
        0.3,       # R2_a: R2常数项
        -1e-7,     # R2_b: R2一次项系数
        1e-13,     # R2_c: R2二次项系数
        1,         # delta_a: δ常数项
        0.0,       # delta_b: δ一次项系数
        0.0        # delta_c: δ二次项系数
    ]
    
    # 参数边界
    bounds = (
        [1.0, 0.1, -1e-6, -1e-12, 0.1, -1e-6, -1e-12, -10.0, -1e-6, -1e-12],  # 下限
        [1000.0, 0.8, 1e-6, 1e-12, 0.8, 1e-6, 1e-12, 10.0, 1e-6, 1e-12]      # 上限
    )
    
    # 拟合
    try:
        print("开始拟合...")
        popt, pcov = curve_fit(
            fit_func, 
            X_train, 
            y_train, 
            p0=initial_guess,
            bounds=bounds,
            maxfev=20000,  # 增加迭代次数
            method='trf'
        )
        
        # 拟合参数的提取
        d_fit, R1_a_fit, R1_b_fit, R1_c_fit, R2_a_fit, R2_b_fit, R2_c_fit, delta_a_fit, delta_b_fit, delta_c_fit = popt
        
        # 所有数据预测值的计算
        wavenumber_sorted = np.sort(wavenumber)
        y_full_pred = fit_func(wavenumber_sorted, *popt)
        
        # 计算R1、R2和δ的函数值
        wavenumber_si = wavenumber_sorted * 100
        R1_values = R1_a_fit + R1_b_fit * wavenumber_si + R1_c_fit * wavenumber_si**2
        R2_values = R2_a_fit + R2_b_fit * wavenumber_si + R2_c_fit * wavenumber_si**2
        delta_values = delta_a_fit + delta_b_fit * wavenumber_si + delta_c_fit * wavenumber_si**2
        
        R1_values = np.maximum(0.01, np.minimum(0.9, R1_values))
        R2_values = np.maximum(0.01, np.minimum(0.9, R2_values))
        
        # 计算训练集和测试集预测
        y_train_pred = fit_func(X_train, *popt)
        y_test_pred = fit_func(X_test, *popt)
        
        # 拟合指标的计算
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"\n拟合结果 - 入射角: {theta_deg}°")
        print(f"厚度 d = {d_fit:.2f} μm")
        print(f"R1系数: a={R1_a_fit:.4f}, b={R1_b_fit:.2e}, c={R1_c_fit:.2e}")
        print(f"R2系数: a={R2_a_fit:.4f}, b={R2_b_fit:.2e}, c={R2_c_fit:.2e}")
        print(f"δ系数: a={delta_a_fit:.4f}, b={delta_b_fit:.2e}, c={delta_c_fit:.2e}")
        print(f"训练集 RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"测试集 RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        
        # 可视化结果
        if visualize:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 主图：完整数据拟合结果
            axes[0, 0].plot(wavenumber, reflectance, 'b.', alpha=0.6, label='所有数据', markersize=2)
            axes[0, 0].plot(wavenumber_sorted, y_full_pred, 'r-', linewidth=2, label='拟合曲线')
            axes[0, 0].set_xlabel('波数 (cm$^{-1}$)')
            axes[0, 0].set_ylabel('反射率')
            axes[0, 0].set_title(f'完整数据拟合结果 (入射角: {theta_deg}°)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # R1和R2函数
            axes[0, 1].plot(wavenumber_sorted, R1_values, 'b-', label='R1(v)')
            axes[0, 1].plot(wavenumber_sorted, R2_values, 'r-', label='R2(v)')
            axes[0, 1].plot(wavenumber_sorted, delta_values, 'g-', label='δ(v)')
            axes[0, 1].set_xlabel('波数 (cm$^{-1}$)')
            axes[0, 1].set_ylabel('系数值')
            axes[0, 1].set_title('R1、R2和δ的函数形式')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 测试集预测
            axes[1, 0].plot(X_test, y_test, 'go', alpha=0.8, label='测试数据', markersize=4)
            axes[1, 0].plot(X_test, y_test_pred, 'ro', alpha=0.6, label='预测值', markersize=3)
            axes[1, 0].set_xlabel('波数 (cm$^{-1}$)')
            axes[1, 0].set_ylabel('反射率')
            axes[1, 0].set_title(f'测试集预测 (R$^2$ = {test_r2:.4f})')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            
        
        return {
            'd': d_fit,
            'R1_a': R1_a_fit, 'R1_b': R1_b_fit, 'R1_c': R1_c_fit,
            'R2_a': R2_a_fit, 'R2_b': R2_b_fit, 'R2_c': R2_c_fit,
            'delta_a': delta_a_fit, 'delta_b': delta_b_fit, 'delta_c': delta_c_fit,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
    except Exception as e:
        print(f"拟合过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 运行两个角度的拟合
if __name__ == "__main__":
    # 文件路径
      # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 构建文件路径
    file_10deg = current_dir / 'test1.xlsx'
    file_15deg = current_dir / 'test2.xlsx'
    
    # 拟合10度数据
    print("=" * 50)
    print("开始拟合10°入射角数据...")
    results_10deg = main_improved(file_10deg, 10, visualize=True)
    
    # 拟合15度数据
    print("=" * 50)
    print("开始拟合15°入射角数据...")
    results_15deg = main_improved(file_15deg, 15, visualize=True)
    
    # 比较两个角度的结果
    if results_10deg and results_15deg:
        print("=" * 50)
        print("两个角度拟合结果比较:")
        print(f"10°厚度: {results_10deg['d']:.2f} μm")
        print(f"15°厚度: {results_15deg['d']:.2f} μm")
        print(f"厚度差异: {abs(results_10deg['d'] - results_15deg['d']):.2f} μm")
        
        # 计算平均厚度
        avg_d = (results_10deg['d'] + results_15deg['d']) / 2
        print(f"平均厚度: {avg_d:.2f} μm")