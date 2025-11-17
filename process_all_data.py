import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def process_dataset(filename):
    """处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"正在处理: {filename}")
    print('='*60)

    # 读取数据
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"错误: 无法读取文件 {filename}")
        print(f"错误信息: {e}")
        return

    dataset_name = filename.replace('_hourly.csv', '').replace('.csv', '')

    print(f"\n数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n前5行数据:")
    print(df.head())

    # 统计信息
    print(f"\n统计信息:")
    print(df.describe())

    # 检查缺失值
    print(f"\n缺失值:")
    missing = df.isnull().sum()
    print(missing)
    if missing.sum() > 0:
        print(f"总缺失值: {missing.sum()}")

    # 转换时间列
    datetime_col = None
    for col in df.columns:
        if 'datetime' in col.lower() or 'date' in col.lower():
            datetime_col = col
            df[col] = pd.to_datetime(df[col])
            print(f"\n时间范围:")
            print(f"  开始: {df[col].min()}")
            print(f"  结束: {df[col].max()}")
            print(f"  跨度: {df[col].max() - df[col].min()}")
            break

    # 能耗列分析
    energy_col = None
    for col in df.columns:
        if col != datetime_col and df[col].dtype in ['float64', 'int64']:
            energy_col = col
            break

    if energy_col:
        print(f"\n{energy_col} 统计:")
        print(f"  最小值: {df[energy_col].min():.2f}")
        print(f"  最大值: {df[energy_col].max():.2f}")
        print(f"  平均值: {df[energy_col].mean():.2f}")
        print(f"  中位数: {df[energy_col].median():.2f}")
        print(f"  标准差: {df[energy_col].std():.2f}")

        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{dataset_name} Power Consumption Analysis', fontsize=16)

        # 时间序列图
        if datetime_col:
            axes[0, 0].plot(df[datetime_col], df[energy_col], linewidth=0.5)
            axes[0, 0].set_title('Time Series')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Power (MW)')
            axes[0, 0].grid(True, alpha=0.3)

        # 分布直方图
        axes[0, 1].hist(df[energy_col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribution')
        axes[0, 1].set_xlabel('Power (MW)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # 箱线图
        axes[1, 0].boxplot(df[energy_col].dropna())
        axes[1, 0].set_title('Box Plot')
        axes[1, 0].set_ylabel('Power (MW)')
        axes[1, 0].grid(True, alpha=0.3)

        # 月度平均趋势
        if datetime_col:
            df['Year_Month'] = df[datetime_col].dt.to_period('M')
            monthly_avg = df.groupby('Year_Month')[energy_col].mean()
            axes[1, 1].plot(monthly_avg.index.astype(str), monthly_avg.values, marker='o')
            axes[1, 1].set_title('Monthly Average')
            axes[1, 1].set_xlabel('Year-Month')
            axes[1, 1].set_ylabel('Avg Power (MW)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = f'{dataset_name}_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n图表已保存: {output_file}")

    print(f"\n{filename} 处理完成!")

# 要处理的所有文件
files_to_process = [
    'DAYTON_hourly.csv',
    'DEOK_hourly.csv',
    'DOM_hourly.csv',
    'DUQ_hourly.csv',
    'EKPC_hourly.csv',
    'FE_hourly.csv',
    'NI_hourly.csv',
    'PJME_hourly.csv',
    'PJMW_hourly.csv',
    'PJM_Load_hourly.csv',
    'pjm_hourly_est.csv'
]

if __name__ == '__main__':
    for filename in files_to_process:
        if os.path.exists(filename):
            process_dataset(filename)
        else:
            print(f"警告: 文件不存在 - {filename}")

    print(f"\n{'='*60}")
    print("所有数据集处理完成！")
    print('='*60)
