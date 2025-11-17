import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
print("正在读取 COMED_hourly.csv...")
df = pd.read_csv('COMED_hourly.csv')

print("\n" + "="*50)
print("COMED 数据集基本信息")
print("="*50)

# 1. 数据基本信息
print("\n数据形状:", df.shape)
print("\n列名:", df.columns.tolist())
print("\n数据类型:")
print(df.dtypes)

print("\n前5行数据:")
print(df.head())

# 2. 数据统计信息
print("\n" + "="*50)
print("统计信息")
print("="*50)
print(df.describe())

# 3. 检查缺失值
print("\n" + "="*50)
print("缺失值检查")
print("="*50)
print(df.isnull().sum())

# 4. 转换时间列
if 'Datetime' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print("\n时间范围:")
    print(f"开始时间: {df['Datetime'].min()}")
    print(f"结束时间: {df['Datetime'].max()}")
    print(f"时间跨度: {df['Datetime'].max() - df['Datetime'].min()}")

# 5. 能耗列分析（假设第二列是能耗数据）
energy_col = df.columns[1]
print(f"\n" + "="*50)
print(f"{energy_col} 分析")
print("="*50)
print(f"最小值: {df[energy_col].min()}")
print(f"最大值: {df[energy_col].max()}")
print(f"平均值: {df[energy_col].mean():.2f}")
print(f"中位数: {df[energy_col].median():.2f}")
print(f"标准差: {df[energy_col].std():.2f}")

# 6. 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('COMED 电力消耗分析', fontsize=16)

# 时间序列图
if 'Datetime' in df.columns:
    axes[0, 0].plot(df['Datetime'], df[energy_col], linewidth=0.5)
    axes[0, 0].set_title('时间序列图')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('电力消耗 (MW)')
    axes[0, 0].grid(True, alpha=0.3)

# 分布直方图
axes[0, 1].hist(df[energy_col], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('电力消耗分布')
axes[0, 1].set_xlabel('电力消耗 (MW)')
axes[0, 1].set_ylabel('频次')
axes[0, 1].grid(True, alpha=0.3)

# 箱线图
axes[1, 0].boxplot(df[energy_col])
axes[1, 0].set_title('电力消耗箱线图')
axes[1, 0].set_ylabel('电力消耗 (MW)')
axes[1, 0].grid(True, alpha=0.3)

# 月度平均趋势（如果有日期时间列）
if 'Datetime' in df.columns:
    df['Year_Month'] = df['Datetime'].dt.to_period('M')
    monthly_avg = df.groupby('Year_Month')[energy_col].mean()
    axes[1, 1].plot(monthly_avg.index.astype(str), monthly_avg.values, marker='o')
    axes[1, 1].set_title('月度平均电力消耗')
    axes[1, 1].set_xlabel('年-月')
    axes[1, 1].set_ylabel('平均电力消耗 (MW)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('COMED_analysis.png', dpi=300, bbox_inches='tight')
print("\n可视化图表已保存为: COMED_analysis.png")

print("\n" + "="*50)
print("COMED 数据处理完成！")
print("="*50)
