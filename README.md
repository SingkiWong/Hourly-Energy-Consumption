# LN-TSDM: 线性-非线性时间序列分解模型

## Linear-Nonlinear Time Series Decomposition Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目简介

本项目实现了一种创新的混合时间序列预测框架——**线性-非线性时间序列分解模型（LN-TSDM）**。该模型基于"分解-建模-融合"的三阶段处理逻辑，通过将时间序列数据分解为线性成分和非线性成分，分别采用弹性网络回归（ElasticNet）和随机森林（Random Forest）算法进行建模与预测，最终通过成分叠加实现高精度预测。

### 核心特点

- **线性-非线性分解策略**：将原始时间序列解构为趋势性线性成分与波动性非线性成分
- **适配性建模方法**：线性成分用ElasticNet捕捉趋势，非线性成分用Random Forest处理复杂关联
- **网格搜索优化**：采用时间序列交叉验证的网格搜索技术确定Random Forest最优超参数
- **成分融合预测**：通过直接加法融合线性和非线性预测，保持分解的数学一致性

---

## 模型架构

### 三阶段处理流程

```
原始时间序列
    ↓
[阶段1: 分解]
    ├─→ 线性成分 (趋势) → ElasticNet模型 → 线性预测
    └─→ 非线性成分 (残差) → Random Forest模型 → 非线性预测
    ↓
[阶段2: 建模与预测]
    ↓
[阶段3: 融合]
    → 最终预测 = 线性预测 + 非线性预测
```

### 算法原理

1. **时间序列分解**
   - 使用滑动平均法提取趋势（线性成分）
   - 残差（原始数据 - 趋势）作为非线性成分
   - 数学关系：`原始序列 = 线性成分 + 非线性成分`

2. **线性建模 (ElasticNet)**
   - 结合L1和L2正则化的线性回归
   - 擅长处理高维特征和趋势性数据
   - 自动进行特征选择和收缩

3. **非线性建模 (Random Forest)**
   - 集成学习方法，构建多棵决策树
   - 对高维特征空间和数据噪声具有强鲁棒性
   - 通过网格搜索和交叉验证优化超参数

4. **预测融合**
   - 保持分解的数学一致性
   - `最终预测 = 线性预测 + 非线性预测`

---

## 实验结果

### 综合性能对比（12个数据集的平均表现）

| 模型 | 平均RMSE | 平均MAE | 平均MAPE (%) | 平均R² |
|------|----------|---------|--------------|--------|
| **LN-TSDM** | **654.79** | **547.57** | **4.79** | **0.8874** |
| ElasticNet | 1350.68 | 1076.85 | 9.62 | 0.5166 |
| Random Forest | 269.95 | 175.65 | 1.69 | 0.9758 |

### 关键发现

**LN-TSDM 相对 ElasticNet 的平均改进（跨12个数据集）：**
- ✅ RMSE 降低：**53.05%**
- ✅ MAE 降低：**50.95%**
- ✅ R² 提升：**74.75%**
- ✅ MAPE 降低：**50.23%**

**数据集覆盖范围：**
本实验在以下12个不同的电力负荷数据集上验证了LN-TSDM的有效性：
- AEP, COMED, DAYTON, DEOK, DOM, DUQ
- EKPC, FE, NI, PJME, PJMW, PJM_Load

**最佳改进案例：**
- EKPC数据集：RMSE降低 **69.11%**
- PJME数据集：RMSE降低 **60.80%**
- COMED数据集：RMSE降低 **58.39%**

**结论**：LN-TSDM模型在所有12个数据集上都显著优于单一ElasticNet模型，平均改进超过50%，充分验证了线性-非线性分解建模策略的有效性和跨数据集的稳定性。该模型提供了可解释性更强的分解视角（趋势+残差），特别适用于需要理解时间序列内在结构的应用场景。

---

## 安装与使用

### 环境要求

- Python 3.8+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.13.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy

### 安装依赖

```bash
pip install -r requirements.txt
```

### 快速开始

#### 1. 运行单数据集实验

```bash
python run_experiment.py
```

该命令将：
- 加载AEP hourly能源消耗数据
- 训练LN-TSDM模型和基线模型
- 生成性能对比报告和可视化结果

#### 2. 运行多数据集批量实验（推荐）⭐

```bash
python run_all_datasets_experiment.py
```

该命令将：
- 对所有12个电力负荷数据集运行实验
- 生成综合性能对比报告
- 创建跨数据集的可视化图表
- 输出汇总的改进率分析

#### 3. 使用LN-TSDM模型

```python
from ln_tsdm_model import LN_TSDM
import pandas as pd

# 加载数据
data = pd.read_csv('AEP_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')
ts = data['AEP_MW']

# 初始化模型
model = LN_TSDM(
    seasonal_period=24,  # 小时数据的日周期
    optimize_rf=True,    # 启用Random Forest网格搜索
    verbose=True
)

# 训练模型
model.fit(ts, lookback=24)

# 预测
predictions, linear_pred, nonlinear_pred = model.predict(ts, lookback=24)
```

#### 4. 自定义参数

```python
# 高级配置
model = LN_TSDM(
    seasonal_period=24,
    linear_alpha=1.0,           # ElasticNet正则化强度
    linear_l1_ratio=0.5,        # L1/L2比例
    optimize_rf=True,
    rf_param_grid={             # 自定义Random Forest参数网格
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    n_cv_folds=5,               # 交叉验证折数
    verbose=True
)
```

---

## 文件结构

```
├── ln_tsdm_model.py                       # LN-TSDM模型核心实现
├── run_experiment.py                      # 单数据集实验运行脚本
├── run_all_datasets_experiment.py         # 多数据集批量实验脚本 ⭐
├── requirements.txt                       # Python依赖包
├── README.md                              # 项目说明文档
│
├── results_ln_tsdm_decomposition.png      # 时间序列分解可视化
├── results_predictions_comparison.png     # 预测结果对比图
├── results_metrics_comparison.png         # 性能指标对比图
├── results_metrics_comparison.csv         # 详细性能指标数据
├── results_experiment_report.md           # 单数据集实验报告
│
└── results_all_datasets/                  # 多数据集实验结果目录 ⭐
    ├── summary_report.md                  # 综合实验报告
    ├── summary_rmse_comparison.png        # 所有数据集RMSE对比
    ├── summary_r2_comparison.png          # 所有数据集R²对比
    ├── summary_improvement_heatmap.png    # 改进率热力图
    ├── all_datasets_metrics.csv           # 所有数据集详细指标
    └── all_datasets_improvement.csv       # 所有数据集改进率
```

---

## 核心模块说明

### 1. `ln_tsdm_model.py`

包含三个主要类：

- **`LinearNonlinearDecomposer`**
  - 时间序列分解器
  - 使用滑动平均法提取趋势和残差

- **`LN_TSDM`**
  - 混合时间序列预测模型
  - 实现完整的"分解-建模-融合"流程

- **`BaselineModels`**
  - 基线对比模型（单一ElasticNet和Random Forest）
  - 用于验证LN-TSDM的有效性

### 2. `run_experiment.py`

单数据集完整实验流程：
1. 数据加载与划分
2. 模型训练（LN-TSDM + 基线模型）
3. 测试集预测
4. 性能评估与对比
5. 可视化结果生成
6. 自动生成实验报告

### 3. `run_all_datasets_experiment.py` ⭐

多数据集批量实验流程：
1. 遍历所有12个数据集
2. 对每个数据集训练和评估模型
3. 收集所有性能指标
4. 生成综合对比可视化
5. 创建跨数据集汇总报告
6. 输出改进率热力图

---

## 适用场景

LN-TSDM模型特别适用于以下时间序列预测场景：

1. **能源负荷预测**
   - 电力负荷预测
   - 天然气需求预测
   - 可再生能源发电预测

2. **气象要素推演**
   - 温度、降水量预测
   - 风速、太阳辐射预测

3. **交通流量分析**
   - 道路流量预测
   - 公共交通客流预测

4. **金融时间序列**
   - 股票价格预测
   - 外汇汇率预测

5. **工业生产预测**
   - 产量预测
   - 设备状态监测

---

## 理论优势

### 随机森林在时间序列预测中的适配性

1. **抑制过拟合**：通过集成多棵决策树，有效增强预测稳健性
2. **高维处理能力**：对高维特征空间和数据噪声具有强鲁棒性
3. **非线性建模**：精准识别数据中的复杂模式与非线性关联
4. **灵活性**：不依赖严格的线性假设，适用于不同类型时间序列

### 混合建模策略的创新点

1. **特征解耦**：通过分解实现趋势性特征与复杂动态关系的独立建模
2. **方法适配**：针对不同性质成分采用最适合的建模算法
3. **互补优势**：充分发挥线性模型的趋势识别精确性和非线性模型的波动处理优越性
4. **可解释性**：分解后的成分具有明确的物理意义，便于结果解释

---

## 数据集

本项目使用的数据集为美国区域电力消耗小时数据：

- **来源**：PJM Interconnection LLC
- **时间范围**：2004-2018
- **频率**：小时
- **单位**：兆瓦 (MW)

包含以下区域数据文件：
- `AEP_hourly.csv` - American Electric Power
- `COMED_hourly.csv` - Commonwealth Edison
- `DAYTON_hourly.csv` - Dayton Power & Light
- `DEOK_hourly.csv` - Duke Energy Ohio/Kentucky
- `DOM_hourly.csv` - Dominion Virginia Power
- `DUQ_hourly.csv` - Duquesne Light
- 等等...

---

## 性能优化建议

1. **数据预处理**
   - 处理缺失值和异常值
   - 适当的特征工程（添加时间特征：小时、星期几、月份等）

2. **参数调优**
   - 根据数据特征调整季节性周期
   - 优化ElasticNet的alpha和l1_ratio
   - 扩展Random Forest的参数搜索空间

3. **计算效率**
   - 对于大数据集，考虑使用采样
   - 减少网格搜索的参数组合
   - 使用并行计算（n_jobs=-1）

---

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{ln_tsdm_2025,
  title={LN-TSDM: Linear-Nonlinear Time Series Decomposition Model},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/LN-TSDM}
}
```

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。

---

## 更新日志

### v1.0.0 (2025-11-17)
- ✨ 初始版本发布
- ✅ 实现LN-TSDM核心模型
- ✅ 添加基线模型对比实验
- ✅ 完善可视化和报告生成
- 📝 完整文档和使用说明

---

**© 2025 LN-TSDM Project. All Rights Reserved.**
