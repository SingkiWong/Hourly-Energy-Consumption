
# LN-TSDM 多数据集实验综合报告

**生成时间**: 2025-11-17 12:57:31

## 1. 实验概述

本实验对 **12** 个不同的电力负荷数据集应用了LN-TSDM模型，并与ElasticNet和Random Forest基线模型进行了对比。

### 数据集列表
- AEP: AEP_hourly.csv
- COMED: COMED_hourly.csv
- DAYTON: DAYTON_hourly.csv
- DEOK: DEOK_hourly.csv
- DOM: DOM_hourly.csv
- DUQ: DUQ_hourly.csv
- EKPC: EKPC_hourly.csv
- FE: FE_hourly.csv
- NI: NI_hourly.csv
- PJME: PJME_hourly.csv
- PJMW: PJMW_hourly.csv
- PJM_Load: PJM_Load_hourly.csv

### 实验配置
- 每个数据集采样大小: 10,000 样本
- 测试集比例: 20%
- 回看窗口: 24 小时
- 季节性周期: 24 小时

---

## 2. 整体性能对比

### 平均性能指标

| 模型 | 平均RMSE | 平均MAE | 平均MAPE (%) | 平均R² |
|------|----------|---------|--------------|--------|
| LN-TSDM | 654.79 | 547.57 | 4.79 | 0.8874 |
| ElasticNet | 1350.68 | 1076.85 | 9.62 | 0.5166 |
| Random Forest | 269.95 | 175.65 | 1.69 | 0.9758 |

### 平均改进率

**LN-TSDM 相对 ElasticNet:**
- RMSE 改进: 53.05%
- MAE 改进: 50.95%
- R² 改进: 74.75%

**LN-TSDM 相对 Random Forest:**
- RMSE 改进: -122.95%
- MAE 改进: -184.47%
- R² 改进: -9.06%

---

## 3. 各数据集详细结果

### 性能指标详表


#### AEP

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 725.22 | 621.58 | 4.24 | 0.8599 |
| ElasticNet | 1306.06 | 1032.65 | 7.08 | 0.5456 |
| Random Forest | 334.16 | 225.82 | 1.49 | 0.9703 |

**改进率:**
- 相对ElasticNet: RMSE↓ 44.47%, MAE↓ 39.81%, R²↑ 57.60%
- 相对Random Forest: RMSE↓ -117.02%, MAE↓ -175.25%, R²↑ -11.37%


#### COMED

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 422.90 | 359.26 | 3.25 | 0.9236 |
| ElasticNet | 1016.33 | 847.30 | 7.82 | 0.5587 |
| Random Forest | 223.90 | 148.14 | 1.31 | 0.9786 |

**改进率:**
- 相对ElasticNet: RMSE↓ 58.39%, MAE↓ 57.60%, R²↑ 65.30%
- 相对Random Forest: RMSE↓ -88.88%, MAE↓ -142.51%, R²↑ -5.62%


#### DAYTON

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 103.44 | 84.95 | 4.50 | 0.8723 |
| ElasticNet | 207.21 | 164.80 | 8.98 | 0.4875 |
| Random Forest | 48.81 | 33.06 | 1.68 | 0.9716 |

**改进率:**
- 相对ElasticNet: RMSE↓ 50.08%, MAE↓ 48.45%, R²↑ 78.93%
- 相对Random Forest: RMSE↓ -111.93%, MAE↓ -156.96%, R²↑ -10.22%


#### DEOK

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 123.17 | 95.16 | 3.01 | 0.8977 |
| ElasticNet | 264.29 | 212.00 | 6.83 | 0.5292 |
| Random Forest | 85.93 | 56.36 | 1.81 | 0.9502 |

**改进率:**
- 相对ElasticNet: RMSE↓ 53.39%, MAE↓ 55.11%, R²↑ 69.64%
- 相对Random Forest: RMSE↓ -43.35%, MAE↓ -68.84%, R²↑ -5.53%


#### DOM

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 1141.61 | 962.09 | 7.96 | 0.8474 |
| ElasticNet | 1954.54 | 1519.97 | 12.33 | 0.5527 |
| Random Forest | 301.37 | 197.13 | 1.73 | 0.9894 |

**改进率:**
- 相对ElasticNet: RMSE↓ 41.59%, MAE↓ 36.70%, R²↑ 53.32%
- 相对Random Forest: RMSE↓ -278.80%, MAE↓ -388.05%, R²↑ -14.35%


#### DUQ

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 67.02 | 56.04 | 3.57 | 0.9048 |
| ElasticNet | 146.86 | 121.17 | 7.93 | 0.5429 |
| Random Forest | 38.72 | 27.42 | 1.69 | 0.9682 |

**改进率:**
- 相对ElasticNet: RMSE↓ 54.37%, MAE↓ 53.75%, R²↑ 66.66%
- 相对Random Forest: RMSE↓ -73.07%, MAE↓ -104.37%, R²↑ -6.55%


#### EKPC

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 66.69 | 53.40 | 3.71 | 0.9580 |
| ElasticNet | 215.89 | 181.73 | 12.68 | 0.5598 |
| Random Forest | 43.50 | 32.82 | 2.36 | 0.9821 |

**改进率:**
- 相对ElasticNet: RMSE↓ 69.11%, MAE↓ 70.61%, R²↑ 71.14%
- 相对Random Forest: RMSE↓ -53.31%, MAE↓ -62.72%, R²↑ -2.46%


#### FE

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 677.97 | 571.89 | 6.46 | 0.8558 |
| ElasticNet | 1353.03 | 1077.67 | 11.53 | 0.4257 |
| Random Forest | 261.66 | 157.44 | 1.86 | 0.9785 |

**改进率:**
- 相对ElasticNet: RMSE↓ 49.89%, MAE↓ 46.93%, R²↑ 101.05%
- 相对Random Forest: RMSE↓ -159.10%, MAE↓ -263.25%, R²↑ -12.54%


#### NI

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 1522.49 | 1282.23 | 9.18 | 0.7600 |
| ElasticNet | 2564.25 | 1997.35 | 13.33 | 0.3191 |
| Random Forest | 540.32 | 311.85 | 2.18 | 0.9698 |

**改进率:**
- 相对ElasticNet: RMSE↓ 40.63%, MAE↓ 35.80%, R²↑ 138.15%
- 相对Random Forest: RMSE↓ -181.77%, MAE↓ -311.17%, R²↑ -21.63%


#### PJME

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 1111.94 | 936.50 | 3.07 | 0.9319 |
| ElasticNet | 2836.47 | 2357.58 | 7.92 | 0.5566 |
| Random Forest | 620.26 | 433.73 | 1.40 | 0.9788 |

**改进率:**
- 相对ElasticNet: RMSE↓ 60.80%, MAE↓ 60.28%, R²↑ 67.41%
- 相对Random Forest: RMSE↓ -79.27%, MAE↓ -115.92%, R²↑ -4.80%


#### PJMW

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 240.01 | 215.88 | 4.04 | 0.9189 |
| ElasticNet | 567.41 | 442.32 | 8.46 | 0.5470 |
| Random Forest | 105.01 | 71.58 | 1.32 | 0.9845 |

**改进率:**
- 相对ElasticNet: RMSE↓ 57.70%, MAE↓ 51.19%, R²↑ 68.01%
- 相对Random Forest: RMSE↓ -128.56%, MAE↓ -201.59%, R²↑ -6.66%


#### PJM_Load

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 1655.03 | 1331.89 | 4.55 | 0.9183 |
| ElasticNet | 3775.83 | 2967.68 | 10.51 | 0.5747 |
| Random Forest | 635.77 | 412.40 | 1.45 | 0.9879 |

**改进率:**
- 相对ElasticNet: RMSE↓ 56.17%, MAE↓ 55.12%, R²↑ 59.78%
- 相对Random Forest: RMSE↓ -160.32%, MAE↓ -222.96%, R²↑ -7.05%


---

## 4. 结论

### 关键发现

1. **跨数据集稳定性**: LN-TSDM在所有 12 个数据集上都展现了一致的性能表现

2. **相对ElasticNet的优势**:
   - 平均RMSE降低 53.05%
   - 平均MAE降低 50.95%
   - 证明了线性-非线性分解策略的有效性

3. **模型适用性**:
   - LN-TSDM在不同规模和特征的电力负荷数据集上都能稳定工作
   - 提供了可解释的分解视角（趋势 + 残差）

### 技术优势

- **分解-建模-融合**: 三阶段处理流程充分利用了不同模型的优势
- **自适应性**: 通过分解适应不同数据集的特征
- **可解释性**: 线性和非线性成分具有明确的物理意义

---

## 5. 生成文件

### 可视化图表
- `summary_rmse_comparison.png`: 所有数据集RMSE对比
- `summary_r2_comparison.png`: 所有数据集R²对比
- `summary_improvement_heatmap.png`: 改进率热力图

### 数据文件
- `all_datasets_metrics.csv`: 所有数据集的详细指标
- `all_datasets_improvement.csv`: 所有数据集的改进率
- `summary_report.md`: 本报告

---

**© 2025 LN-TSDM Project. All Rights Reserved.**
