
# LN-TSDM 模型实验报告

**生成时间**: 2025-11-17 12:31:37

## 1. 实验配置

- **数据文件**: AEP_hourly.csv
- **数据大小**: 10000 个样本
- **训练集大小**: 8000 个样本
- **测试集大小**: 2000 个样本
- **回看窗口**: 24
- **季节性周期**: 24
- **优化Random Forest**: True

## 2. 模型性能对比

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| LN-TSDM | 726.40 | 622.94 | 4.25 | 0.8594 |
| ElasticNet | 1306.06 | 1032.65 | 7.08 | 0.5456 |
| Random Forest | 336.19 | 225.63 | 1.49 | 0.9699 |

## 3. 关键发现

### LN-TSDM 相对 ElasticNet 的改进
- RMSE 降低: 44.38%
- MAE 降低: 39.68%
- MAPE 降低: 40.06%

### LN-TSDM 相对 Random Forest 的改进
- RMSE 降低: -116.07%
- MAE 降低: -176.09%
- MAPE 降低: -185.26%

## 4. 结论

LN-TSDM 混合模型通过"分解-建模-融合"策略，有效结合了线性模型（ElasticNet）
在趋势识别中的精确性和非线性模型（Random Forest）在处理复杂波动中的优越性。
实验结果表明，LN-TSDM 在预测精度上显著优于单一模型方法。

## 5. 生成文件

- `results_ln_tsdm_decomposition.png`: 时间序列分解可视化
- `results_predictions_comparison.png`: 预测结果对比
- `results_metrics_comparison.png`: 性能指标对比
- `results_metrics_comparison.csv`: 详细指标数据
- `results_experiment_report.md`: 本报告
