# Hourly Energy Consumption Data Analysis Summary

## Overview
This report summarizes the analysis of 13 CSV datasets containing hourly energy consumption data from various regions in the PJM Interconnection.

---

## Dataset Summary

### 1. AEP_hourly.csv
- **Records**: 121,273
- **Time Range**: 2004-10-01 to 2018-08-03 (5,053 days)
- **Power Range**: 9,581 - 25,695 MW
- **Average**: 15,499.51 MW
- **Missing Values**: 0

### 2. COMED_hourly.csv
- **Records**: 66,497
- **Time Range**: 2011-01-01 to 2018-08-03 (2,770 days)
- **Power Range**: 7,237 - 23,753 MW
- **Average**: 11,420.15 MW
- **Missing Values**: 0

### 3. DAYTON_hourly.csv
- **Records**: 121,275
- **Time Range**: 2004-10-01 to 2018-08-03 (5,053 days)
- **Power Range**: 982 - 3,746 MW
- **Average**: 2,037.85 MW
- **Missing Values**: 0

### 4. DEOK_hourly.csv
- **Records**: 57,739
- **Time Range**: 2012-01-01 to 2018-08-03 (2,405 days)
- **Power Range**: 907 - 5,445 MW
- **Average**: 3,105.10 MW
- **Missing Values**: 0

### 5. DOM_hourly.csv
- **Records**: 116,189
- **Time Range**: 2005-05-01 to 2018-08-03 (4,841 days)
- **Power Range**: 1,253 - 21,651 MW
- **Average**: 10,949.20 MW
- **Missing Values**: 0

### 6. DUQ_hourly.csv
- **Records**: 119,068
- **Time Range**: 2005-01-01 to 2018-08-03 (4,961 days)
- **Power Range**: 1,014 - 3,054 MW
- **Average**: 1,658.82 MW
- **Missing Values**: 0

### 7. EKPC_hourly.csv
- **Records**: 45,334
- **Time Range**: 2013-06-01 to 2018-08-03 (1,888 days)
- **Power Range**: 514 - 3,490 MW
- **Average**: 1,464.22 MW
- **Missing Values**: 0

### 8. FE_hourly.csv
- **Records**: 62,874
- **Time Range**: 2011-06-01 to 2018-08-03 (2,619 days)
- **Power Range**: 0 - 14,032 MW
- **Average**: 7,792.16 MW
- **Missing Values**: 0
- **Note**: Contains a minimum value of 0 MW which may indicate data quality issues

### 9. NI_hourly.csv
- **Records**: 58,450
- **Time Range**: 2004-05-01 to 2011-01-01 (2,435 days)
- **Power Range**: 7,003 - 23,631 MW
- **Average**: 11,701.68 MW
- **Missing Values**: 0

### 10. PJME_hourly.csv
- **Records**: 145,366
- **Time Range**: 2002-01-01 to 2018-08-03 (6,057 days)
- **Power Range**: 14,544 - 62,009 MW
- **Average**: 32,080.22 MW
- **Missing Values**: 0
- **Note**: Largest average consumption among all datasets

### 11. PJMW_hourly.csv
- **Records**: 143,206
- **Time Range**: 2002-04-01 to 2018-08-03 (5,967 days)
- **Power Range**: 487 - 9,594 MW
- **Average**: 5,602.38 MW
- **Missing Values**: 0

### 12. PJM_Load_hourly.csv
- **Records**: 32,896
- **Time Range**: 1998-04-01 to 2002-01-01 (1,370 days)
- **Power Range**: 17,461 - 54,030 MW
- **Average**: 29,766.43 MW
- **Missing Values**: 0
- **Note**: Oldest dataset, covering 1998-2002

### 13. pjm_hourly_est.csv (Consolidated Dataset)
- **Records**: 178,262
- **Time Range**: 1998-04-01 to 2018-08-03 (7,428 days)
- **Columns**: 13 (Datetime + 12 energy columns)
- **Total Missing Values**: 1,048,977
- **Note**: This is a consolidated dataset combining all regions with significant missing values due to different time coverage for each region

---

## Key Findings

### Regional Consumption Patterns
1. **Highest Average Consumption**: PJME (32,080 MW)
2. **Lowest Average Consumption**: EKPC (1,464 MW)
3. **Largest Peak Load**: PJME (62,009 MW)
4. **Smallest Peak Load**: PJMW (9,594 MW)

### Data Quality
- Most individual regional datasets have **no missing values**
- The consolidated dataset (pjm_hourly_est.csv) has extensive missing values due to varying data availability across regions
- FE_hourly.csv shows a minimum value of 0 MW which may require further investigation

### Temporal Coverage
- **Longest Coverage**: PJME (16.6 years: 2002-2018)
- **Shortest Coverage**: PJM_Load (3.75 years: 1998-2002)
- **Most Recent Start**: EKPC (2013-06-01)

### Regional Comparison
| Region | Avg MW | Std Dev | Peak MW |
|--------|---------|---------|---------|
| PJME   | 32,080  | 6,464   | 62,009  |
| PJM_Load | 29,766 | 5,850 | 54,030  |
| AEP    | 15,500  | 2,591   | 25,695  |
| NI     | 11,702  | 2,371   | 23,631  |
| COMED  | 11,420  | 2,304   | 23,753  |
| DOM    | 10,949  | 2,414   | 21,651  |
| FE     | 7,792   | 1,331   | 14,032  |
| PJMW   | 5,602   | 979     | 9,594   |
| DEOK   | 3,105   | 600     | 5,445   |
| DAYTON | 2,038   | 393     | 3,746   |
| DUQ    | 1,659   | 302     | 3,054   |
| EKPC   | 1,464   | 379     | 3,490   |

---

## Generated Analysis Files

Each dataset has a corresponding analysis visualization file:
1. AEP_analysis.png
2. COMED_analysis.png
3. DAYTON_analysis.png
4. DEOK_analysis.png
5. DOM_analysis.png
6. DUQ_analysis.png
7. EKPC_analysis.png
8. FE_analysis.png
9. NI_analysis.png
10. PJME_analysis.png
11. PJMW_analysis.png
12. PJM_Load_analysis.png
13. pjm_hourly_est_analysis.png

Each visualization includes:
- Time series plot
- Distribution histogram
- Box plot
- Monthly average trend

---

## Recommendations

1. **Data Quality**: Investigate the 0 MW reading in FE_hourly.csv
2. **Missing Data**: Consider interpolation strategies for the consolidated dataset
3. **Seasonal Analysis**: Further analysis could explore seasonal patterns and year-over-year trends
4. **Forecasting**: These cleaned datasets are ready for time series forecasting models
5. **Correlation Analysis**: Investigate relationships between different regional consumptions

---

*Report Generated: 2025-11-17*
