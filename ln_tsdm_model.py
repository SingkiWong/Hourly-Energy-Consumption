"""
线性-非线性时间序列分解模型 (LN-TSDM)
Linear-Nonlinear Time Series Decomposition Model

该模型通过将时间序列数据分解为线性部分和非线性部分，
分别采用弹性网络回归（ElasticNet）和随机森林算法进行建模与预测。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import STL
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class LinearNonlinearDecomposer:
    """
    线性-非线性时间序列分解器

    使用STL分解方法将时间序列分解为趋势(线性)、季节性和残差成分
    线性部分 = 趋势 + 季节性
    非线性部分 = 残差
    """

    def __init__(self, seasonal_period=24):
        """
        参数:
            seasonal_period: 季节性周期（对于小时数据，24表示日周期）
        """
        self.seasonal_period = seasonal_period
        self.stl = None

    def decompose(self, time_series):
        """
        分解时间序列为线性和非线性部分

        使用滑动平均进行简单的趋势-季节性分解
        线性部分 = 趋势（通过滑动平均获得）
        非线性部分 = 原始数据 - 趋势

        参数:
            time_series: 原始时间序列 (pandas Series)

        返回:
            linear_component: 线性成分（趋势）
            nonlinear_component: 非线性成分（原始-趋势）
        """
        # 使用滑动平均提取趋势（线性部分）
        # 窗口大小为seasonal_period以平滑季节性模式
        window_size = self.seasonal_period * 7  # 使用一周的窗口

        # 计算趋势：使用中心化移动平均
        trend = time_series.rolling(window=window_size, center=True).mean()

        # 填充边缘的NaN值（使用向前/向后填充）
        trend = trend.fillna(method='bfill').fillna(method='ffill')

        # 线性部分 = 趋势
        linear_component = trend

        # 非线性部分 = 残差（原始数据 - 趋势）
        nonlinear_component = time_series - trend

        # 创建一个简单的结果对象用于可视化
        class SimpleDecompResult:
            def __init__(self, observed, trend, seasonal, resid):
                self.observed = observed
                self.trend = trend
                self.seasonal = pd.Series(0, index=observed.index)  # 简化：不单独提取季节性
                self.resid = resid

        result = SimpleDecompResult(time_series, trend, None, nonlinear_component)

        return linear_component, nonlinear_component, result


class LN_TSDM:
    """
    线性-非线性时间序列分解模型 (LN-TSDM)

    核心思想：
    1. 分解：将时间序列分解为线性和非线性成分
    2. 建模：线性成分用ElasticNet，非线性成分用Random Forest
    3. 融合：加权求和两个模型的预测结果
    """

    def __init__(self,
                 seasonal_period=24,
                 linear_alpha=1.0,
                 linear_l1_ratio=0.5,
                 optimize_rf=True,
                 rf_param_grid=None,
                 n_cv_folds=5,
                 verbose=True):
        """
        参数:
            seasonal_period: 季节性周期
            linear_alpha: ElasticNet正则化强度
            linear_l1_ratio: ElasticNet的L1与L2比例
            optimize_rf: 是否对Random Forest进行网格搜索优化
            rf_param_grid: Random Forest参数网格
            n_cv_folds: 交叉验证折数
            verbose: 是否打印详细信息
        """
        self.seasonal_period = seasonal_period
        self.linear_alpha = linear_alpha
        self.linear_l1_ratio = linear_l1_ratio
        self.optimize_rf = optimize_rf
        self.n_cv_folds = n_cv_folds
        self.verbose = verbose

        # 默认Random Forest参数网格
        if rf_param_grid is None:
            self.rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            self.rf_param_grid = rf_param_grid

        # 初始化组件
        self.decomposer = LinearNonlinearDecomposer(seasonal_period)
        self.linear_model = ElasticNet(alpha=linear_alpha, l1_ratio=linear_l1_ratio, max_iter=10000)
        self.nonlinear_model = None
        self.scaler_X = StandardScaler()
        self.scaler_y_linear = StandardScaler()
        self.scaler_y_nonlinear = StandardScaler()

        # 存储分解结果
        self.linear_component = None
        self.nonlinear_component = None
        self.decomposition_result = None

    def create_features(self, data, lookback=24):
        """
        创建时间序列特征

        参数:
            data: 时间序列数据
            lookback: 回看窗口大小

        返回:
            X: 特征矩阵
            y: 目标变量
        """
        X, y = [], []

        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    def fit(self, time_series, lookback=24):
        """
        训练LN-TSDM模型

        参数:
            time_series: 原始时间序列 (pandas Series)
            lookback: 回看窗口大小
        """
        if self.verbose:
            print("=" * 80)
            print("开始训练 LN-TSDM 模型")
            print("=" * 80)

        # 步骤1: 分解时间序列
        if self.verbose:
            print("\n[步骤 1/3] 分解时间序列...")
        self.linear_component, self.nonlinear_component, self.decomposition_result = \
            self.decomposer.decompose(time_series)

        if self.verbose:
            print(f"  ✓ 线性成分范围: [{self.linear_component.min():.2f}, {self.linear_component.max():.2f}]")
            print(f"  ✓ 非线性成分范围: [{self.nonlinear_component.min():.2f}, {self.nonlinear_component.max():.2f}]")

        # 步骤2: 创建特征
        if self.verbose:
            print(f"\n[步骤 2/3] 创建特征 (回看窗口={lookback})...")

        # 线性特征
        X_linear, y_linear = self.create_features(self.linear_component.values, lookback)
        # 非线性特征
        X_nonlinear, y_nonlinear = self.create_features(self.nonlinear_component.values, lookback)

        if self.verbose:
            print(f"  ✓ 线性特征形状: {X_linear.shape}")
            print(f"  ✓ 非线性特征形状: {X_nonlinear.shape}")

        # 标准化
        X_linear_scaled = self.scaler_X.fit_transform(X_linear)
        y_linear_scaled = self.scaler_y_linear.fit_transform(y_linear.reshape(-1, 1)).ravel()
        y_nonlinear_scaled = self.scaler_y_nonlinear.fit_transform(y_nonlinear.reshape(-1, 1)).ravel()

        # 步骤3: 训练模型
        if self.verbose:
            print(f"\n[步骤 3/3] 训练模型...")

        # 训练线性模型 (ElasticNet)
        if self.verbose:
            print("  → 训练 ElasticNet 模型 (线性成分)...")
        self.linear_model.fit(X_linear_scaled, y_linear_scaled)

        # 训练非线性模型 (Random Forest)
        if self.optimize_rf:
            if self.verbose:
                print("  → 使用网格搜索优化 Random Forest 模型 (非线性成分)...")

            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)

            # 网格搜索
            rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf_base,
                self.rf_param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_nonlinear, y_nonlinear_scaled)

            self.nonlinear_model = grid_search.best_estimator_

            if self.verbose:
                print(f"  ✓ 最优参数: {grid_search.best_params_}")
                print(f"  ✓ 最优交叉验证分数: {-grid_search.best_score_:.4f}")
        else:
            if self.verbose:
                print("  → 训练 Random Forest 模型 (非线性成分)...")
            self.nonlinear_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            self.nonlinear_model.fit(X_nonlinear, y_nonlinear_scaled)

        if self.verbose:
            print("\n" + "=" * 80)
            print("LN-TSDM 模型训练完成！")
            print("=" * 80)

    def predict(self, time_series, lookback=24):
        """
        使用LN-TSDM模型进行预测

        参数:
            time_series: 原始时间序列 (pandas Series)
            lookback: 回看窗口大小

        返回:
            predictions: 预测结果
        """
        # 分解时间序列
        linear_comp, nonlinear_comp, _ = self.decomposer.decompose(time_series)

        # 创建特征
        X_linear, _ = self.create_features(linear_comp.values, lookback)
        X_nonlinear, _ = self.create_features(nonlinear_comp.values, lookback)

        # 标准化
        X_linear_scaled = self.scaler_X.transform(X_linear)

        # 预测
        y_linear_pred_scaled = self.linear_model.predict(X_linear_scaled)
        y_nonlinear_pred_scaled = self.nonlinear_model.predict(X_nonlinear)

        # 反标准化
        y_linear_pred = self.scaler_y_linear.inverse_transform(
            y_linear_pred_scaled.reshape(-1, 1)
        ).ravel()
        y_nonlinear_pred = self.scaler_y_nonlinear.inverse_transform(
            y_nonlinear_pred_scaled.reshape(-1, 1)
        ).ravel()

        # 融合预测：由于分解为 原始 = 线性 + 非线性，所以预测也应该相加
        # 而不是加权平均！
        predictions = y_linear_pred + y_nonlinear_pred

        return predictions, y_linear_pred, y_nonlinear_pred


class BaselineModels:
    """
    基线模型：单一ElasticNet和单一Random Forest
    用于对比实验
    """

    def __init__(self, model_type='elasticnet', optimize=False, verbose=True):
        """
        参数:
            model_type: 'elasticnet' 或 'randomforest'
            optimize: 是否进行参数优化
            verbose: 是否打印详细信息
        """
        self.model_type = model_type
        self.optimize = optimize
        self.verbose = verbose
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def create_features(self, data, lookback=24):
        """创建时间序列特征"""
        X, y = [], []

        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    def fit(self, time_series, lookback=24):
        """训练基线模型"""
        if self.verbose:
            print(f"\n训练 {self.model_type.upper()} 基线模型...")

        # 创建特征
        X, y = self.create_features(time_series.values, lookback)

        # 标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # 训练模型
        if self.model_type == 'elasticnet':
            self.model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
            self.model.fit(X_scaled, y_scaled)

        elif self.model_type == 'randomforest':
            if self.optimize:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                tscv = TimeSeriesSplit(n_splits=5)
                rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
                grid_search = GridSearchCV(
                    rf_base, param_grid, cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X, y_scaled)
                self.model = grid_search.best_estimator_

                if self.verbose:
                    print(f"  最优参数: {grid_search.best_params_}")
            else:
                self.model = RandomForestRegressor(
                    n_estimators=100, max_depth=20,
                    random_state=42, n_jobs=-1
                )
                self.model.fit(X, y_scaled)

        if self.verbose:
            print(f"  ✓ {self.model_type.upper()} 模型训练完成")

    def predict(self, time_series, lookback=24):
        """预测"""
        X, _ = self.create_features(time_series.values, lookback)

        if self.model_type == 'elasticnet':
            X_scaled = self.scaler_X.transform(X)
            y_pred_scaled = self.model.predict(X_scaled)
        else:
            y_pred_scaled = self.model.predict(X)

        # 反标准化
        y_pred = self.scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()

        return y_pred


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    评估模型性能

    参数:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称

    返回:
        metrics: 评估指标字典
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

    return metrics


def plot_decomposition(decomposition_result, title="时间序列分解"):
    """绘制时间序列分解图"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))

    # 原始序列
    axes[0].plot(decomposition_result.observed, label='原始序列', color='blue')
    axes[0].set_ylabel('原始值')
    axes[0].legend()
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # 趋势
    axes[1].plot(decomposition_result.trend, label='趋势', color='green')
    axes[1].set_ylabel('趋势')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 季节性
    axes[2].plot(decomposition_result.seasonal, label='季节性', color='orange')
    axes[2].set_ylabel('季节性')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 残差
    axes[3].plot(decomposition_result.resid, label='残差(非线性)', color='red')
    axes[3].set_ylabel('残差')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel('时间')

    plt.tight_layout()
    return fig


def plot_predictions(y_true, predictions_dict, title="模型预测对比", sample_size=500):
    """
    绘制预测对比图

    参数:
        y_true: 真实值
        predictions_dict: 预测值字典 {模型名: 预测值}
        title: 图表标题
        sample_size: 显示样本数量
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # 只显示部分数据以便观察
    plot_range = slice(0, min(sample_size, len(y_true)))

    # 上图：预测对比
    axes[0].plot(y_true[plot_range], label='真实值', color='black', linewidth=2, alpha=0.7)

    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, (name, pred) in enumerate(predictions_dict.items()):
        axes[0].plot(pred[plot_range], label=name, color=colors[i % len(colors)],
                    linewidth=1.5, alpha=0.7)

    axes[0].set_ylabel('能源消耗 (MW)')
    axes[0].set_title(f'{title} - 预测对比 (前{sample_size}个样本)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 下图：误差对比
    for i, (name, pred) in enumerate(predictions_dict.items()):
        error = y_true - pred
        axes[1].plot(error[plot_range], label=f'{name} 误差',
                    color=colors[i % len(colors)], linewidth=1, alpha=0.7)

    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_ylabel('预测误差 (MW)')
    axes[1].set_xlabel('样本索引')
    axes[1].set_title('预测误差对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_df):
    """绘制模型性能指标对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3

        ax = axes[row, col]
        bars = ax.bar(metrics_df['Model'], metrics_df[metric], color=colors[idx], alpha=0.7)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} 对比')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    # 删除多余的子图
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("LN-TSDM 模型模块已加载")
    print("主要类:")
    print("  - LinearNonlinearDecomposer: 线性-非线性分解器")
    print("  - LN_TSDM: 混合时间序列预测模型")
    print("  - BaselineModels: 基线对比模型")
