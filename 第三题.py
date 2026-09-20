import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 在线读取Carseats数据集，无需本地csv文件
data = pd.read_csv("https://www.statlearning.com/s/Carseats.csv")

# 对ShelveLoc分类变量生成哑变量，drop_first=True，基准组为Bad
data_dummy = pd.get_dummies(data, prefix="ShelveLoc", columns=["ShelveLoc"], drop_first=True, dtype="int")

# 设置特征列表，划分自变量、因变量
feature_cols = ["Price", "Income", "Advertising", "ShelveLoc_Good", "ShelveLoc_Medium"]
features = data_dummy[feature_cols]
target = data_dummy["Sales"]

# 构造回归设计矩阵，添加截距项
X_design = sm.add_constant(features)

# 拟合OLS多元线性回归模型
ols_model = sm.OLS(endog=target, exog=X_design).fit()

# 打印回归结果汇总表
print("=======多元线性回归模型汇总=======")
print(ols_model.summary())

# 计算VIF方差膨胀因子，只用不含常数项的特征矩阵
vif_result = pd.DataFrame({
    "特征名称": features.columns,
    "VIF值": [variance_inflation_factor(features.values, idx) for idx in range(features.shape[1])]
})
print("\n=======VIF多重共线性检验结果=======")
print(vif_result)
