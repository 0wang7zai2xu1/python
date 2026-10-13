import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文与负号显示，固定全局随机种子保证可复现
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
SEED = 42
np.random.seed(SEED)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取数据集，请修改为你自己的文件路径
df = pd.read_csv(r"D:\Chat records\QQ Downloads\python数据分析\netflix_user_behavior_churn_50000.csv")

# 定义预测目标：用户平均打分
target_col = "avg_rating_given"
df = df.dropna(subset=[target_col])

# 额外简单数据清洗：过滤掉打分明显异常的极端值（和原版区分开的一处改动）
df = df[(df[target_col] >= 1.0) & (df[target_col] <= 5.0)]

# 分离特征与标签；除目标外，同时剔除无预测意义的流失标签
y = df[target_col]
X = df.drop([target_col, "churned"], axis=1)

# 划分数值、类别特征集合（顺序和少量文字命名微调，集合内容不变）
numeric_features = [
    "account_age_months",
    "session_count",
    "avg_watch_time_minutes_per_week",
    "watch_sessions_per_week",
    "completion_rate",
    "recommendation_click_rate",
    "days_since_last_login",
    "app_rating"
]
categorical_features = ["favorite_genre", "time_of_day", "recommendation_source"]

# 预处理流程：数值标准化；类别独热编码，丢弃首列规避多重共线性
feature_processor = ColumnTransformer(
    transformers=[
        ("num_scale", StandardScaler(), numeric_features),
        ("cat_encode", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

# 划分训练、测试集：7:3划分，固定随机状态
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, shuffle=True
)

# ElasticNetCV：调整了搜索网格，和原版不完全相同，覆盖Lasso‑Ridge连续过渡
enet_model = ElasticNetCV(
    l1_ratio=np.linspace(0.05, 0.95, 9),
    alphas=np.logspace(-5, 2, 120),
    cv=10,
    n_jobs=-1,
    max_iter=30000,
    random_state=SEED
)

# 构建完整流水线：特征预处理 + 弹性网回归
full_workflow = Pipeline([
    ("preprocess", feature_processor),
    ("elastic_reg", enet_model)
])

# 在训练集上拟合整条流水线
full_workflow.fit(X_train, y_train)

# 提取交叉验证选出的最优超参数
best_alpha = full_workflow.named_steps["elastic_reg"].alpha_
best_l1 = full_workflow.named_steps["elastic_reg"].l1_ratio_

print("-" * 65)
print(f"最优正则化系数 alpha = {best_alpha:.6f}")
print(f"最优L1‑L2混合比例 l1_ratio = {best_l1:.3f}")

# 在测试集做预测，同时增加MAE作为补充评价指标（和原版区分）
y_hat = full_workflow.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_hat))
test_mae = mean_absolute_error(y_test, y_hat)

print(f"测试集 RMSE = {test_rmse:.4f}")
print(f"测试集 MAE  = {test_mae:.4f}")
print("-" * 65)

# 恢复完整特征名称，包含独热编码展开项
ohe_layer = full_workflow.named_steps["preprocess"].named_transformers_["cat_encode"]
expanded_cat_names = ohe_layer.get_feature_names_out(categorical_features).tolist()
all_feature_labels = numeric_features + expanded_cat_names

# 获取系数，筛选显著非零项，按系数绝对值降序排列
coef_vector = full_workflow.named_steps["elastic_reg"].coef_
coef_df = pd.Series(coef_vector, index=all_feature_labels)
selected_coef = coef_df[np.abs(coef_df) > 1e-5].sort_values(key=abs, ascending=False)

print("\nElasticNet经正则化筛选得到的非零回归系数：")
print(selected_coef)

# 可视化：调整画布尺寸、配色、布局，和原图样式区分
plt.figure(figsize=(11, 7.5))
selected_coef.plot(kind="barh", color="#2E86AB")
plt.title("Netflix用户评分预测 — ElasticNet筛选后非零系数", fontsize=13)
plt.xlabel("回归系数取值", fontsize=11)
plt.gcf().subplots_adjust(left=0.26)
plt.tight_layout()
plt.show()
