import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
import warnings
warnings.filterwarnings("ignore")

# 读取数据
df = pd.read_csv("Hitters.csv").dropna()
X = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1)
y = df["Salary"]

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

alphas = np.logspace(-3, 3, 200)
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# ===================== Ridge 岭回归 =====================
ridge_cv = RidgeCV(alphas=alphas, cv=cv)
ridge_cv.fit(X_scaled, y)
ridge_coefs = np.array([Ridge(alpha=a).fit(X_scaled,y).coef_ for a in alphas])

plt.figure(figsize=(10,7))
plt.plot(alphas, ridge_coefs)
plt.axvline(ridge_cv.alpha_, ls="--", c="k", label=f"lambda_min={ridge_cv.alpha_:.3f}")
plt.xscale("log")
plt.title("Ridge Coefficient Path")
plt.xlabel("lambda")
plt.ylabel("Coefficient")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

# ===================== Lasso回归 =====================
lasso_cv = LassoCV(alphas=alphas, cv=cv, random_state=42, max_iter=100000)
lasso_cv.fit(X_scaled, y)
lasso_coefs = np.array([Lasso(alpha=a, random_state=42, max_iter=100000).fit(X_scaled,y).coef_ for a in alphas])

plt.figure(figsize=(10,7))
plt.plot(alphas, lasso_coefs)
plt.axvline(lasso_cv.alpha_, ls="--", c="k", label=f"lambda_min={lasso_cv.alpha_:.3f}")
plt.xscale("log")
plt.title("Lasso Coefficient Path")
plt.xlabel("lambda")
plt.ylabel("Coefficient")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

# ===================== ElasticNet弹性网 =====================
enet_cv = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv=cv, random_state=42, max_iter=100000)
enet_cv.fit(X_scaled, y)
enet_coefs = np.array([ElasticNet(alpha=a,l1_ratio=0.5,random_state=42, max_iter=100000).fit(X_scaled,y).coef_ for a in alphas])

plt.figure(figsize=(10,7))
plt.plot(alphas, enet_coefs)
plt.axvline(enet_cv.alpha_, ls="--", c="k", label=f"lambda_min={enet_cv.alpha_:.3f}")
plt.xscale("log")
plt.title("ElasticNet Coefficient Path")
plt.xlabel("lambda")
plt.ylabel("Coefficient")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
