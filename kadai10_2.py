

# ライブラリをインポート
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'IPAexMincho'
# irisデータ(csv形式)の読み込み
iris_df = pd.read_csv('/home/yn/デスクトップ/Kadai10/data3.csv')
iris_df2 = iris_df[["最高気温", "日照時間","平均湿度","天気概況改"]]
iris_data = iris_df[[ "最高気温","日照時間", "平均湿度"]]
iris_target = iris_df['天気概況改']



# ペアプロットの表示
sns.pairplot(iris_df2, hue="天気概況改", height=2)

# ランダムフォレストを作成してF値を計算 (木の数は1～10)
for i in range(1, 50):
    rf = RandomForestClassifier(n_estimators=i, max_depth=18, random_state=1)
    rf = rf.fit(iris_data, iris_target)
    print(f'n_estimators={i}')
    print('f1_score:', f1_score(iris_target,rf.predict(iris_data), average="micro"))

# 説明変数の重要度をグラフに出力
rf = RandomForestClassifier(n_estimators=30, max_depth=18, random_state=1)
rf = rf.fit(iris_data, iris_target)
importances = rf.feature_importances_
feature_names = [ "最高気温","日照時間", "平均湿度"]
plt.figure(figsize=(12, 5))
plt.barh(range(len(feature_names)), rf.feature_importances_, align='center')
plt.yticks(np.arange(len(feature_names)), feature_names)
plt.show()

