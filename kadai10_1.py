

# ライブラリをインポート
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'IPAexMincho'
# wetherデータ(csv形式)の読み込み
wether_df = pd.read_csv('/home/yn/デスクトップ/Kadai10/data3.csv')

wether_data = wether_df[["平均気温", "最高気温", "最低気温", "降水量","日照時間", "降雪量", "平均風速", "平均蒸気圧", "平均湿度", "平均現地気圧"]]
wether_target = wether_df['天気概況改']
classnames=list(set(wether_target))
print(len(classnames))
xlist=[]
ylist=[]

print(classnames)

clf=DecisionTreeClassifier(criterion="entropy",max_depth=5)
clf=clf.fit(wether_data,wether_target)

plt.figure(figsize=(15,15))

plot_tree(clf, filled=True, class_names=classnames, feature_names=["平均気温", "最高気温", "最低気温", "降水量", "日照時間", "降雪量", "平均風速", "平均蒸気圧", "平均湿度", "平均現地気圧"])
plt.show()

# ペアプロットの表示
#sns.pairplot(wether_df, hue="天気概況", height=2)

