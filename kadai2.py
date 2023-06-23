import csv
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import seaborn as sns


from pmdarima import auto_arima



df1 = pd.read_csv("/home/yn/デスクトップ/kadai/nikkei_stock_average_daily_jp.csv",encoding="shift_jis")

plt.rcParams['font.family'] = 'IPAexMincho'

selected_index = ["終値"]
#selected_index=["平均気温","最高気温","最低気温","降水量","日照時間","降雪量","平均風速","平均蒸気圧","平均湿度","平均現地気圧"]

t=df1.loc[:,"データ日付"].values[:-1]
v=df1.loc[:,"終値"].values[:-1]

#plt.figure()
#plt.rcParams['font.family'] = 'IPAexMincho'
#plt.plot(t,v)
#plt.show()


# 時系列データをトレンド、規則的変動成分、不規則変動成分に分解
res = sm.tsa.seasonal_decompose(v, period=12)
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(411)
ax1.set_ylabel('observed')
ax1.plot(res.observed)
ax2 = fig.add_subplot(412)
ax2.set_ylabel('trend')
ax2.plot(res.trend)
ax3 = fig.add_subplot(413)
ax3.set_ylabel('seasonal')
ax3.plot(res.seasonal)
ax4 = fig.add_subplot(414)
ax4.set_ylabel('resid')
ax4.plot(res.resid)
plt.show()


# モデル構築と検証のためのデータを準備
train = v[:]  # 学習用データ 2021年まで
test = v[500:]  # 検証用データ 2022年分

#stepwise_fit = auto_arima(train, seasonal=True, trace=True, m=12, stepwise=True)                         
#stepwise_fit.summary()



SARIMA = sm.tsa.SARIMAX(train, order=(0, 1, 0), seasonal_order=(2, 0, 0, 12)).fit()
print(1)
pred = SARIMA.predict(844)
# 実データと予測結果の図示
print(pred)
plt.plot(train, label="train")

plt.plot(pred, "r", label="pred")
plt.show()



