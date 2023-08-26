import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression


# WIKI/GOOGL verilerini dataframe olarak aliyoruz
df = quandl.get("WIKI/GOOGL")
                 
# istedigimiz columnlari belirliyoruz
df = df[['Adj. Open', 'Adj. Close', 'Adj. Low', 'Adj. High']]
# open-close ve low-high değerlerini baz alarak columnlar olusturuyoruz
df['Open-Close Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100  
df['Low-High Change'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100  
# goruntulemek istedigimiz dataframe columnlarini tekrar duzenliyoruz
df = df[['Adj. Open', 'Adj. Close', 'Open-Close Change', 'Adj. Low', 'Adj. High', 'Low-High Change']]

# tahmin columnlarinda close column baz alarak forecast_out kadar uzaktaki veriyi aliyoruz 
forecast_col = 'Adj. Close'
forecast_out = math.ceil(0.001*len(df))
df['Forecast'] = df[forecast_col].shift(-(forecast_out)) 
# bazi veriler NA olacagi icin bunlari dataframeden kaldiriyoruz
df.dropna(inplace=True)

# X icin forecast column harici columnlari, y icin forecast columnunu aliyoruz
X = np.array(df.drop(['Forecast'], axis=1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
y = np.array(df['Forecast'])

# X ve y columnlarinin uzunluklarini yazdir
print(len(X), len(y))

# test (20%) ve train olmak üzere 2 kume olusturuyoruz ve sonuclari goruntuluyoruz
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=10)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
# belirledigimiz arrayin tahmin sonuclarini aliyoruz
forecast_set = clf.predict(X_lately)

# tahminde bulundugumuz veri adedi, tahmin edilen verilerin sonuclarini ve dogruluk yuzdesini yazdiriyoruz
print(forecast_out, forecast_set, accuracy)
print(df.tail())