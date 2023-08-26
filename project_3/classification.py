import numpy as np
from sklearn import model_selection, neighbors, svm
import pandas as pd

# dosyayı oku ve '?' verilerini -99999 olarak degistir, id column kaldir ve data frame guncelle
df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

# class columnunu y, harici columnlari X'e ata
X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

# verilerin %20 sini test icin kullan, KNeighborsClassifier methodunu kullanarak dogruluk oranini yazdir
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
# clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)

# tahmin verileri icin ornek sample kullan ve sonucunu yazdir
predict_breast_datas = np.array([[4,2,1,1,1,2,3,2,1],[8,7,6,4,2,1,3,7,4]])
predict_breast_datas = predict_breast_datas.reshape(len(predict_breast_datas), -1)
prediction = clf.predict(predict_breast_datas)
print('Prediction:',prediction)