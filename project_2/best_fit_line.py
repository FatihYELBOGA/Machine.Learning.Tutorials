from statistics import mean
import numpy as np
import matplotlib .pyplot as plt
from matplotlib import style
import random


# x ve y datasetlerimizi veri sayisini ve araligi baz alarak dolduruyoruz
def create_x_and_y_values(data_length, interval, step=2, correlation='positive'):
    first_step = 0
    x_dataset = []
    y_dataset = []
    for i in range(data_length):
        x_dataset.append(i)
        y_dataset.append(first_step + random.randrange(-interval, interval))
        if(correlation=='positive'):
            first_step += step
        elif(correlation=='negative'):
            first_step -= step

    return np.array(x_dataset, dtype=np.float64), np.array(y_dataset, dtype=np.float64)

# x ve y eksenindeki verilere göre en iyi m ve b degerlerini al
def find_best_fit_line(x, y):
    m = ((mean(x) * mean(y)) - mean(x*y)) / ((mean(x)*mean(x)) - mean(x*x))
    b = mean(y) - (m * mean(x))
    return m, b

# y eksenlerindeki degerlerin farkinin karesinin toplamini al
def find_squared_error(y_orig, y):
    return sum((y_orig-y)**2)

# r kare teorisine gore sonucu bul
def find_coefficient_of_r_squared_theory(y_orig, y):
    y_mean_data = [mean(y) for y_data in y]
    sqaured_error_top = find_squared_error(y_orig, y)
    sqaured_error_bottom = find_squared_error(y_orig, y_mean_data)
    return 1 - (sqaured_error_top / sqaured_error_bottom)

# x ve y eksenindeki veriler
x, y = create_x_and_y_values(20, 20, step=5, correlation='positive')

# m ve b degerlerini yazdir
m, b = find_best_fit_line(x, y)
print("m and b values:", m, "and", b)

# m ve b degerleri sonrası y eksenindeki yeni verileri olustur
best_fit_data = []
for x_data in x:
    best_fit_data.append((m * x_data) + b)

# r kare teorisi sonucu cikan katsayiyi yazdir
err = find_coefficient_of_r_squared_theory(y, best_fit_data)
print("Coefficient of the R squared theory:", err)

# tahmin verisi olarak 7 yi kullan
predict_x = 7
predict_y = (m * predict_x) + b

# baslangictaki x ve y degerlerini nokta olarak,  tahmin 7 yi nokta olarak yazdır, en iyi y degerlerini fonksiyon olarak,
style.use('fivethirtyeight')
plt.scatter(x,y)
plt.scatter(predict_x, predict_y, s=100, color="r")
plt.plot(x, best_fit_data)
plt.show()