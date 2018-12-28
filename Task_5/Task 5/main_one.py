## Практическое задание № 5. Линейная регрессия (с одной переменной)

# Инициализация
import numpy as np
import matplotlib.pyplot as plt

from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

# ================= Часть 1. Визуализация данных =================

print('Часть 1. Визуализация данных')

# Загрузка данных и формирование матрицы объекты-признаки X и вектора меток y
data = np.loadtxt('data1.txt', delimiter = ',')
m = data.shape[0]

X = np.array(data[:, 0:1]); X = np.concatenate((np.ones((m, 1)), X), axis = 1)
y = np.array(data[:, 1:2])

# Визуализация данных
plotData(data)
plt.show()

# ================== Часть 2. Градиентный спуск ==================

print('Часть 2. Градиентный спуск')

# Задание начальных значений параметров модели
theta = np.zeros([2, 1])

# Вычисление значения стоимостной функции для начального theta
J = computeCost(X, y, theta)
print('Значение стоимостной функции: {:.4f}'.format(J))

# Задание параметров градиентного спуска
iterations = 1500
alpha = 0.01


# Выполнение градиентного спуска
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print('Найденные параметры модели: {:.4f} {:.4f}'.format(theta[0, 0], theta[1, 0]))

# Визуализация данных и линии регресии
plotData(data)
plt.plot(X[:, 1], np.dot(X, theta), '-', label = 'Линейная регрессия')
plt.legend(loc = 'upper right', shadow = True, fontsize = 12, numpoints = 1)
plt.show()

# ================= Часть 3. Предсказания прибыли ================

print('Часть 3. Предсказание прибыли')

# Предсказание прибыли для численности населения 35,000 и 70,000
predict1 = np.dot(np.array([[1, 3.5]]), theta)
predict2 = np.dot(np.array([[1, 7]]),   theta)
print('Для численности населения 35,000 предсказанная прибыль = ${:.4f}'.format(predict1[0, 0] * 10000))
print('Для численности населения 70,000 предсказанная прибыль = ${:.4f}'.format(predict2[0, 0] * 10000))