from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype = np.float64)

def create_dataset(num, variance, step, correlation = False):
	val = 1
	ys = []
	for i in range(num):
		ys += [val + random.randrange(-variance, variance)]
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	xs = [i for i in range(num)]
	return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)

def best_fit_line(xs, ys):
	mean_x = mean(xs)
	mean_y = mean(ys)
	m = (mean_x * mean_y  - mean(xs * ys)) / (mean_x ** 2 - mean(xs ** 2))
	b = mean_y - m * mean_x
	return m, b

def squared_error(ys, predict_ys):
	return sum((ys - predict_ys) ** 2)

def coefficient_of_determination(ys, predict_ys):
	mean_y = mean(ys)
	y_mean_line = [mean_y for _ in ys]
	squared_error_regression = squared_error(ys, predict_ys)
	squared_error_mean = squared_error(ys, y_mean_line)
	return 1 - squared_error_regression / squared_error_mean

xs, ys = create_dataset(40, 1000000, 2, correlation = 'pos')

m, b = best_fit_line(xs, ys)
print(m, b)

regression_line = [m * i + b for i in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

predict_x = 7
predict_y = m * predict_x + b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s = 100, color = 'g')
plt.plot(xs, regression_line)
# plt.xlim(0, 10)
# plt.ylim(0, 10)
plt.show()