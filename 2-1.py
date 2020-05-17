import numpy
import matplotlib.pyplot as plt


X = numpy.array([[3.2, 0.8], [5, 5], [3, 3], [4, 3], [1, 1], [2, 2], [1, 4]])
labels = numpy.array([1, 1, 1, 1, -1, -1, -1])
k = 0
b = 0

def tanh(a):
	return numpy.tanh(a)
def derivate_tanh(a):
	return 1 - numpy.square(tanh(a))

step = 0
while step < 100:
	error = 0
	for i in range(len(X)):
		x = X[i][0]
		y = X[i][1]
		label = labels[i]
		f = k*x + b - y
		l = tanh(f)
		k = k - 0.1 * 2 * (l - label) * derivate_tanh(f) * x
		b = b - 0.1 * 2 * (l - label) * derivate_tanh(f)
		error = error + (l - label)
	print(error/len(X))
	step = step + 1

plt.plot(X[0:4,0], X[0:4,1], marker=".", linestyle="")
plt.plot(X[4:, 0], X[4:, 1], marker=".", linestyle="")
plt.plot(numpy.arange(7), k * numpy.arange(7) + b)
plt.show()
