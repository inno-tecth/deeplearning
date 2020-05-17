import numpy
import matplotlib.pyplot as plt
X = numpy.array([1, 2, 3])
Y = numpy.array([1, 2, 3])
k = 0
b = 0
step = 0
rate = 0.1
while step < 100:
	errors = []
	for i in range(len(X)):
		x = X[i]
		y = Y[i]
		k = k - rate * 2 * ((k*x + b) - y) * x
		b = b - rate * 2 * ((k*x + b) - y)
		errors.append((k*x + b - y)**2)
	step = step + 1
	if step % 10 == 0:
		print(numpy.sum(errors)/len(errors))

plt.plot(X, Y, linestyle="", marker="*")
plt.plot(numpy.arange(5), k * numpy.arange(5) + b)
plt.show()
