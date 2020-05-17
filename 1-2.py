import numpy
import matplotlib.pyplot as plt

XY = numpy.array([[1,1], [2, 2], [3, 3]])
UV = numpy.array([[3,3], [5, 5], [7, 7]])

W = numpy.zeros((2, 2))
T = numpy.zeros((2))
step = 0
while step < 200:
	errors = []
	for i in range(len(XY)):
		xy = XY[i]
		uv = UV[i]
		uv_cur = W.dot(xy) + T
		err = (2 * (uv_cur - uv))
		err_1x2 = err.reshape(1, 2)
		W = W - 0.1 * err_1x2.T * xy.reshape(1, 2)
		T = T - 0.1 * err
		
		errors.append(numpy.sum((uv_cur - uv) ** 2))
	step = step + 1
	if step % 10 == 0:
		print(numpy.sum(errors)/len(errors))

plt.plot(UV[:,0], UV[:,1], linestyle="", marker=".")
Y = []
for i in range(len(XY)):
	x = XY[i]
	Y.append(W.dot(x) + T)
Y = numpy.array(Y)
plt.plot(Y[:,0], Y[:,1], linestyle="", marker="+")
plt.show()
