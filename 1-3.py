import numpy
import matplotlib.pyplot as plt

XY = numpy.array([[1,1,1], [2,2,3], [3,3,4]])
UV = numpy.array([[3,3,4], [5,5,6], [7,7,8]])

W = numpy.zeros((3, 3))
T = numpy.zeros((3))
step = 0
while step < 1000:
	errors = []
	for i in range(len(XY)):
		xy = XY[i]
		uv = UV[i]
		uv_cur = W.dot(xy) + T
		err = (2 * (uv_cur - uv))
		err_1x2 = err.reshape(1, 3)
		W = W - 0.01 * err_1x2.T * xy.reshape(1, 3)
		T = T - 0.01 * err
		
		errors.append(numpy.sum((uv_cur - uv) ** 2))
	step = step + 1
	if step % 100 == 0:
		print(numpy.sum(errors)/len(errors))
