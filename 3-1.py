import numpy
W = numpy.array([[1, 0], [0, 1]])
def train(a, b):
	global W
	a = numpy.array(a, ndmin=2).T
	b = numpy.array(b, ndmin=2).T
	step = 0
	error = 0
	while step < 100:
		c = numpy.dot(W, a)
		e = c - b
		offset = numpy.dot(e, a.T)
		W = W -  0.1 * offset
		step += 1
		error = numpy.sum(e)/len(e)
	print(error, W)
def predict(a):
	a = numpy.array(a, ndmin=2).T
	return numpy.dot(W, a).T
train([1, 2], [2,4])
train([2, 3], [4,6])
train([1,0], [2, 0])
train([0,1], [0, 2])
train([0, 100], [0, 200])

print("predict", predict([100,11]))
