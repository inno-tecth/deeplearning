import numpy
#sigmoid
def sigmoid(z):
	return 1.0 / (1.0 + numpy.exp(-x))

def sigmoid_derivative(z):
	return sigmoid(z) * (1 - sigmoid(z))

#tanh
def tanh(z):
	return numpy.tanh(z)
def tanh_derivative(z):
	return 1 - numpy.square(tanh(z))

#relu
def relu(z):
	return (numpy.abs(z) + z)/2
def relu_derivative(z):
	return numpy.where(z > 0, 1, 0)


