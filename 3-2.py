import numpy

class NeuralNetwork:
	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes
		self.wi_h = numpy.eye(hidden_nodes, input_nodes)
		self.wh_o = numpy.eye(output_nodes, hidden_nodes)
	def train(self,inputs, outputs):
		inputs = numpy.array(inputs, ndmin=2).T
		outputs = numpy.array(outputs, ndmin=2).T 
		hidden_outputs = numpy.dot(self.wi_h, inputs)
		final_outputs = numpy.dot(self.wh_o, hidden_outputs)
		
		final_errors = final_outputs - outputs
		hidden_errors = numpy.dot(self.wh_o.T, final_errors)
		
		self.wh_o = self.wh_o - 0.01* numpy.dot(final_errors, hidden_outputs.T)
		self.wi_h = self.wi_h - 0.01* numpy.dot(hidden_errors, inputs.T)
		self.error = numpy.sum(final_errors)/len(final_errors)
	
	def predict(self, inputs):
		inputs = numpy.array(inputs, ndmin=2).T
		ho = numpy.dot(self.wi_h, inputs)
		return numpy.dot(self.wh_o, ho).T



def test():	
	A = [[1, 2], [2, 3], [1, 0], [2, 1]]
	B = [[2, 4], [4, 6], [2, 0], [4, 2]]
	nn = NeuralNetwork(2, 9, 2)
	
	step = 0
	while step < 1000:
		for i in range(len(A)):
			nn.train(A[i], B[i])
		step += 1
		print(nn.error)
	print(nn.predict([9, 11]))
	
def test2():	
	A = numpy.array([[3.2, 0.8], [5, 5], [3, 3], [4, 3], [1, 1], [2, 2], [1, 4]])
	B = numpy.array([1, 1, 1, 1, -1, -1, -1])
	nn = NeuralNetwork(2, 2, 1)
	
	step = 0
	while step < 1000:
		for i in range(len(A)):
			nn.train(A[i], B[i])
		step += 1
		if step % 100 ==0:
			print(nn.error)
	print(nn.predict([2, 2]))
	

test2()
