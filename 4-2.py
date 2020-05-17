import numpy

class NeuralNetwork:
	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes
		self.wi_h = numpy.eye(hidden_nodes, input_nodes)
		self.wh_o = numpy.eye(output_nodes, hidden_nodes)
		self.active = lambda x: 1.0/(1.0 + numpy.exp(-x))
		
	def train(self,inputs, outputs):
		inputs = numpy.array(inputs, ndmin=2).T
		outputs = numpy.array(outputs, ndmin=2).T 
		hidden_inputs= numpy.dot(self.wi_h, inputs)
		hidden_outputs = self.active(hidden_inputs)
		final_inputs = numpy.dot(self.wh_o, hidden_outputs)
		final_outputs = self.active(final_inputs)
		
		final_errors = final_outputs - outputs
		hidden_errors = numpy.dot(self.wh_o.T, final_errors)
		
		self.wh_o = self.wh_o - 0.1* numpy.dot(final_errors * final_outputs * (1.0 - final_outputs), hidden_outputs.T)
		self.wi_h = self.wi_h - 0.1* numpy.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), inputs.T)
		self.error = numpy.sum(final_errors)/len(final_errors)
	
	
	def predict(self, inputs):
		inputs = numpy.array(inputs, ndmin=2).T
		h_i = numpy.dot(self.wi_h, inputs)
		h_o = self.active(h_i)
		f_i = numpy.dot(self.wh_o, h_o)
		f_o = self.active(f_i)
		return f_o.T


nn = NeuralNetwork(784, 200, 10)
file = open("mnist-dataset/mnist-train-100.csv")
training_list = file.readlines()
file.close()

for e in range(5):
	for line in training_list:
		all_values = line.split(",")
		inputs = numpy.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
		targets = numpy.zeros(10) + 0.01
		targets[int(all_values[0])] = 0.99
		nn.train(inputs, targets)
		pass
	print("error:", nn.error)
	pass
file = open("mnist-dataset/mnist-test-10.csv")
test_list = file.readlines()
file.close()
score_card = []
for line in test_list:
	all_values = line.split(",")
	label = int(all_values[0])
	inputs = numpy.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
	ouputs = nn.predict(inputs)
	result = numpy.argmax(ouputs)
	if result == label:
		score_card.append(1)
	else:
		score_card.append(0)
	pass

score_card = numpy.array(score_card)
print(numpy.sum(score_card)/len(score_card))



