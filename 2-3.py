import numpy
import matplotlib.pyplot as plt

D3 = numpy.random.rand(30, 3) * 2 + numpy.array([1, 1, 1])
D2 = []
for i in range(len(D3)):
	M = numpy.array([[2, 2, 0.3], [1,2, 0.4], [1.1,0.2,2]])
	d3 = M.dot(D3[i]) + numpy.array([3,2,1])
	D2.append([d3[0]/d3[2], d3[1]/d3[2]])
D2 = numpy.array(D2)

def active(x, z):
	return x/z

step = 0
W = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
B = numpy.zeros(3)
D2_cur = []
while step < 50000:
	erros = []
	D2_cur = []
	for i in range(len(D3)):
		d3 = D3[i]
		l = D2[i]
		nd3 = W.dot(d3) + B
		nxz = active(nd3[0], nd3[2])
		nyz = active(nd3[1], nd3[2])
		D2_cur.append([nxz, nyz])
		nz = nd3[2]
		c1 = 2 * (nxz - l[0])/(nz ** 2)
		c2 = 2 * (nyz - l[1])/(nz ** 2)
		erros.append( ((nxz - l[0]) ** 2)/2 + ((nyz - l[1])**2)/2  )
		b = numpy.array([
			c1 * nz, 
			c2 * nz,  
			-c1 * nd3[0] - c2 * nd3[1]
		])
		w = b.reshape(3, 1).dot(d3.reshape(1, 3))
		W = W - 0.1 * w
		B = B - 0.1 * b
	step = step + 1
	if step % 1000 == 0:
		err = numpy.sum(erros)/len(erros)
		print(err)
	
D2_cur = numpy.array(D2_cur)
plt.plot(D2[:,0], D2[:,1], linestyle="", marker=".")
plt.plot(D2_cur[:,0], D2_cur[:, 1], linestyle="", marker="+")
plt.show()
