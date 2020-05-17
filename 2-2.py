import numpy
import matplotlib.pyplot as plt

XY = numpy.random.rand(50, 2) + numpy.array([1,1]) 
#(numpy.arange(100) + 1).reshape(50,2)
L = []
for i in range(len(XY)):
	M = numpy.array([[2, 6.9], [0.8,2]])
	l = M.dot(XY[i]) + numpy.array([0.4,8.9])
	L.append(l[0]/l[1])
L = numpy.array(L)

def active(x, y):
	return x/y
step = 0
W = numpy.array([[1, 0], [0, 1]])
B = numpy.zeros((2))
L_cur = []
while step < 1000:
	erros = []
	L_cur = []
	for i in range(len(XY)):
		xy = XY[i]
		l = L[i]
		xy_cur = W.dot(xy) + B
		xy_curw = numpy.array([[xy_cur[0]], [-xy_cur[0]]])
		l_cur = active(xy_cur[0], xy_cur[1])
		L_cur.append(l_cur)
		erros.append((l_cur - l) ** 2)
		c = 2 * (l_cur - l)/(xy[1] ** 2)
		W = W - 0.01 * c * xy_curw.dot(xy.reshape(1,2))
		B = B - 0.01 * c * xy_curw.reshape(2)
	step = step + 1
	if step % 100 == 0:
		err = numpy.sum(erros)/len(erros)
		print(err)
print(W, B)
X = numpy.arange(len(L)) + 1
plt.xlim(0, len(L) + 2)
plt.ylim(0, numpy.max(L) * 2)
plt.plot(X, L, linestyle="", marker=".")
plt.plot(X, L_cur, linestyle="", marker="+")
plt.show()
