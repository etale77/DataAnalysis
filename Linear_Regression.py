import theano
import theano.tensor as T
from theano import shared
import numpy as np



############### Linear Mini-Batch SGD #######################
N=20 #Size of dataset
A=3
B=2
rate =0.0005

variation = np.random.normal(size=(1,N))
data_x = np.array([[i for i in range(N)],[1 for i in range(N)]])
data_y = np.array([[A*i+B for i in range(N)]]) + variation

x = T.matrix()
y = T.matrix()
w = theano.shared(np.array([[2.0,4.0]]))


loss = ((T.dot(w,x)-y).norm(2))/N
d_loss = T.grad(loss,w)
update = [(w, w - rate * d_loss)]


f = theano.function([x,y],loss, updates=update)

while True:
  a = f(data_x,data_y)
  print(a)
  if a<0.3:
    break
print(w.get_value())
