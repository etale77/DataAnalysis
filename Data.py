import cPickle 
import gzip
import theano
import theano.tensor as T
from theano import shared
import numpy as np


################## Prep Data ###################################
"""f = gzip.open(’mnist.pkl.gz’, ’rb’)
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def shared_dataset(data_xy):
  data_x, data_y = data_xy
  shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
  shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
  return shared_x, T.cast(shared_y, ’int32’)
  
test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)
batch_size = 500 # size of the minibatch

data = train_set_x[2 * batch_size: 3 * batch_size]
label = train_set_y[2 * batch_size: 3 * batch_size]"""
#################################################################




############### Linear Mini-Batch SGD #######################
N=20 #Size of dataset
A=3
B=2
rate = 0.01

variation = np.random.normal(size=(N,1))
data_x = np.array([[i for i in range(N)],[1 for i in range(N)]])
data_y = np.array([A*i+B for i in range(N)]) + variation

x = T.matrix()
y = T.matrix()
w = theano.shared(np.random.normal(size=(1,2)))


loss = T.sum(T.dot(w,x) - y)
d_loss = T.grad(loss,w)
update = [(w, w - rate * d_loss)]

f = theano.function([x,y],loss, updates=update)

while True:
  a = f(data_x,data_y)
  print(a)
  if a<0.001:
    break
print(w)
















