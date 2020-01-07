#coding:utf-8
# plot the losses generated in training phase

import numpy as np
import matplotlib.pyplot as plt
import pylab
 
def plotData(x, y):
  length = len(y)
 
  pylab.figure(1)
 
  pylab.plot(x, y, 'rx')
  pylab.xlabel('x')
  pylab.ylabel('y')
 
  pylab.show()
 
x = [float(l) for l in open("loss_log.txt")]

plt.plot(x)
plt.ylabel('loss')
plt.xlabel('iters')

plt.show()
 
 