import numpy as np
import tensorflow as tf

#create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3   #0.1 goal weight 0.3 goal bias

##creat tensorflow structure start##
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)    #0.5 is learning rate
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
##creat tensorflow structure end##

sess=tf.Session()
sess.run(init)  #激活網絡跑到init位置

for step in range(201): #迭代201次
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))
    

'''
应该说是一次神经网络的会话,
在这次会话里可以让 session run 你想要指向 神经网络图片中运行的地方﻿
'''
'''
a=
'''