# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:10:09 2018

@author: ankitswarnkar
"""
import numpy as np
import tensorflow as tf
from basic_mnist import load_data

data_ = load_data()
X_train, Y_train, X_test,Y_test = data_.return_data()
learning_rate = 0.01
batch_size = 32

#Scale the data
X_train = X_train/255.0
X_test = X_test/255.0
n_class = 10
n_feature_ = X_train.shape[1]

X = tf.placeholder(dtype=tf.float32,shape=(None,n_feature_))
Y = tf.placeholder(dtype=tf.float32,shape=(None,n_class))
W = tf.Variable(np.random.randn(n_feature_,n_class),dtype=tf.float32)
b = tf.Variable(np.random.rand(),tf.float32)

logit_ = tf.matmul(X,W) + b
pred = tf.nn.sigmoid(logit_)


loss_ = tf.reduce_mean(-1 * tf.reduce_sum(Y*tf.log(pred),reduction_indices=1))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(Y,1)),tf.float32))

opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
obj = opt.minimize(loss_)

def get_batches(X_train,Y_train,batch_size=32):
    len_ = X_train.shape[0]
    for idx in range(0,len_,batch_size):
        xtrain = X_train[idx:idx+batch_size,:]
        ytrain = Y_train[idx:idx+batch_size,:]
        yield (xtrain,ytrain)

epoch_ = 100
sess = tf.Session()
init_var = tf.global_variables_initializer()
sess.run(init_var)

for ep_ in range(epoch_):
    loss_epch = []
    acc = 0.0
    for batch_x, batch_y in get_batches(X_train,Y_train,batch_size=batch_size):
        _,loss_val,acc = sess.run([obj,loss_,accuracy],feed_dict={X:batch_x,Y:batch_y})
        loss_epch.append(loss_val)
    if ep_ % 10 ==0:
        print("Epoch Loss",ep_," : ",np.mean(loss_epch)," Acc: ",acc)
sess.run(accuracy,feed_dict={X:X_train,Y:Y_train})