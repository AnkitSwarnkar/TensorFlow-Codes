# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:56:31 2018
"""
import tensorflow as tf
import numpy as np
from basic_mnist import load_data

data_ = load_data()
X_train, Y_train, X_test,Y_test = data_.return_data()
print("[INFO] Data Loaded")
learning_rate = 0.01
epoch = 100
batch_size = 32
#Scale the data
X_train = X_train/255.0
X_test = X_test/255.0

Y_train_ = np.argmax(Y_train,1)
n_class = 10
n_feature = X_train.shape[1]

X_batch = tf.placeholder(shape=[None,n_feature],dtype=tf.float32)
Y_batch = tf.placeholder(shape=[None,1],dtype=tf.float32)

W = tf.Variable(np.random.randn(n_feature,1),dtype=tf.float32)

b = tf.Variable(np.random.randn(1),dtype=tf.float32)

#XW+b
y  = tf.matmul(X_batch,W) + b
loss_ = tf.reduce_mean(tf.squared_difference(y,Y_batch))
#loss_2 = tf.reduce_sum(tf.pow(y-Y_batch,2))/(2 * tf.shape(y)[0])
opt = tf.train.GradientDescentOptimizer(learning_rate)

obj = opt.minimize(loss_)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def get_samples(X_train,Y_train, batch_size = 32):
    len_ = X_train.shape[0]
    for idx  in range(0,len_,batch_size):
        xtrain = X_train[idx:idx+batch_size,:]
        ytrain = Y_train[idx:idx+batch_size,]
        ytrain = np.reshape(ytrain,(ytrain.shape[0],1))
        yield (xtrain,ytrain)

for ep_ in range(epoch):
    loss_v = []
    for x_train,y_train in get_samples(X_train,Y_train_,batch_size=batch_size):   
        _,loss_val = sess.run([obj,loss_],feed_dict={X_batch:x_train,Y_batch:y_train})
        loss_v.append(loss_val)
    if ep_ % 10 ==0:
        print("Loss at End of Epoch ",ep_," : ",np.mean(loss_v))

print("Optimization Done")
training_loss = sess.run(loss_, feed_dict={X_batch: X_train, 
                                           Y_batch:  np.reshape(Y_train_,(Y_train_.shape[0],1))}
                        )

Y_test_ = np.argmax(Y_test,1)

testing_loss = sess.run(loss_, feed_dict={X_batch: X_test, 
                                          Y_batch:  np.reshape(Y_test_,(Y_test_.shape[0],1))}
                        )






