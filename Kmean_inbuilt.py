# -*- coding: utf-8 -*-
"""
Created on Fri May 18 09:50:35 2018

@author: ankitswarnkar
"""
import tensorflow as tf
import numpy as np
from basic_mnist import load_data
from tensorflow.contrib.factorization import KMeans

"""
We will use mini batch Version of Kmeans
"""


data_ = load_data()
X_train, Y_train, X_test,Y_test = data_.return_data() #Resturn one hot

batch_size = 32
epoch = 30 #Increase epoch
n_class =10
n_feature = X_train.shape[1]

#Input Images
X_data = tf.placeholder(shape=[None,n_feature],dtype=tf.float32)
Y_data = tf.placeholder(shape =[None,n_class],dtype=tf.float32)

#Kmeans
kmeans_ = KMeans(inputs=X_data,num_clusters=n_class,use_mini_batch=True,
       distance_metric='cosine')

(all_scores, cluster_idx, scores, cluster_centers_initialized,
 init_op,train_op)=kmeans_.training_graph()

avg_distance = tf.reduce_mean(scores) #Intra clusters

init_vars = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_vars,feed_dict={X_data:X_train})
sess.run(init_op,feed_dict={X_data:X_train})#Graph init
#Epoch
for  i in range(epoch):
    _,avg_d = sess.run([train_op,avg_distance],feed_dict={X_data:X_train})
    if i % 5 ==0:
        print("IntraCluster Distance Step ",i," :",avg_d)

idx = sess.run(cluster_idx,feed_dict={X_data:X_train})

y_true= np.argmax(Y_train,axis=1)

correct =0
for result,true_ in zip(idx[0],y_true):
    if result == true_:
       correct +=1
       
print("Correct Points",correct)
