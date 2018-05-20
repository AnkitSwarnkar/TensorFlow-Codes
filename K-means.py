# -*- coding: utf-8 -*-
"""
K-means algorithm using Tensorflow and MNIST data

"""
from basic_mnist import load_data
import numpy as np
import tensorflow as tf

#Read the data

data_ = load_data()
X_train, Y_train, X_test,Y_test = data_.return_data()
print("[INFO] Data Loaded")

epoch_steps = 1000
batch_size = 32
num_class = 10
n_points = X_train.shape[0]
n_features = X_train.shape[1]
cluster_assingments = tf.Variable(tf.zeros([n_points],dtype=tf.int16))

#initialize cluster
#Select random points
start_pos = tf.Variable(X_train[np.random.randint(n_points,size=num_class),:],
                                dtype=tf.float32)
centroids = tf.Variable(start_pos.initialized_value(),dtype=tf.float32)

prev_assigment = tf.Variable(tf.zeros((n_points,),dtype=tf.int64))

# find the distance between all points: 
    #http://stackoverflow.com/a/43839605/1090562
#generate (n,k) dim=vector
'''
Idea: (a-b)**2 = a**2 + b**2 - 2 * a* b
'''
p1 = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(X_train),1),1),
               tf.ones((1,num_class)))

# Another n,k
p2 = tf.transpose(tf.matmul(
                   tf.expand_dims(
                    tf.reduce_sum(tf.square(centroids),1),1),
                                  tf.ones(shape=(1,n_points))
                          )
                   )
distance_ = p1 + p2 - 2 * tf.matmul(X_train,centroids,transpose_b=True)

cluster_assingments = tf.argmin(distance_,axis = 1)

#Recalculate Center

sum_new_centroid = tf.unsorted_segment_sum(X_train,cluster_assingments,
                                           num_class)
count_new_centroid = tf.unsorted_segment_sum(tf.ones((n_points,1)),
                                             cluster_assingments,num_class)

new_means = sum_new_centroid/count_new_centroid

boolean_change = tf.reduce_any(tf.not_equal(cluster_assingments,prev_assigment))

with tf.control_dependencies([boolean_change]):
    loop = tf.group(prev_assigment.assign(cluster_assingments),
                                        centroids.assign(new_means))

#Start Session Here
sess = tf.Session()
sess.run(tf.global_variables_initializer())

has_changed, cnt = True, 0

while has_changed and cnt < 30:
    cnt += 1
    has_changed,_ = sess.run([boolean_change,loop])
    
res = sess.run(cluster_assingments)

y_true= np.argmax(Y_train,axis=1)
correct =0
for result,true_ in zip(res,y_true):
    if result == true_:
       correct +=1

print("Correct Points",correct)

