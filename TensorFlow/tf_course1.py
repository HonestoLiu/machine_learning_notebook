# -*- coding: utf-8 -*-#
# Name:         tf_course1
# Author:       liuhong
# Date:         2019/4/2

import tensorflow as tf
from numpy.random import RandomState

# 网络参数：权重
w1 = tf.Variable(tf.random_normal([2,3], stddev=1.0, dtype=tf.float32, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1.0, dtype=tf.float32, seed=1))

# 网络输入输出
x = tf.placeholder(tf.float32, shape=(None, 2), name='Input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='Output')

# 网络结构--前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 损失函数
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                                (1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10,1.0)))

# 优化算法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 生成数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]


steps = 5000    #设定循环训练轮次
batch_size = 10
config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333))
with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # 查看初始参数
    print("参数w1: \n",sess.run(w1))
    print("参数w2: \n", sess.run(w2))

    # 训练轮次
    for i in range(steps):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        # 训练神经网络
        sess.run(train_step,feed_dict={x: X[start:end], y_: Y[start:end]})
        # 打印交叉熵
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training steps, cross entropy on all data is %g" %(i, total_cross_entropy))

    print(sess.run(w1))
    print(w2.eval())