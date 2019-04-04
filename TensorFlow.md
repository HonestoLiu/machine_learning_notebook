- 训练神经网络的过程：  
1. 定义神经网络的结构+前向传播的输出结果  
2. 损失函数+反向传播的优化算法  
3. 生成会话+反复训练  

- 损失函数：  
1. 分类：交叉熵$-\sump_i logq_i$+`softmax` -->  `tf.nn.softmax_cross_entropy_with_logits(labels=?, logits=?)`  
2. 回归：均方误差（MSE, mean squared error）--> `tf.reduce_mean(tf.square(pre-true))`  
3. 自定义损失函数：`loss = tf.reduce_sum(tf.where(tf.greater(v1,v2),func1(v1,v2),func2(v1,v2)))`  

- 优化算法
1. 梯度下降法：`局部最优`+`收敛速度慢` --> 随机梯度下降（stochastic gradient descent，每轮迭代优化随机选取一条训练数据） --> minibatch gradient descent(不稳定）
2. 动量Momentum：在一定程度上保留之前更新的方向，同时利用当前batch的梯度微调最终的更新方向 --> `稳定`+`可能跳出局部最优`  
3. Adam(Adaptive Moment Estimation): Adam算法根据损失函数对每个参数的梯度的一阶矩估计和二阶矩估计动态调整针对于每个参数的学习速率  

- 学习率的设置  
指数衰减法，先使用较大的学习率快速得到一个比较优的解，然后逐步减小学习率进行微调 `tf.train.exponential_decay(learning_rate, global_step, decay_steps, and decay_rate)  )`

- 正则化  
巴拉巴拉。

- 接触过得tf函数  
1. `tf.clip_by_value()`
2. `tf.nn.softmax_cross_entropy_with_logits` \ `tf.sparse_softmax_cross_entropy_with_logits()`  
3. `tf.train.GradientDescentOptimizer`+` tf.train.MomentumOptimizer`+`tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-08)`  


