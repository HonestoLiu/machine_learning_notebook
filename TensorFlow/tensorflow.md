- 训练神经网络的过程：  
1. 定义神经网络的结构+前向传播的输出结果  
2. 损失函数+反向传播的优化算法  
3. 生成会话+反复训练  

- 损失函数：  
1. 分类：交叉熵$-\sum p_i logq_i$+`softmax` -->  `tf.nn.softmax_cross_entropy_with_logits(labels=?, logits=?)`  
2. 回归：均方误差（MSE, mean squared error）--> `tf.reduce_mean(tf.square(pre-true))`  
3. 自定义损失函数：`loss = tf.reduce_sum(tf.where(tf.greater(v1,v2),func1(v1,v2),func2(v1,v2)))`  

- 优化算法
1. 梯度下降法：`局部最优`+`收敛速度慢` --> 随机梯度下降（stochastic gradient descent，每轮迭代优化随机选取一条训练数据） --> minibatch gradient descent(不稳定）
2. 动量Momentum：在一定程度上保留之前更新的方向，同时利用当前batch的梯度微调最终的更新方向 --> `稳定`+`可能跳出局部最优`  
3. Adam(Adaptive Moment Estimation): Adam算法根据损失函数对每个参数的梯度的一阶矩估计和二阶矩估计动态调整针对于每个参数的学习速率  

- 学习率的设置  
指数衰减法，先使用较大的学习率快速得到一个比较优的解，然后逐步减小学习率进行微调 `tf.train.exponential_decay(learning_rate, global_step, decay_steps, and decay_rate)  )`  

- 滑动平均模型  
通过控制衰减率来控制参数更新前后之间的差距，从而达到减缓参数的变化值.  
1. 滑动平均类： `ema  = tf.train.ExponentialMovingAverage(0.99,step)`  
2. 定义变量的滑动平均操作： `maintain_averages_op = ema.apply([var])`  
3. 获得滑动平均之后变量的取值： `ema.average(var)`  

- 正则化  
`regu_loss = tf.contrib.layers.l2_regularizer(lambda).(var)` +  `tf.contrib.layers.l1_regularizer(lambda).(var)`  
1. 添加损失函数到损失函数集合`losses`: `tf.add_to_collection('losses', regu_loss)`  
2. 总损失函数：`loss = tf.add_n(tf.get_collection('losses'))`, get_collection返回一个列表  

- 常用激活函数  
1. sigmoid `梯度消失`+`输出非零均值`+`幂运算耗时`  
2. tanh `梯度消失`+`幂运算耗时`  
3. Relu `速度快`+`死神经元` --> Leaky Relu 


- 接触过得tf函数  
1. `tf.clip_by_value()`
2. `tf.nn.softmax_cross_entropy_with_logits` \ `tf.sparse_softmax_cross_entropy_with_logits()`  
3. `tf.train.GradientDescentOptimizer`+` tf.train.MomentumOptimizer`+`tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-08)`  


