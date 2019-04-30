# 机器学习  
## 模型评估与选择  
机器学习算法在学习过程中对某种类型假设（未出现的样本？）的偏好，称为“归纳偏好”，任何有效的机器学习算法必然有其归纳偏好。谈论算法的相对优劣，必然要针对具体问题，归纳偏好和问题越匹配，效果越好。学习器在训练集上的误差称为**经验误差**，在新样本上的误差称为**泛化误差**。我们总是希望得到泛化误差小的分类器，而我们通常假设训练样本和新样本符合独立同分布，极小化经验误差在一定程度上是在极小化泛化误差。过拟合是无法避免的，其产生的原因主要是学习器的学习能力过于强大。  
模型的选择和评估： 选择测试集上的“测试误差”作为泛化误差的近似。数据集的选择方法：留出法、交叉验证法（p次k折交叉验证取均值）、有放回采样。在测试集上度量性能的方法：
- 错误率、准确率
$$accuracy=\frac{1}{m}\sum_{i=1}^m I(f(x_i)=y_i)$$
- 查准率、查全率、F1  
混淆矩阵：

|真实情况-预测结果|正例|反例|
|-|-|-|
|正例|TP(真正例)|FN(假反例)|
|反例|FP(假正例)|FN(真反例)|

查准率$P=\frac{TP}{TP+FP}$，查全率$R=\frac{TP}{TP+FN}$，折中度量$\frac{1}{F1}=\frac{1}{2}(\frac{1}{R}+\frac{2}{P}).$  
对于多次试验得到多个混淆矩阵，先算P-R后平均的称**macro**，先平均后算P-R的称**micro**.

- ROC、AUC  
预测值是实值或者概率的情况下，逐个把样本预测为正例，用“真正例率”为纵轴，“假正例率为”横轴，可画出roc曲线，其下面积为$auc=1-l_{rank}.$

## 感知器
二分类的线性模型：$f(x)=sign(w \cdot x+b)$,策略是极小化损失函数：$min_{w,b} L(w,b)=-\sum_{x_i \in M}y_i (w_i \cdot x_i+b)$，M是误分类样本集合，通常采用随机梯度下降法求解.

## k近邻
多数表决分类策略
$$for (x,y),$y=argmax_{c_j}\sum_{x_i \in N_k(x)} I(y_i=c_j)$$

## 朴素贝叶斯
朴素贝叶斯是典型的生成学习方法，目标是学习联合概率分布P(X,Y)，进而求得后验概率分布P(Y|X).其基本假设是属性的条件独立性，即$P(X=x|Y=c_k)=\prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k).$  
$$P(Y|X)=\frac{P(X,Y)}{P(Y)}=\frac{P(Y)P(X|Y)}{\sum_Y P(Y)P(X|Y)}$$ 预测时$y=argmin_{c_k}P(Y=c_k)\prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k).$  估计类条件概率时，常使用类条件概率估计假设分布的参数，使用拉普拉斯平滑解决概率为0的情形.

## Tree分类回归树
针对离散特征的分类树：
1. 特征选择：信息增益、信息增益比、基尼指数
2. 决策树的生成：递归生成 ID3(信息增益)、C4.5(信息增益比)、CART(基尼指数)
3. 剪枝：决策树学习的损失函数 $C_{\alpha}(T) = \\sum_{t=1}^{|T|}N_tH_t(T)+\alpha|T|$ ,其中$|T|$是叶子节点数，$N_t$是该叶子节点的样本数，$H_t$是熵.
对每个叶子节点递归操作，直到不能继续为止.
- 对连续特征使用二分法化成离散特征.
- CART回归树  
选取某个特征$x_j$及其切分点s，将样本二分，求解$min_{j,s}$.

## LR逻辑斯蒂回归
- 二分类  
$P(Y=1|x)=\frac{e^{w\cdot x}}{1+e^{w\cdot x}}$  
$P(Y=0|x)=\frac{1}{1+e^{w\cdot x}}$
- 多分类
$P(Y=k|x)=\frac{e^{w_k \cdot x}}{1+\sum_{k=1}^{K} e^{w_k \cdot x}}, k=1,2,\cdots,K-1$  
$P(Y=K|x)=\frac{1}{1+\sum_{k=1}^{K} e^{w_k \cdot x}}$  
其中$x \in \mathrm{R}^{n+1},w_k \in \mathrm{R}^{n+1}.$
- 使用似然函数求解参数w  
$L(w)=\prod_{i=1}^N P(Y=1|x)^{y_i} P(Y=0|x)^{1-y_i}$  
对数似然函数 $LL(w)=log(L(w)).$  
使用梯度下降法或者拟牛顿法求解w.

## svm支撑向量机
- 最基本模型-最大化间隔
$$min_{w,b} \ \ \frac{1}{2}||w||^2$$
$$s.t. \ \ y_i(w^{T}x_i+b) \geq 1,\ i=1,2,\cdots,m.$$
- 对偶问题
$$max_{\alpha} \ \ \sum_{i=1}^m \alpha_i-\frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^{T}x_j$$
$$s.t. \ \ \sum_i \alpha_i y_i=0,\ \alpha_i \geq 0,\ i=1,2,\cdots,m.$$
高效算法SMO，其思想每次选取$\alpha_i,\alpha_j$并固定其他参数进行优化.
- 核方法
$$max_{\alpha} \ \ \sum_{i=1}^m \alpha_i-\frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j \mathcal{k}(x_i,x_j)$$
$$s.t. \ \ \sum_i \alpha_i y_i=0,\ \alpha_i \geq 0,\ i=1,2,\cdots,m.$$
- 软间隔  
替代$l_{0/1}$损失函数：hinge损失$max(0,1-z)$、指数损失$e^{-z}$、对率损失$log(1+e^{-z})$.
$$max_{\alpha} \ \ \sum_{i=1}^m \alpha_i-\frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^{T}x_j$$
$$s.t. \ \ \sum_i \alpha_i y_i=0,\ 0 \leq \alpha_i \leq C,\ i=1,2,\cdots,m.$$

## Adaboost
Adaboost提高那些被前一轮分类器错误分类样本的权重；多分类器的组合，Adaboost采取加权多数表决的方法.
1. 权值分布：$D_m=(w_{m1},w_{m2},\cdots,w_{mN})$，计算第m个基分类器：
$$G_m(x): \mathcal{X} \to \{ -1,+1 \}$$
2. 分类误差：$e_m=\sum w_{mi}I(G_m(x_i) \\neq y_i)$,基分类器的权重系数：$\alpha_m=\frac{1}{2}ln(\frac{1-e_m}{e_m})$.
3. 更新权重：$w_{m+1,i}=\frac{w_{mi}e^{-\alpha_m y_i G_m(x_i))}}{Z_m}$，其中$Z_m=\sum w_{mi}e^{-\alpha_m y_i G_m(x_i)}.$
4. 最终分类器：$f(x)=\sum \alpha_m G_m(x_i), \ G(x)=sign(f(x)).$  
Adaboost算法的另一个解释，即认为Adaboost算法是加法模型、损失函数为指数函数、学习算法为“前向分步”时的二分类学习方法。
- GBDT算法  
以决策树为基函数的提升方法称为提升树。回归问题的提升树（平方误差损失函数）策略很简单：每步使用回归树拟合残差。损失函数是平方损失和指数损失函数时，每一步优化是简单的。对一般损失函数而言，采应梯度提升算法（gradient boosting），即使用$- \[ \frac{\partial L(y,f(x_i))}{\partial f(x_i)} \]$作为残差的近似值，拟合一个回归树。  
Adaboost个体学习器间存在强依赖关系，需要串行序列化生成.不存在强依赖关系、同时并行生成的方法有Bagging、Random Forest. Bagging每次随机采样训练样本得到基分类器，再将基分类器结合.RF采用决策树作为基分类器，决策树训练过程中引入了随机属性选择，即每次随机选取属性子集构建决策树.


