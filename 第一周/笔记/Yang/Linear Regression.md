# Linear Regression



## Least mean squares, LMS algorithm



### gradient descent

![image-20220413094341713](https://s2.loli.net/2022/04/13/NsZ8QemVKIuxbH2.png)

![image-20220413094455471](C:/Users/yangshuo/AppData/Roaming/Typora/typora-user-images/image-20220413094455471.png)

![image-20220413094556663](https://s2.loli.net/2022/04/13/maOhxgGW8tZUyIl.png)



### Batch gradient descent

![image-20220413094620336](https://s2.loli.net/2022/04/13/3f9GQkX4ibqLYEp.png)

looks at every example in the entire training set on every step





### Stochastic gradient descent (SGD)

![image-20220413094948060](https://s2.loli.net/2022/04/13/RCfwcPsBoTIFy3i.png)

In this algorithm, we repeatedly run through the training set, and each time
we  encounter  a  training  example,  we  update  the  parameters  according  to
the  gradient  of  the  error  with  respect  to  that  single  training  example  only.

Batch Gradient descent在开始一步时必须要扫描整个训练集,当数据集规模m非常大时这样的代价将非常大.而SGD可以立即开始,并在处理每个样本的时候都可以继续更新.因此SGD会比批量梯度下降更快接近最优点,当然也可能出现参数不会收敛到最小值点,而是在最小值点周围震荡.通过在训练过程中不断缩小学习率可以让参数更接近最小值点.



## The normal equations

### 一些数学记号

![image-20220413101453857](https://s2.loli.net/2022/04/13/cqnMrzjXmDVUI1p.png)

![image-20220413101509424](https://s2.loli.net/2022/04/13/pTo4iA8GCxLkas5.png)

一个例子:
![image-20220413101633708](https://s2.loli.net/2022/04/13/Cytre5HmsDFh9VB.png)

**trace**, *tr*

![image-20220413101846956](https://s2.loli.net/2022/04/13/Fye6oUiL2vxDjXK.png)

性质:

-   $tr AB = tr BA$
-   ![image-20220413102106009](https://s2.loli.net/2022/04/13/jPU9xNbmHFLSyO4.png)

![image-20220413102233739](https://s2.loli.net/2022/04/13/Vrwn7xjlieJbKyz.png)

![image-20220413102348156](https://s2.loli.net/2022/04/13/IxDgzEUq8mC5BrO.png)

![image-20220413103333023](https://s2.loli.net/2022/04/13/E5zZ6ITyWUxX2BG.png)

![image-20220413103355177](https://s2.loli.net/2022/04/13/A2CYIyclg94ZnTD.png)

![image-20220413103415405](https://s2.loli.net/2022/04/13/oVutGz6r8WH5QP7.png)

![image-20220413103449193](https://s2.loli.net/2022/04/13/jTxmbP2vJqnIO5e.png)

![image-20220413103701642](https://s2.loli.net/2022/04/13/6uNvZHWpQsgLKOU.png)

![image-20220413103733026](https://s2.loli.net/2022/04/13/cGaWHfR4XO5SKhv.png)



![image-20220413162841156](C:/Users/yangshuo/AppData/Roaming/Typora/typora-user-images/image-20220413162841156.png)

## Probabilistic interpretation

![image-20220413164849373](https://s2.loli.net/2022/04/13/gYsUpGhkFfudDeH.png)

![image-20220413164900464](https://s2.loli.net/2022/04/13/Irjq7yi3zROXoNs.png)

![image-20220413164914284](https://s2.loli.net/2022/04/13/bJEk2lqtvG8ZAKU.png)

### likelihood function

we should should choose θ so as to make the data as high probability as possible

![image-20220413165249084](https://s2.loli.net/2022/04/13/NlewDhVMPCLyxQq.png)

![image-20220413165440959](https://s2.loli.net/2022/04/13/csXInghMU61fYNK.png)

![image-20220413165819046](https://s2.loli.net/2022/04/13/L4Au8iWRQjHw2F6.png)

![image-20220413165847894](https://s2.loli.net/2022/04/13/sncrwiB4dCmzq1L.png) 

在数据上基于先前概率假设,最小二乘回归就相当于关于$\theta$的最大似然估计.因此这是一组假设,在这样的假设下,最小二乘回归是一种非常自然的方法,它实际上做的是极大似然估计.然而想要最小二乘估计成为一个非常完美的程序,关于概率的假设并不是必须的,而且可能确实存在其它自然的假设来被用于证明最小二乘估计.

在我们先前的讨论中,我们最终的关于$\theta$的选择不依赖于$\sigma^2$而且尽管$\sigma^2$未知,我们也能取得相同的结果.在后面谈及指数簇和更一般的线性模型时也会使用这样的一个事实.



## Locally weighted linear regression 局部加权线性回归

有时候使用线性函数并不能很好的拟合数据集,添加一个额外的平方项,拟合的效果会好一些.然而特征并非添加的越多越好,因为有些特征可能是噪音,造成过拟合,在训练集上表现很好,但是到测试集上,运行效果非常差.



-   Underfitting 欠拟合

模型没有捕捉到数据内部的结构

-   Overfitting过拟合

特征的选择对于确保学习算法的性能非常重要

![image-20220413202801180](https://s2.loli.net/2022/04/13/tizQXwABrqKvdnl.png)

如果权重w非常大,要选择一个合适的学习率$\theta$将损失变得很小;如果w非常小,那么平方损失项在拟合中将会被忽略.关于权重的一个选择是:

![image-20220413204040722](https://s2.loli.net/2022/04/13/TzgfbP3pLQMxFiA.png)

尽管权重的 公式和高斯分布的公式很像,但是它们之间没有任何关系,权重w并不是一个随机变量.参数$\tau$被称为带宽参数*bandwidth*,控制$x(i) 与 x$之间距离下降的速度.



**parametric vs. non-parametric**

parametric : it  has  a  fixed,  finite  number  of  parameters;一旦你和出来$\theta$就可以将它们存储起来,在将来的预测中不再需要训练数据.

non-parametric: 需要保存完整的训练集在进行预测时.我们需要保持的东西,为了假设,需要随着训练集线性增长.



