# master_degree_paper_arrangement
master degree paper arrangement

# 论文内容组织 #

题目---基于深度学习模型RNN的（及其）改进技术（模型）在文本分类领域的研究（暂定是这个题目）（中文分词也可以提一下）
##1.摘要

##2.绪论

###2.1 课题研究背景和研究意义

###2.2 国内外研究现状

###2.3 论文的主要工作

##0. plus 相关研究简介
###0.1 文本表示

###0.2 分类器

###0.3 深度学习

##3.文本的分类过程和深度学习方法介绍

##4.文本表示方法 采用wordembedding或者其他的

##5.基于Attention based model的LSTM 方法原理

###5.1 Attention based model 原理

资料参考：

    知乎简介 --- https://www.zhihu.com/question/36591394
    国外的介绍资料自己翻译一下 --- http://yanran.li/peppypapers/2015/10/07/survey-attention-model-1.html
    鼻祖论文 --- 《Neural Machine Translation by Jointly Learning to Align and Translate》

    内容：	在encoder-decoder模型中，将长文本或者句子编码成固定长度的向量


###5.2 

##6.实验与结果分析

##7.总结与展望

## 参考文献

## 致谢

## 硕士期间的论文和专利发表

## 硕士期间参与的项目


##RECURRENT	 NEURAL NETWORK TUTORIAL

Part 1 RNN简介

Part 2 使用Python Numpy 和 Theano实现一个RNN

Part 3 BPTT和梯度消失问题

part 4 使用Python和Theano实现一个GRU/LSTM 的RNN

LSTM是如何计算隐层状态的：http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 这篇文章说的很详细需要翻译出来（简书上已经有译文了：http://www.jianshu.com/p/9dc9f41f0b29）

RNN取得成功的例子（语音识别，语言建模，翻译，图片描述）：http://karpathy.github.io/2015/05/21/rnn-effectiveness/

LSTM是个更好的模型，没有长期依赖问题的出现，什么是长期以来问题？

长期依赖问题：译文简书中有提到类似的概念

**记住长期的信息是LSTM的默认行为，而非需要很大代价才能获得的能力**

**Attention based model的核心在于计算注意力分布概率**。


论文设计的分类器的组成部分：

- 文本表示  （采用word embedding，word2vec生成K维向量表示）
- 特征提取 (Attention based LSTM 提取信息，生成特征向量)
- 分类 （分类器选择很多，SVM，逻辑回归啊，NB等，可以采用不同的分类器做实验对比，选择结果最优的那一种方法）

维度灾难？

Theano tensorflow 等DL框架的使用 

Attention based model能够保存丰富的语义信息

单向的模型只能考虑到上文的信息，而双向的才能考虑下文的信息，借鉴bi-LSTM 可以采用双向的Attention based LSTM 能够考虑上下文的语义信息，效果更好。


GRU和LSTM

论文：Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
CSDN博客网址：http://blog.csdn.net/meanme/article/details/48845793

传统的RNN有两个限制：

- Due to the vanishing gradient problem, RNN’s effectiveness is limited when it needs to go back deep in to the context.
- There is no finer control over which part of the context needs to be carried forward and how much of the past needs to be “forgotten”


WILDML 的文章 --- IMPLEMENTING A GRU/LSTM RNN WITH PYTHON AND THEANO的译文：http://blog.csdn.net/u013713117/article/details/53956303

**介绍RNN,CRU,LSTM非常好的一篇博文** --- http://bigdatadigest.baijia.baidu.com/article/546456

梯度下降法的简单理解：http://www.360doc.com/content/12/0529/22/4617781_214618453.shtml

梯度消失或者梯度爆炸的说明：http://blog.csdn.net/qq_29133371/article/details/51867856

softmax函数就是**将一个K维的任意实数向量压缩（映射）成另一个K维的实数向量，其中向量中的每个元素取值都介于（0，1）之间**。

搞清楚 softmax和sigmoid函数的区别？


梯度消失和梯度爆炸？ 原因可以作为论文的内容，也可能会被答辩组老师提问。

> 1、如果说,深度学习的训练过程中遇到了梯度发散，也即前面的层学习正常，后面层的权重基本上不进行更新，导致的是后面的层基本上学习不到东西。那么其实，后面的层的参数基本上就是我们的随机初始化后的参数，相当于对输入的样本同一做了一个映射。只是后面的层有进行学习，那么在这种情况下，DL的效果和浅层的效果其实是类似的。解释:为什么梯度发散会不利于学习，梯度一旦发散就等价于浅层网络的学习，那么使用DL就没有任何优势。



> 2、在深度网络中，不同的层学习的速度差异很大，因为学习速率= 激活值*残差,而残差是从上层的残差加权得到的,也与激活函数有关。尤其是，在网络中后面的层学习的情况很好的时候，先前的层次常常会在训练时停滞不变，基本上学不到东西。这种停滞并不是因为运气不好。而是，有着更加根本的原因是的学习的速度下降了，这些原因和基于梯度的学习技术相关。梯度发散的本质原因还是基于BP的反向传播算法，先天不足。同理也会存在梯度爆炸问题。



> 3、不稳定的梯度问题(梯度收敛和梯度爆炸)： 根本的问题其实并非是梯度消失问题或者激增的梯度问题，而是在前面的层上的梯度是来自后面的层上项的乘积。当存在过多的层次时，就出现了内在本质上的不稳定场景。唯一让所有层都接近相同的学习速度的方式是所有这些项的乘积都能得到一种平衡。是不是可以考虑对于不同的层使用不同的学习率。如果没有某种机制或者更加本质的保证来达成平衡，那网络就很容易不稳定了。简而言之，真实的问题就是神经网络受限于不稳定梯度的问题。所以，如果我们使用标准的基于梯度的学习算法，在网络中的不同层会出现按照不同学习速度学习的情况。一般会发现在 sigmoid网络中前面的层的梯度指数级地消失。所以在这些层上的学习速度就会变得很慢了。这种减速不是偶然现象：也是我们采用的训练的方法决定的。

> 梯度爆炸和梯度消失的原因？国外博文中有提到。



**双向的LSTM（GRU），Bi-LSTM可以不使用均等权值的前向和后向，前向的还是后向的影响大一点。影响大的权重多，影响特征向量多一些，后向的小一点。一般来说后面的内容受前向的内容的影响大一些。** 我的创新？这个权值占比怎么去计算。

双向的RNN和LSTM：**http://blog.csdn.net/jojozhangju/article/details/51982254**

特征值的叠加方式？

RNN的训练方式是BPTT（Back Prropagation Through TIme），werbo等人在90年的时候弄出来的

RNN的介绍：http://m.blog.csdn.net/article/details?id=51334470

**介绍RNN训练的论文** ： SutskEver,Training Recurrent Neural Networks.PhD thesis,Univ.Toronto(2012)

**RNN的进展和改进模型**： http://blog.csdn.net/heyongluoyao8/article/details/48636251  

**梯度消失和梯度爆炸的原因解释**： http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/ （ 这篇文章中介绍了梯度爆炸和消失问题的解释）


### 文本表示的研究论述---介绍几种文本表示的方法


###深度学习模型方面：先介绍RNN，再介绍LSTM/GRU，由于RNN 值考虑了上文的信息影响，所以出现了bi-RNN，借鉴bi-RNN，构想了bi-LSTM/GRU，Attention model的引入（为什么引入这个模型），最后设计双向的基于Attention model的LSTM/GRU的分类算法，做对比实验，分析对比各组结果并分析原因得出结论

####RNN简介：

因为梯度消失和梯度爆炸问题，出现了很多改进，大致有三点：
	
#### 几种文本表示方式的简介

#### bi-RNN

#### LSTM的引入及简介

#### GRU简介

#### bi-LSTM/GRU

#### Attention model(注意力模型)

#### 加入Attention model的双向LSTM/GRU 模型应用与设计的实验系统  文本表示模型采用word embedding，特征提取采用BI-LSTM/GRU,使用单向的LSTM/GRU，双向的LSTM/GRU做对比实验，应用多个数据集（国内的，需要分词--淘宝评论，豆瓣，搜狗实验室；国外的烂番茄网站，亚马逊商城评价，这些数据都需要我们去搜集）

###为什么 传统的机器学习做文本分类不好？

首先

97.5%准确率的深度学习中文分词（字嵌入+Bi-LSTM+CRF）：http://www.17bigdata.com/97-5准确率的深度学习中文分词（字嵌入bi-lstmcrf）.html













