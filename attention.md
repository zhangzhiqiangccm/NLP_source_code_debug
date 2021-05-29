## 概述
之前的博客中，笔者都曾提到attention机制。这种考虑全局，关注重点的机制在深度学习中很常见，尤其是self-attention将自然语言处理带到一个新高度。attention增加了深度学习的可解释性，并且应用广泛，在自然语言处理，计算机视觉，推荐系统中到处可见。它克服了循环神经网络解决过长序列时的问题，并且也可以像卷积神经网络那样能够并行计算。本文就列举几个比较经典的attention模型，简述其原理。本文一些内容参考领英团队发表的文章《An Attentive Survey of Attention Models》。
## Attention家族
### 经典attention模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427103621900.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
最开始，attention是在机器翻译中被提出。在Seq2Seq模型中，当我们输入一个序列Xi时，经过一个RNN（或者其变体）进行编码，将编码的结果再通过一个RNN（或者其变体）进行解码（如图a）。这样的Seq2Seq模型会有两个弊端：1.不管输入是多长的序列，始终将其压缩成一个固定长度的向量传入解码器，这一定会丢失一些信息；2.输出的每个token并不能关联到输入中与之相对应的token。而注意力机制就是通过允许解码器访问整个编码的输入序列的隐层，在输入端引入一个权重，解码时优先考虑位置信息，解码输出相对应的token（如图b）。下面我们详细解释一下
首先，利用一个函数来整合编码器隐藏状态h与上一时刻解码器隐藏状态s。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427105835629.png)这里我们用a来代替函数操作，这种函数可以自行确定，但是输入不变。如这样的操作，用两个矩阵参与变换：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427110230460.png)
当然这种直接相乘的方式并不是唯一的，其他研究中也提到了另外的操作。笔者在一篇文章中也找到了文章作者对这类经典操作的总结。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427111000400.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
接着，进行归一化操作，得到权重矩阵值：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427110321601.png)
接下来，将权重与输入状态的隐藏值相乘，得到上下文向量ci，如：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427110648505.png)最后，解码器的隐藏状态就可以加入上下文向量，考虑了整个输入序列及其对应位置关注部分的的信息。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427111148939.png)最终的输出：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427111211380.png)
这就是最经典的attention。
### 局部attention
上面的模型中attention的计算方式是考虑了输入的所有元素的隐层状态。后来又有学者提出如下图这样的结构，并没有考虑输入token序列所有的隐层状态，而是只考虑当前编码位置周边的隐层状态。这种局部attention减少了自己算资源，并且也能够保留相关信息。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427111322414.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
如上图，与全局attention不同的是，在进行权重计算时，使用一个窗口对编码器隐层进行滑动，这个窗口长度为2D+1，即考虑了当前token和它前后D长度的tokens。这里的有一个对其位置Pt，

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020042711342552.png)sigmoid是一个概率中，公式中的S是输入序列的长度（图中有点不一样，叫Tx，请读者注意一下）。这样这个滑动的位置也会随着输入序列长度的不同而变化。此外还要注意一下，注意力矩阵的权重相比之前的全局attention还增加了一个高斯分布乘在后面：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427113658360.png)这也是加入了信息，使得离当前token较远的分配少一些权重，较近的分配近一些的权重。上面就是局部attention原理。

### 指针生成网络
指针生成网络让从源文本中生成单词变得更加容易，甚至可以复制原文本中非正式单词。这就使得模型能够处理那些从未出现过的单词，允许使用更小规模的词汇集。如下图，该模型相比于传统模型提出了Pgen，利用这个概率值，将输出分布矩阵与输入序列的权重矩阵进行关联，从而得到最终生成的注意力分布情况。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427114213860.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
### 记忆网络
记忆网络最早应用于问答系统中。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427114607845.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
如上图，将原始文本嵌入之后，一部分与Question的嵌入进行计算，得到输入后的权重。这个权重再与原始文本结合得到一个中间层的输出，这个输出再次与Question结合得到问题的回答。
### self-attention
self-attention因为Transformer模型而大放异彩，提出Transformer模型的那篇文章《Attention is all you need》的文章题目也是透露出self-attention的强大。不需要循环神经网络，也是能够解决循环神经网络处理的问题，而且性能更优。在后面的博文中，我会详解这篇文章。这里呢，我们先大概感受一下self-attention。下图是Transformer结构。左边部分是编码器（BERT结构），右边部分是解码器（GPT, PGT-2结构）。当然这个是最小单元，真实结构中是多种这样的单元的组合。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427143554676.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
在transformer结构中，self-attention结构尤其关键。将一句话输入后，进过embedding即进入多头注意力机构。嵌入后输入是个三维张量数据。将最后一维（词嵌入这个维度）拆成三份矩阵，并进行矩阵操作：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427144117933.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
拆成Q, K ,V三个矩阵，对这三个矩阵进行操作。这就是注意力矩阵的基本思想。这里暂不展开，后续在其他博文中详细介绍。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427144315953.png)
### co-attention模型
共同注意力模型就是对多个输入序列进行操作，并共同学习它们的注意力权重，以捕获这些输入之间的交互作用。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427144627202.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
上面这张图来源于2016年使用co-attention的例子。在进行视觉问答时，不仅对图像进行了attention操作，也对问句进行相关操作，二者结合得到答案。
### 多层attention叠加
在文本中，也有多个attention叠加操作的情况。从单词到句子，从句子到文档，这样抽象程度不断增加。这样使得文档分类更有可解释性。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427145007169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
### 可解释性
上面提到了可解释性，这里列举本文博客开头提到的综述文章中的例子。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200427145301834.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
第一个是机器翻译，可见对应两种语言的两个词之间关系比较大。第二个图是推荐系统，可见user1和user2对不同风格的图片的注意力也是不同的。第三个是视觉问答。对生成句子中的people来说，图片中的较亮部分与之对应。通过这样的可视化研究，我们完全可以看到attention矩阵很好地解释了这类模型。
## 总结
attention机制不仅无处不在，而且很有必要无处不在。在机器翻译，问答系统，计算机视觉中不可或缺，虽然变体较多，但是核心思想仍在，目前仍然在告诉发展中。如果想对attention有更深的认识，建议去精度相关论文，笔者在这里抛砖引玉，希望能够对读者有所帮助。