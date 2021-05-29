## 概述
 不管是人工智能，还是数据科学，其核心都是数学原理。机器学习中，如何将世间万物变成数字，以便使用数学方法解决问题尤为重要。图像普遍是以像素值为基础输入模型，而文本又该如何处理成数字呢？
 ## 文本表示
 ### 词袋模型
 最基础的文本表示模型是词袋模型。也就是把每篇文档看作是一袋子词，忽略每个词出现的顺序。每篇文档可以表示成一个长向量，向量中的每个维度代表一个单词，对该维度对应的权重也就反映了这个词在文章中的重要程度。我们常用TF-IDF来计算权重。
 ### N-gram模型
 上述的词袋模型会出现一个教明显的问题。如“natural language processing”这个短语如果拆成三个单词，这三个单词分别出现与三个词一起出现在文章语义上完全不同。所以，将文档切分的方式会丢失许多词之间的关联信息。那么语言模型应运而生。可以将连续出现的n个词（n<=N）组成的词组（N-gram）也作为一个单独的特征放到向量表示中，构成N-gram模型。这里需要补充一点：实际应用中，我们会对单词进行词干抽取处理（Word Stemming），将同一个词的多种词性变化统一起来。
 ### 主题模型
 这也是早期自然语言处理领域的经典模型之一，用于从文本库中发现有代表性的主题（得到每个主题上面词的分布特性）
 ### 词嵌入与深度学习模型
 词嵌入是一类将词向量化的模型的统称，核心思想是将每个词都映射成低维空间（通常维度在50到300左右）上的一个稠密向量（Dense Vector）。其hi，我们也可以把每一维看作一个隐含的主题，只不过不像主题模型中主题那样直观（主题模型有严谨的推导）。
 ## Word2vec
 2013年，谷歌提出Word2vec模型，15年之后该模型被广泛应用，到目前也是最常用的词嵌入模型之一。它是一种浅层的神经网络模型，有两种结构，分别是连续词袋（Continues Bag of Words）和跳字模型(Skip-gram)。当然word2vec之后还有许多主流的词嵌入模型出现，这个我会在后面的博文中介绍。本博问只关注word2vec及其原理。
 ### 原理
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200422170341801.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)CBOW和Skip-gram都是一个浅层的神经网络。输入层每个词都是由独热码表示。一个是利用周围词来预测当前词，另一个是使用当前词来预测周边词。经过浅层神经网络后，输出层也是一个N维向量，最后对输出层向量应用sofmax激活函数，计算每个词的生成概率。
 ### word2vec与LDA的区别与联系
 LDA是利用文档中单词的共现关系来对单词按主题聚类，可以将其理解为对“文档-单词”矩阵进行分解，得到“文档-主题”，“主题-单词”两个概率分布。而word2vec其实是对“上下文-单词”矩阵进行学习，其中上下文由周围的几个单词组成，因此得到的词向量表示更多融合了上下文共现的特征。主题模型和词嵌入两类方法最大的不同其实在于模型本身，主题模型是一种基于概率图模型的生成式模型，其似然函数可以写成若干条件概率连乘的形式，其中包括需要推测的隐含变量（即主题）；而词嵌入模型一般表达为神经网络的形式，似然函数定义在网络的输出之上，需要通过学习网络的权重以得到单词的稠密向量表示。
 ### pytorch实现word2vec
首先导入必要的包
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
dtype = torch.FloatTensor
```
接下来构造一个简单的数据集，并且生成整个数据的词典。

```python
sentences = [ "i like dog", "i like cat", "i like animal",
              "dog is animal", "cat is animal","dog like apple", "cat like fish",
              "dog like milk", "i like apple", "i hate apple",
              "i like movie", "i like book","i like music","cat hate dog", "cat like dog"]

word_sequence = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
```
确定一些基本的参数，确定批次等：

```python
batch_size = 20  
embedding_size = 5 
voc_size = len(word_list)
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)
    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]]) 
        random_labels.append(data[i][1])  
    return random_inputs, random_labels
```
构造一个窗口的skip_gram模型。对每个词来说，每次取出它的前一个词和后一个词作为特征。
```python
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
    for w in context:
        skip_grams.append([target, w])
```
接下来构造模型，这里需要构造浅层神经网络的权重矩阵，以及与权重相乘得到输出的矩阵，使得输出维度单词数：
```python
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Parameter(-2 * torch.rand(voc_size, embedding_size) + 1).type(dtype) 
        self.WT = nn.Parameter(-2 * torch.rand(embedding_size, voc_size) + 1).type(dtype) 
    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = torch.matmul(X, self.W) # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.WT) # output_layer : [batch_size, voc_size]
        return output_layer
```
训练过程中，输入数据维度是[batch_size, voc_size]，经过隐层后维度就变成[batch_size, embedding_size]，再乘完矩阵后最终输出的维度还是[batch_size, voc_size]。
接下来实例化模型，并确定损失，优化器等。

```python
model = Word2Vec()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(5000):
    input_batch, target_batch = random_batch(skip_grams, batch_size)
    input_batch = Variable(torch.Tensor(input_batch))
    target_batch = Variable(torch.LongTensor(target_batch))
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()
```
最后我们画图看一下结果：

```python
for i, label in enumerate(word_list):
    W, WT = model.parameters()
    x,y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
```
结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200422174650348.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTIzNjY1,size_16,color_FFFFFF,t_70)
这里我们仅仅是展示了一下例子，数据较少，所以模型并没有完全收敛，最终的结果也不太好。
### Tensorflow2中的word2vec
 那么在tensorflow2.0中，word2vec该如何实现呢？因为上面在pytorch中已经从原理角度展示了如何构建word2vec，原理角度都差不多。这里主要展示一下在tf2中word2vec该怎么用。

```python
import tensorflow as tf
docs =[ "i like dog", "i like cat", "i like animal",
              "dog is animal", "cat is animal","dog like apple", "cat like fish",
              "dog like milk", "i like apple", "i hate apple",
              "i like movie", "i like book","i like music","cat hate dog", "cat like dog"]
# 只考虑最常见的15个单词
max_words = 15
# 统一的序列化长度
# 截长补短 0填充，当然这里没有超过3的句子，默认是从前面填充0，也可以修改成从后面填充
max_len = 3
# 词嵌入维度
embedding_dim = 3
# 分词
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
# fit_on_texts 获取训练文本的词表
tokenizer.fit_on_texts(docs)
# 字典索引
word_index = tokenizer.word_index
# 序列化
sequences = tokenizer.texts_to_sequences(docs)
# 统一序列长度
data = tf.keras.preprocessing.sequence.pad_sequences(sequences = sequences, maxlen= max_len)
# 添加Embedding层，传入字表长度，句子最大长度和嵌入维度
model = tf.keras.models.Sequential()
embedding_layer = tf.keras.layers.Embedding(input_dim=max_words, output_dim= embedding_dim, input_length=max_len)
model.add(embedding_layer)
model.compile('rmsprop', 'mse')
out = model.predict(data)
# 查看权重
layer = model.get_layer('embedding')
print(layer.get_weights())
```
最终我们输出的out的维度是15*3*3。15是这一波的字表长度，3是每个句子的单词数，最后的3是嵌入的维度。
### Gensim中的word2vec
 其实我们在实际使用过程中，基本都是使用gensim这个包。自己手写的程序不一定有gensim好。gensim非常简单，如果单纯只是为了得到一个嵌入的效果完全可以使用它，下面展示一下，如何使用gensim：


```python
from gensim.models import Word2Vec
import re
docs = [ "i like dog", "i like cat", "i like animal",
              "dog is animal", "cat is animal","dog like apple", "cat like fish",
              "dog like milk", "i like apple", "i hate apple",
              "i like movie", "i like book","i like music","cat hate dog", "cat like dog"]
sentences = []
# 去标点符号
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for doc in docs:
    doc = re.sub(stop, '', doc)
    sentences.append(doc.split())
# size嵌入的维度，window窗口大小，workers训练线程数
# 忽略单词出现频率小于min_count的单词
# sg=1使用Skip-Gram，否则使用CBOW
model = Word2Vec(sentences, size=5, window=1, min_count=1, workers=4, sg=1)
```
这样我们就可以非常方便地训练词向量了。当然gensim功能不止于word2vec，还有TF-IDF，LSA，LDA，包括相似度计算，信息检索等，是nlp入门的神器。

 