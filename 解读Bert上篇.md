## 前言

自然语言处理是计算机科学与人工智能领域的一个重要研究分支，旨在让机器理解和处理人类语言，典型任务如情感分析，机器翻译，智能问答，对话系统，文本生成，信息抽取等。

不管是中文还是英文，都是由字符构成词，再进而组织成短语和句子，乃至段落和篇章。不同于计算机视觉或者语音识别，NLP 模型的输入是离散的（一般为字序列或者词序列）。比如在情感分析中，模型输入为词序列“今天/心情/很好”，模型最终根据“很好”的情感极性以及对整个序列的“语义理解”，来判断出这个评论为正向的评价。而模型理解这句话为正向情感的第一步，就是将词序列转换为计算机能够识别的数字编码方式。

独热编码（one-hot）是最简单的一种编码文本的方式。首先我们可以根据大量的文本数据，建立一个大小为 |V| 的词表，进而将每一个词用不同的 id 对应。对于词 Wi，其对应的 id 为 i，则用独热编码就能被表示成一个长度为|V|的向量，其中第i 维度的值为1，其他维度的值全为0。而对于不在该词表中的词，就统一表示成OOV（out-of-vocabulary）。

虽然独热编码简单有效，但是编码非常稀疏（每个词的向量维度均等于词表大小，且只在一个维度上取值为 1）。并且，不同词之间的表示显然是相互独立的，因此无法表征词与词之间的相对距离远近，即无法编码词级别的语义信息。比如“周到”和“贴心”都可以用于评价服务的好坏，并且语义相似，但是从两者的独热编码中无法看出两个词的距离更近，在独热编码中，“周到”与“糟糕”的距离跟“周到”与“贴心”的距离是一样的。

分布式表示（Distributed Representation）的出现弥补了独热编码的不足。同样是把词表示为向量，分布式表示在词向量的每个维度上都有数值，更有效地利用了空间，并且可以通过“距离度量公式”来计算两个向量间的相似程度，如余弦距离、欧式距离。Google 2013 年提出的 Word2vec，通过改造语言模型，利用相邻词预测特征词（CBOW）或者利用特征词预测相邻词（Skip-Gram）来进行无监督学习，将稀疏的独热表示嵌入到低维空间。得到的 word embedding 能够一定程度上表征词的语义信息，支持一些“线性的语义计算”，比如“king + man - woman = queen”。

<img src="C:\Users\志强张\AppData\Roaming\Typora\typora-user-images\image-20201014210538475.png" alt="image-20201014210538475" style="zoom:50%;" />

￼
分布式表示能将长度为 |V| 的独热编码嵌入到更低维的空间，并且语义相近的词对应的词向量也更为相似。

![image-20201014210641555](C:\Users\志强张\AppData\Roaming\Typora\typora-user-images\image-20201014210641555.png)

以 word2vec 为代表的传统词向量模型主要是学习词汇级表示（lexical level representation），这种表示是上下文无关的。而我们知道，一个词的语义在不同的上下文中可能会发生变化，例如，“我/爱/吃/苹果”和“苹果/手机/打折/了”这两句话中的“苹果”的含义是不同的。

为了得到与上下文相关的词向量表示，ELMo 使用 stacked bi-lstm（两个单向的多层lstm拼接）训练语言模型，综合不同层的 lstm 输出得到最终的 word embedding。这种方式得到的 word embedding 从两个方向考虑了上下文（从左到右和从右到左），并且结合不同层的 lstm 输出考虑了不同粒度的语义表示，相比 word2vec 有更强的表征能力。

但是，由于ELMo只有三层，表达能力依然比较有限，使用方式上也是仍是同 word2vec 相似，即，将ELMo输出的 contextual word embedding 作为 feature 输入到下游模型中（这种方式称为feature-based），对于不同的下游任务还是需要根据任务设计相应的模型（task specific architecture）来实现特定任务。

<img src="C:\Users\志强张\AppData\Roaming\Typora\typora-user-images\image-20201014210725157.png" alt="image-20201014210725157" style="zoom:67%;" />

随后，Open AI GPT 和 BERT 的出现颠覆了 feature-based 的使用方式，提出了预训练-精调的两阶段模型训练范式（Pretrain-then-finetune），不再需要针对下游任务设计特定的模型就可以轻松在各大典型NLP任务上取得SOTA结果。

在预训练阶段，模型参数随机初始化后，在大量语料上进行语言模型为代表的无监督训练，得到通用的预训练模型（拥有更深更强的上下文相关的语义；而后进行的精调则是指根据任务范式（如分类、序列标注等）来改变预训练模型的输出形式，并在任务数据上进行训练。这样，通过预训练模型得到更深更强的结合上下文的语义表示。对于不同的下游任务只需要添加少量任务相关的参数，进行微调就可以实现在不同任务上的高准确性。这种新的方式改变了 NLP 领域根据不同任务定制模型结构的研究模式，也让仅有少量标注数据也能实现高准确性成为可能。预训练（Pretrain）和精调（Fine-tune）的两步式训练框架的成功得益于更大的模型，更大规模的数据和不断强大的计算资源。

以BERT为例，使用BooksCorpus (800M words) 和English Wikipedia (2,500M words) 数据做预训练。单条数据最大长度为512，以256条数据为一个batch，更新100万次，相当于在一个33亿词表上训练40个epochs。训练一个BERT-BASE（12层Transformer）需要在16块TPU上训练4天，训练一个BERT-LARGE（24层Transformer）需要在64块TPU上训练4天。

<img src="C:\Users\志强张\AppData\Roaming\Typora\typora-user-images\image-20201014210825401.png" alt="image-20201014210825401" style="zoom:67%;" />

GPT 和 BERT 都是使用多层 Transformer 编码器作为主干结构，不同的是 BERT 将模型尺寸增大到24层，同时用了比GPT多约4倍的语料进行预训练。重要的是，BERT 还提出了新的预训练任务 MLM（Masked Language Model），在训练时可以同时编码上下文信息，优于传统的单向语言模型，使得预训练后的 BERT 在 11 个自然语言理解任务（由8个自然语言理解任务组成的 GLUE 标准评测集和问答任务 SQuAD v1.1、SQuAD v2.0和SWAG）上脱颖而出，超越 SOTA，开启了 NLP 大规模语训练模型的新时代。

##  Transformer

Transformer 是 2017 年由 Google 提出的结构，用于机器翻译的编码和解码。后来因为其强大的编码能力和可并行性被广泛作为编码器使用。每个 Transformer 模块可以被称为模型中的一层（Layer），目前常用的模型会由多层 transformer 堆叠组成，从 low-level 到 high-level 逐渐提取结合上下文的语义表示。

Transformer 接收的输入为一个长度为 n 的词序列的向量 $x = (x_1,…,x_n)$，$ x\in R^{n \times d}$ 。其中d为向量的维数。输出为编码后的向量 $z = (z_1, …, z_n)$

  )，$z \in R^{n \times d}$  。其结构如下图所示，transformer 由两个子层（Sub-layers）构成，第一个 sub-layer 为多头的自注意力机制（Multi-head self-attention mechanism），第二个 sub-layer 为简单的前馈神经网络（Feed Forward Network）。为了更好地训练模型，加速收敛，还会在每个 sub-layer 后进行残差连接（Residual connection）和层归一化（Layer normalisation）。上一层 layer 的输出作为下一层 layer 的输入，不断堆叠最终得到结合上下文的 high-level 的语义表示。

![image-20201014210953530](C:\Users\志强张\AppData\Roaming\Typora\typora-user-images\image-20201014210953530.png)

* Multi-Head Attention

Multi-Head Attention 是 transformer 的核心部分，它源自于注意力机制（Attention）。Attention最早出现在机器翻译任务中，随后发展了各种变体并被广泛应用于 NLP 的各个领域。它的做法是赋予各个词不同的权重，最后通过线性加权获得新的词、句表示。这样其实是对句子进行了隐式的结构化，把重要的部分提取出来，剔除噪声，更好地获得语义信息。 BERT中使用的MHA主要基于 Scaled Dot-Product Attention。

Scaled Dot-Product Attention 的输入为 query，key 和 value，分别记为 $Q \in R^{n \times d_{k}}$ ; $K \in R^{n \times d_{k}}$ 和 $V \in R^{n \times d_{v}}$V∈R ，其中 Q 和 K 的维数一致。Attention 权重计算的公式如下：
$$
Attention(Q, K, V) = softmax(\frac{QK^{T}} {\sqrt(d_k)})V
$$
Q 和 K 计算内积，得到一个 $n \times n$  维度的矩阵。这里除 $d_k$ 是因为计算 $QK^T$ 时涉及到乘积的累加，对于较大的 $d_k$ ，点积结果大幅增大，将 Softmax 函数推向具有极小梯度的区域（为了阐明点积变大的原因，假设 q 和 k 是独立的随机变量，平均值为 0，方差 1，这样他们的点积为 $q\cdot k=\sum_{i=1}^{d_k}q_i\cdot k_i$   ，均值为 0 方差为 $d_k$）。为了抵消这种影响，论文中用 $\frac{1}{\sqrt{d_k}}$  来缩放点积，消除维度的影响。

之后，对矩阵的每一行使用 softmax 归一得到对应的权重，对 V 进行加权求和，作为 Attention 模块的最终输出，如下图左所示：

![image-20201014211035747](C:\Users\志强张\AppData\Roaming\Typora\typora-user-images\image-20201014211035747.png)

多头注意力（Multi-Head Attention，上图右）是对 Q、K、V 进行多次线性映射，综合多次 Attention 的结果作为最终的输出，其目的是通过不同的注意力头抽取不同的特征，就像 CNN 中会有多个卷积核一样。

对于 $head_i$  ，Q、K、V 对应的投影矩阵为 ${W_i}^Q \in R^{d_{model \times d_k}}$、${W_i}^K \in R^{d_{model \times d_k}}$、$${W_i}^V \in R^{d_{model \times d_v}}$$，将其对应的投影作为输入计算 Attention。最后将多头的 Attention 输出拼接起来，通过全连接将最终输出投影到维数为 $d_{model}$ 的空间中。

$$ head_i = Attention(Q{W_i}^Q, K{W_i}^K, V{W_i}^V)， i = 1,…,h$$ 

$$MultiHead(Q, K, V) = Concat(head_1,…,head_h)W^O$$

记h为多头个数，$d_k = d_v = d_{model} / h $. 由于随着头数的增加 $d_k$ 和 $d_v$ 维数会成比例减少，所以 multi-head attention 的计算复杂度和单头 attention 相似。

* Feed Forward Network（FFN）
  Transformer 中的 Feed Foward 结构实际上是一个两层的全连接，加上 ReLU 激活函数[10]。上文的多头注意力本质是线性加权，所以 FFN 的主要目的是给 transformer 加入非线性模块，提升模型的拟合能力。

记输入为 xx,则经过 FFN 后的输出为：

$$FFN(x) = max(0, xW_1+b_1)W_2 + b_2$$

* Add & LayerNorm
  由于 transformer 在实际应用中是多层堆叠的，而每一层 transformer 又由两个 sub-layer 构成，为了让每一阶段的输入分布更加稳定，保证梯度的有效传递，在 sub-layer 后增加了残差连接[11]和 Layernorm[12] 的机制。

记 sublayer 的输入为 xx，其最终传递给下一阶段的输出为：

$$ y = LayerNorm(x + Sublayer(x))$$

其中 LayerNorm 是在 feature 维度的归一化。

* 与 CNN 和 RNN 的区别
  卷积神经网络（CNN）[13]和循环神经网络（RNN）[14]及其变体LSTM、GRU 等，也是自然语言处理中常用的文本编码器。循环神经网络逐个接收不同时刻的词向量 $x_t$ ,  结合历史信息$h_{t-1}$ 得到更新的 $h_t$ ，其递归的结构天然适合处理文本这样的序列型数据。

![image-20201014211357474](C:\Users\志强张\AppData\Roaming\Typora\typora-user-images\image-20201014211357474.png)

卷积神经网络一开始用于处理图片数据（$W \times H$矩阵型数据），由于我们将长度为 n 的文本序列表示为向量后也是矩阵的形式（$N \times dimension$)。所以可以使用和 $k \times dimension$ 大小的 kernel 做卷积，获取相邻词之间的信息，类似传统的 n-gram 模型。最后通过 max-over-time-pooling 得到最终的表示。

![image-20201014211440233](C:\Users\志强张\AppData\Roaming\Typora\typora-user-images\image-20201014211440233.png)

由于 RNN 的递归结构使得其无法实现并行计算，而 CNN 和 tranformer 都可以通过并行提高计算速率。除此之外，长距离依赖问题（long-term dependency），即模型无法捕获序列中较远的依赖关系，是自然语言处理的一个难点，我们希望模型能够通过最短得路径捕获文本中任意位置间得依赖关系。从这一维度，RNN 获取长度为 $n$ 文本序列中任意位置依赖关系的最远路径为 o(n)，CNN 的最远路径为 $o(log_k(n))$，而 transfomer 仅需要 o(1)。

所以，对比 RNN 和 CNN，transformer 能够更容易地学习文本中的长依赖关系，并且可以完全实现并行计算，加速模型的计算效率，因此获得了更好的效果和算法人员的青睐。

为了 BERT 模型能够更加灵活的处理多种 NLP 任务，需要模型可以接收单个句子或者句子对作为输入。正如前文介绍的 transformer 接收词序列的向量输入$ x=(x_1,…,x_n)$，那么在 BERT 中是如何将单个句子或者句对转换为向量呢？

如下图所示，首先得到句子的字／词序列。并在句子的结尾处添加特殊的 [SEP] 用于标识一个句子的结束。在序列的开头添加 [CLS]用于学习整个句子或者句对的语义表示，更多关于[CLS]的作用会在下文 pre-training 和 fine-tunging 部分中介绍。

![image-20201014211549166](C:\Users\志强张\AppData\Roaming\Typora\typora-user-images\image-20201014211549166.png)

* Token embedding
  对于字／词序列，会通过查表得到相应的 token embedding。值得注意的是，特殊的 [CLS]和 [SEP] 也在词表中，有自己的 embedding，并且所有的 token embedding 都是可以学习的。

* Positional embedding
  由于 BERT 使用 transformer 作为编码器，为了保留文本的位置信息，所以需要额外加入 positional embedding。BERT 模型使用的是 Learned Positional Embedding 编码绝对位置。直接对不同的位置随机初始化一个 postion embedding，加到 token embedding 上输入模型，作为参数进行训练。
  更多 positional embedding 的解析可以参考这里。

* Segment embedding
  因为 BERT 的输入可以是句对的形式，虽然在输入 token 上已经使用 [SEP] 区分句尾了，但是由于 transformer中的核心机制 attention 数会忽略前后顺序关系的，所以我们还需要添加 segment embedding，用于区分每个 token 所属的 segment。$E_A$ 标识 token 来自 sentence A，$E_B$ 标识 token 来自 sentence B。总的来说，BERT 接收文本输入，通过切词，对于每一个 token，将其对应的 token embedding、segment embedding 和 position embedding 求和，作为该 token 的表示向量输入到模型中。

  ### 输出

  transformer接收长度为n的词序列的向量表示 x = （$x_1$, $x_2$ ... $x_n$），输出为编码后的向量 z = ($z_1$, $z_2$ ....$z_n$)，每一个token $x_i$ 结合上下文被表示成了向量$z_i$ 。每一个sub-words和特殊的[CLS] 的向量表示会在不同任务中用于训练或预测。

  ## 语言模型

  

  语言模型（Language Model，LM）的目标为基于上文预测后续出现的词，是一种常见的自监督学习任务。但是由于语言模型是单向的，只能从左到右或者从右到左，而简单的两方向拼接并不能真正学习到双向的同时基于上文和下文的语义表示，所以BERT提出了Masked LM。通过对序列随机mask，将真实的 token 代替为 [MASK]，通过该 token 的上下文向量表示预测真实的 token。

以从左到右的语言模型为例，根据前文${x_{<t}}$  预测当前词 $x_t$ ,训练目标是最大化似然函数

$$max_{\theta} logp_{\theta}(x) = \sum_{t=1}^{T} logp_{\theta}(x_t|x_{<t})$$

而MLM则是基于随机mask处理的输入序列$\hat{x}$ 预测被mask的词。

$$max_{\theta} logp_{\theta}(\bar{x}|\hat{x}) \approx \sum_{t=1}^{T} m_t logp_{\theta}(x_t|\hat{x})$$

其中 $m_t=1$表示 $x_t$ 被 mask。

具体的训练数据构造方法是，对每个数据句子中 15% 的概率随机抽取 token，以 80% 的概率替换为 [MASK]，10% 的概率替换为其他 token，10% 的概率保持不变。之所以不全替换为 [MASK]，是为了减少 pre-training 和 fine-tuning 阶段输入分布的不一致，因为 fine-tuning 阶段的输入是没有 [MASK] 的。

### Next Sentence Prediction（NSP）

Masked LM 任务可以学习双向的上下文关系。但是很多 NLP 的任务除了理解单个句子以外，还需要理解两个句子的关系，比如问答和推理任务。为了让模型能够理解学习句对之间的关系，所以提出了第二个预训练任务NSP。NSP取[CLS]的最终输出进行二分类，判断当前输入的两句话是否连贯，类似句子粒度的语言模型，让模型学习输入句子对之间的关系。

具体的训练数据构造方法是，对于 sentence A，sentence B 以 50% 的概率为 sentence A 的下一句，以 50% 的概率进行随机负采样，即从语料库中随机抽取一个句子。

## Fine-tuning

基于上面两个任务的预训练，具体 sub-words 的向量表示综合了双向的上下文信息，[CLS] 的向量表示在结合整个句子信息的基础上还编码了句对之间的关系。在 fine-tuning 阶段，针对不同的下游任务，只需要在 pre-trained 模型基础上增加任务相关的输出处理，利用领域内的标注数据进行再训练，让模型更好地拟合任务数据的分布。

下图展示了几种常用任务的输入输出:

![image-20201014211757520](C:\Users\志强张\AppData\Roaming\Typora\typora-user-images\image-20201014211757520.png)

（a）对于句对的分类任务，如 MNLI、QQP、QNLI、STS-B、MRPC、RTE 和 SWAG，输入 sentence 1 和 sentence 2，使用句首 [CLS] 的语义表示向量 C，接一个 softmax 分类层进行分类。

（b）对于单个句子的分类任务，如 SST-2 和 CoLA，输入 sentence 1，sentence 2 为空，同样根据最终 [CLS] 的表示向量 C 进行预测。

（c）对于问答任务 SQuAD v1.1，基于篇章 Paragraph 回答问题 Question，预测答案的起始位置（start/end span）。sentence 1 为 Question，sentense 2 为参考的 Pragraph。稍微特殊的是，在 fine-tuning 阶段增加了一个 S 和 E 向量，对于 $t_i$ 为答案开始的概率为

$P_i = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}$ 


类似地，对于 $t_i$ 为答案结束的概率为

$P_i = \frac{e^{E \cdot T_i}}{\sum_j e^{E \cdot T_j}}$


fine-tuning 阶段，训练的目标函数为最大化真实答案起始位置的概率和。最终预测时，预测的答案 span 为

$argmax_{i \leq j} {S \cdot T_i + E \cdot T_j}$ 

（d）对于单个句子的标注任务，如 CoNLL-2003 NER，实际上是对每一个 sub-words 做多分类，最终得到一个序列的标注结果。

## 精调实战

NLP 作为人类和计算机的沟通桥梁，目前已经有很多重要的落地场景，包括但不限于搜索引擎，推荐系统、对话机器人等。这些常见场景中都有包含NLP任务，比如对话机器人中的意图识别就是典型的分类任务，槽位填充是一个序列标注问题；搜索引擎中的用户query和候选文档的匹配度计算就是一个NLP中的匹配任务。根据输入输出的形式，NLP任务可以汇总为以下四类任务：

文本分类：输入文本，输出类别，主要有主题分类、情感分类等。
句间关系：输入多条文本，输出关系类别，主要有文本匹配、蕴含识别等自然语言推理（Natural Language Inference，NLI）任务。
序列标注：输入文本，对每个字/词输出类别，主要有分词、词性标注和命名实体识别（Named Entity Recognition，NER）等任务。
文本生成：输入是原始的数据、文本或者图像，输出一段可读的自然语言文本，主要有机器翻译、摘要生成等任务。
其中，文本分类、句间关系、序列标注三个任务统称为文本理解（Natural Language Understanding，NLU）任务。因为它们的目标主要是理解输入文本的语义信息，而文本生成任务（Natural Language Generation，NLG）则不仅要理解文本，还需要生成一段可读的文本作为输出。

NLU 任务是所有场景中的必备模块，所以本节课我们主要介绍如何用 BERT 进行 NLU 任务的精调。

考虑到 Tensorflow 有一定的学习门槛，且部分不同版本的 API 变化较大，本教程主要使用易上手的 PyTorch 框架进行教学。具体的模型结构主要使用 Hugging Face 开源的 transformers 库，相较于其他开源实现更加全面且清晰。如果你之前没有接触过也没关系，在下文中，我们将对于如何使用 transformers 库进行详细的介绍。

运行环境
首先，我们利用 conda 创建一个Python的虚拟环境，并安装 PyTorch。conda的安装和基本使用可以参考教程。

conda 命令创建虚拟环境
`conda create -n torch python=3.6`
安装 PyTorch，根据 PyTorch 官方指导进行安装。我们利用 conda 包工具安装，默认源的下载速度非常慢，改成清华源下载。
`conda install pytorch torchvision cudatoolkit=9.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/`
安装 Hugging Face 的 transformers 库，transformers 库对 BERT、XLNet 等 10 个预训练模型进行了接口封装，方便调用。
`pip install transformers`
下载我们提供的 BERT 的代码文件和预训练模型文件，并将 bert_config.json 改为 config.json。（如果从亚马逊云下载预训练模型会非常慢，所以我们使用开源项目 Chinese-BERT-wwm 中提供的 BERT-wwm 中文模型）
熟悉 transformers 库使用
恭喜你跨过了环境安装这第一道难关！准备就绪后，我们就可以动手搞起代码啦～先来了解如何使用 Hugging Face transformers 库中的BERT模型，以及熟悉我们的代码结构，了解代码的各个文件和模块的作用。

transformers 库提供了 10 种预训练模型的使用接口，包括 BERT，XLNet，RoBERTa，DistillBERT 等，每种模型都有模型结构和分词器两个主类，模型结构类决定了使用哪种基础模型作为编码器，和分词器已经对应好了。BERT 对应的是 BertModel 和 BertTokenizer 这两个类，通过以下三行代码就能完成 BERT 的模型加载。 pretrained_weights 参数代表使用 BERT 的哪个模型，这里和上文中下载的模型文件保持一致，加载 chinese_wwm_pytorch 模型。

```python
pretrained_weights = 'chinese_wwm_pytorch'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model = BertForTokenClassification.from_pretrained(pretrained_weights)
```


Google 为 BERT 提供了 5 种预训练模型，分别是 bert-base-uncased，bert-base-cased，bert-large-uncased，bert-base-multilingual，bert-base-chinese。pretrained_weights参数可以是以上 5 种官方提供的模型名称，也可以是本地模型的存储路径，第一次运行代码时会先下载对应的模型文件。

模型加载是模型初始化的过程，加载完成后，通过 tokenizer.encode() 函数得到数据的格式化输入，之后转换成tensor送入模型，就可以得到 BERT 编码的句子输出向量了。tokenizer 类的功能主要是对文本预处理，预处理包括对句子分词，对句子首尾添加特殊字符[CLS]、[SEP]，补齐 (pad) 或者截断 (truncate) 成固定长度的句子等。

```python
input_ids = torch.tensor([tokenizer.encode("夕小瑶的卖萌屋", add_special_tokens=True)]) 
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]
    print(last_hidden_states)
```


通常我们在 task-specific 数据集上微调 BERT 就能得到不错的效果，transformers 为了方便使用者在下游任务中快速地使用 BERT ，还提供了多种针对特定任务封装的模型接口，包括

```python
BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM,\
BertForNextSentencePrediction, BertForSequenceClassification,\ 
BertForTokenClassification, BertForQuestionAnswering]
```


借助这些专门为某一类任务打造的接口，一方面，我们可以非常快速地上手一个新任务，另一方面，可以非常方便地修改模型顶层（输出层）的结构，从而找到在自己项目数据集上的最优的模型结构配置。比如 BertForSequenceClassification 是专门为分类任务设计的一个类，经 12层/24 层 的 Transformer 编码得到隐态表示之后，连接了一层线性分类层。

###　代码结构

我们提供的 pytorch 版 BERT 代码分为 6 个部分：main.py，model.py，prepro.py，parser_args.py，src 文件夹以及 shell 脚本文件。

main.py：主函数入口文件，包括模型的训练和验证流程，模型加载、模型保存、模型输出，控制日志输出等。

model.py：模型定义模块，即定义模型计算图，负责定义模型从输入到输出（loss, logits）的处理流程。因为本次介绍的三个任务在 transformers 中已经有现成的接口，所以暂时不涉及这个模块的处理。

prepro.py：数据处理模块，负责将文本序列的句子处理成模型需要的 tensor，包括不同任务的 processor 预处理函数。

parser_args.py：统一设置超参数的文件。

src ：transfomers 库的源代码文件，包括已经封装好的各种模型类，配置类，我们这一节课程不涉及这里的代码。

脚本文件：train.sh，eval.sh。训练和测试的脚本文件，可一键运行。

模型的训练：

```python
# task_output_name是本次训练任务的输出路径
sh train.sh <task_output_name>
```


模型的评估：

```python
# task_output_name是本次训练任务的输出路径
sh eval.sh <task_output_name>
```

接下来，我们将使用 BertForSequenceClassification 完成分类和匹配任务的精调，使用 BertForTokenClassification 完成序列标注任务的精调。

本次实战教程以中文任务为例讲述。首先，下载三个开源中文数据集并处理成指定的数据存储格式。

分类数据集：今日头条中文新闻数据集 TNEWS
下载链接：https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip

匹配数据集：蚂蚁金融语义相似度数据集 AFQMC
下载链接：https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip

序列标注数据集：中文微博命名实体识别数据集
下载链接：https://github.com/quincyliang/nlp-dataset/tree/master/ner-data/weibo

下载的原始数据集中已经包含了训练集和测试集，我们只需要把数据集处理成 BERT 要求的格式就可以了，并保存为 train.tsv 和 test.tsv 文件。在下一小节中，我将一步步带着大家完成该数据的处理。

文本分类的存储格式如下：”\t” (Tab 制表符) 作为句子和标签的分隔符。

文本匹配的存储格式如下：”\t” (Tab 制表符) 作为元素之间的分隔符。

序列标注存储格式如下：”\t” (Tab 制表符) 作为句子和标签的分隔符， “ “ （空格）作为每个token label 的分隔符。

这里我们给出数据预处理的部分核心代码，位于data_preprocess.py文件中。分类和匹配数据的处理步骤如下：

```python
import json
labels = {}
train_fw = open("train.tsv", "w")
with open("train.json", "r") as f:
    for line in f:                                        
        tmp = json.loads(line.strip())
        if tmp["label"] not in labels:
            labels[tmp["label"]] = tmp["label_desc"]
        # 分类
        train_fw.write(tmp["sentence"] +"\t"+tmp["label"]+"\n")
        # 匹配
        train_fw.write(tmp["sentence1"] +"\t"+tmp["sentence2"]+"\t"+ tmp["label"]+"\n")
with open("label.txt", "w") as label_fw:
    for l in labels.keys():
        label_fw.write(l+"\n")
```


微博命名实体识别数据集的处理步骤。

```python
import codecs
import csv 
import pandas as pd
ner_labels_dict = []
data = []
sentence = []
label_list = []
```

```python
# load downloaded data
with codecs.open("weiboNER_2nd_conll.train", "r", "utf-8") as f:
    for line in f:                                                                                                                
        if line.strip() != "": 
            tmp= line.strip().split("\t")
            if tmp[-1] not in ner_labels_dict:
                ner_labels_dict.append(tmp[-1])
            sentence.append(tmp[0][0])
            label_list.append(tmp[-1])
        elif line == "\n":
            data.append((sentence, label_list))
            sentence = []
            label_list = []
        else:
            continue
```


```python
# save as train.tsv
with open("train.tsv", "w") as f:
    for ex in data:
        s = "".join(ex[0])
        l = " ".join(ex[1])
        f.write(s+"\t"+l+"\n")
# save as label.txt
with open("label.txt", "w") as f:
    for k in ner_labels_dict:
        f.write(k+"\n")
```


### 数据格式化处理  

数据在送入模型计算编码之前，首先需要经过数据处理模块，转换成 BERT 定义的输入数据的格式。我们先来回顾一下，BERT 是怎么处理文本序列的。BERT 对中文数据都是按字处理的，并且提供了一份 2w+ 的中文词典，用于把文本序列转换成数字 id 序列。transformers 中提供的 tokenizer 类可以很方便地完成这些步骤。

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("chinese_wwm_pytorch")
tokens = tokenizer.tokenize("欢迎来到本次课程")
input_ids = tokenizer.convert_tokens_to_ids(tokens)
```

除了分词和转换成数字下标以外，BERT 还要求：

* 添加特殊字符 [CLS] 和 [SEP]
* 追加 [PAD] 或者截断成固定长度的句子
* 定义 attention_mask 区分哪些是真正的字符
* 定义 token_type_ids 区分不同的句子

```python
# prepro.py
tokens = tokenizer.tokenize(text, add_special_tokens=True)
tokens = ["[CLS]"]+tokens[:max_length-2]+["[SEP]"]
text_len = len(tokens)
input_ids = tokenizer.convert_tokens_to_ids(tokens+["[PAD]"]*(max_length-text_len))
attention_mask = [1]*text_len+[0]*(max_length-text_len)
token_type_ids = [0]*max_length
```


之后，还要把序列化的输入转换成模型可以处理的 tensor，然后利用 pytorch 中已经封装好的数据加载器，打乱和加载训练集。

```python
# prepro.py
# convert ids to tensor
all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)                         
all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)
# main.py
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset 
# 数据迭代器
dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1,  all_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
```

 ### 加载预训练模型

数据准备就绪后，我们首先加载我们的预训练模型，这里我们采用从指定路径加载模型配置和模型权重参数，这一步是构建计算图和初始化的过程，并设置好优化器 optimizer 和学习率 scheduler。

```python
# main.py
# 加载配置项，num_labels是分类的类别数量
config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )  
model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )  

# 优化器和学习率设置
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay,},
{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
"weight_decay": 0.0},]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
)
```

### 训练

训练的超参数在 parser_args 中根据自己的需求自定义，比如训练轮数，num_steps，batch_size 大小等。pytorch 中梯度会自动累加，所以每次参数更新迭代结束后要手动对梯度清零。

```python
# main.py
global_step = 0                                   
tr_loss, logging_loss = 0.0, 0.0
model.zero_grad()
train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch")
set_seed(args) 
# train
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        model.train()
        batch = tuple(t.to(args.device) for t in batch)
        # inputs
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": batch[3]}
        outputs = model(**inputs)
        loss = outputs[0]  
        # BP
        loss.backward()
        tr_loss += loss.item()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        global_step += 1
```


训练过程中可以打印 loss 值，查看训练过程的 loss 变化趋势，也可以边训练边验证模型的学习效果，保存中间训练过程的模型模型参数和和结果，这些炼丹技巧在接下来的课程当中，我们会展示给大家，这里不做赘述了。

### 验证测试

模型验证的过程基本和训练的流程是一致的，需要注意的一点是我们需要从模型输出中拿到模型预测的结果，并拷贝到 cpu 中，计算准确率等评价指标。

```python
# main.py
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
eval_loss = 0.0
nb_eval_steps = 0
preds = None
out_label_ids = None
#eval
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": batch[3]}
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        # copy "logits" to cpu 
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
# average
eval_loss = eval_loss / nb_eval_steps
if args.output_mode == "classification":
    preds = np.argmax(preds, axis=1)
    
#differant metrics based on different task
result = compute_metrics(preds, out_label_ids)
```


以上介绍了精调 BERT 的通用流程，对于具体的任务有不同的处理，下面我们将详细介绍分类、匹配、命名实体识别（Named Entity Recognition， NER）三个任务的具体实现细节和区别～

### 分类任务

首先定义任务的 processor 类，根据上文中分类任务数据的存储格式，读取数据集和标签。单句分类只有 text_a 和 label。

```python
# prepro.py
class TNewsProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples
        def get_test_examples(self, data_dir, file_name):
            ...
        def get_labels(self, data_dir):
            ...
# 然后对读取的 text_a 进行格式化的特征处理
# single sentence
def convert_text_to_ids(text):
        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        tokens = ["[CLS]"]+tokens[:max_length-2]+["[SEP]"]
        text_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens+["[PAD]"]*(max_length-text_len))
        attention_mask = [1]*text_len+[0]*(max_length-text_len)
        token_type_ids = [0]*max_length
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        return tokens, input_ids, attention_mask, token_type_ids

```

其次，加载分类的模型文件，和上文 3.3 节一样。这里三种任务的配置和分词器都采用 BERT 的 BertConfig 和 BertTokenizer，只有模型类型不同，文本匹配其实是一个简单的二分类任务，所以和分类任务一样使用 BertForSequenceClassification 类，NER 使用词级分类模型 BertForTokenClassification。

```python
# main.py
MODEL_CLASSES = { 
    "classification": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "matching": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "ner": (BertConfig, BertForTokenClassification, BertTokenizer),
}
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(...)
tokenizer = tokenizer_class.from_pretrained(...)
model = model_class.from_pretrained(...)
```


最后，训练模型以及验证效果。这里模型的输出需要 src/transformers/modeling_bert.py 中查看BertForSequenceClassification 类的 outputs 格式，第一位置是 loss，排在第二位的是 logits，是模型没经过 softmax 的预测结果，用此处的 logits 计算模型预测的准确率。

```python
# Training
if args.do_train:
    train_dataset, train_examples = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
if args.do_eval:
    result = evaluate(args, model, tokenizer, prefix=prefix)
```

## 匹配任务

匹配任务和上述分类任务唯一的不同是 processor 类和数据特征处理部分，需要读取每一个 pair 对的 text_a，text_b，label 是 0 和 1。BERT中的句子对是合并成一个句子编码的，只不过两个句子之间用分隔符号 [SEP] 隔开，并且用不同 token_type_id 区分，0 表示是 text_a 的内容，1 表示是 text_b 的内容。
这里有个比较细节和隐蔽的处理步骤，当 text_a + text_b 的长度超过最大句子长度时，如何截断处理，我们采用和tf版相同的截断技巧，优先截断较长的句子。

```python
# prepro.py
class AFQMCProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
        def get_test_examples(self, data_dir, file_name):
            ...
        def get_labels(self, data_dir):
            ...
# sentence pair
def convert_text_to_ids_for_matching(text_a, text_b):
        tokens_a = tokenizer.tokenize(text_a)  
        tokens_b = tokenizer.tokenize(text_b) 
        # Truncates a sequence pair in place to the maximum length，same with tf.
        if len(tokens_a) + len(tokens_b) > (max_length-3):
            _truncate_seq_pair(tokens_a, tokens_b, max_length-3) 
        tokens =  ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        text_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens+["[PAD]"]*(max_length-text_len))
        attention_mask = [1]*text_len+[0]*(max_length-text_len)
        token_type_ids = [0]*(len(tokens_a) + 2) + [1]*(len(tokens_b)+1)+[0]*(max_length-text_len)
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        return tokens, input_ids, attention_mask, token_type_ids
```


其他步骤和分类是一样的。详见上述过程。

## 命名实体识别任务

命名实体识别（NER）任务是典型的序列标注任务，比较常用的标签体系是采用 BIO 标注，对人名（PER）、地名（LOC）、机构名（GOV）等实体打 tag。因为句子里的每一个 token 都有标签，所以 NER 的每一个 example 的 label 都是一个 list，模型的 label 输入的 shape 是 [batch_size, seq_len, num_labels]，在 processor 读取和特征格式化处理的时候和前两者不同。

```python
# prepro.py
class WeiboNerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            label = line[1].split(" ")
            examples.append(NERInputExample(guid=guid, text_a=text_a,  label=label))
        return examples
        def get_test_examples(self, data_dir, file_name):
            ...
        def get_labels(self, data_dir):
            ...
```


NER 任务的标签处理环节，为每一个 token 带上 label。

```python
# prepro.py
if output_mode == "ner":
            label_id = [label_map["O"]]
            for j in range(len(tokens1)-2):
                label_id.append(label_map[examples[i].label[j]])
            label_id.append(label_map["O"])
            if len(label_id) < max_length:
                label_id = label_id +[label_map["O"]]*(max_length-len(label_id))
```


NER 使用 Hugging Face 的 transformers 库中的词级分类模型BertForTokenClassification，本质是一个单句多 label 分类任务，模型输出 outputs 第一位置依旧是 loss 值，第二位置是 scores，shape=[batch_size, seq_len, num_labels]，对句子中每一个 position 的 token 都给出了预测得分，得分最高的 idx 是模型当前位置 token 的 predicted_label，根据此结果我们可以计算准确率。

```python
# main.py
# metric(accuracy)
if args.output_mode == "classification":
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)
else:
    # ner: logits=scores .shape=[examp_nums, seq_len, num_labels]
    preds = np.argmax(preds, axis=2)
    result = ner_compute_metrics(preds, out_label_ids)
```



在上一节中，我们已经学会了如何利用 BERT 进行分类、匹配、序列标注三个基本 NLP 任务的实战，大家也肯定为 BERT 优秀的效果惊叹不已。在这一节中，我们将详细讲一下如何对 BERT 进行精调可以使我们的效果更上一层楼。

## 精调基础

本章分为 3 节来具体讲一下在 BERT 不同使用阶段的精调技巧， 包括数据预处理、模型运行和输出三个阶段。

为简化模型细节，方便大家调参，本节课学习的代码将上节课 BERT 代码中的分类问题单独抽取出来，供大家学习。代码结构如下：

run.py： 模型入口，供大家一键运行。
utils.py： 功能函数，主要包含数据准备与预处理。
train_eval.py： 训练、验证、测试主函数。
models/bert.py： BERT模型结构，包含对模型超参数的封装。
bert_pretrain： 预训练好的 BERT 权重，下载地址见文件中的 README。

### 预处理阶段

神经网络的输入一般都是矩阵，这就要求输入的每一个句子需要保证固定长度。然而，现实中句子长度各异，无法保证全部相等。这时候就需要做到“取长补短”——长句子只能取其中一部分，短句子则要进行补齐。

这样，就会出现一个典型的问题——句子的长度到底应该是多少呢？是不是越长越好呢？事实上，随着文本长度在一定范围内的增加，最终的结果一般是要更好。但当超过某个阈值，某些 NLP 任务（如文本分类、信息抽取）的效果并不会再显著增加。而且所需显存容量会随之线性增加， 运行时间也接近线性增长。所以受机器性能的影响，并不能无限制的增加文本长度，需要在效率和精度上做一个权衡。

BERT 预设的最大文本长度为 512，这就要求了输出句子的长度不能超过这个长度。值得注意的是，对于中文句子，由于 BERT 是基于字的处理方式，所以只要保证每个句子 len(sent)<= 512 即可。但是对于英文句子则不能直接这么表示，因为 BERT 采用了 WordPiece
模型生成英文 token。最终生成的句子长度，往往要大于原文本。举个例子：

```python
tokenizer.tokenize('我爱中国！')  #['我', '爱', '中', '国', '!']
tokenizer.tokenize('Massachusetts ASTON MAGNA Great Barrington')  #['Massachusetts', 'AS', '##TO', '##N', 'MA', '##G', '##NA', 'Great', 'Barr', '##ington']
```


如果超过这个 512 这个长度，建议根据任务的不同采用截断或分批读取的方式来读入。

### 超长文本处理

#### 截断 trick

由于 BERT 支持最大长度为 512，那么如何截取文本也成为一个很关键的问题。

一个合理的做法是先做 case 分析，人工判断对目标结果影响大的模式（如答案，情感词，主实体等）一般是怎样分布的。如果发现绝大部分模式位于开头，那就直接截取 head 部分即可；如果发现大部分聚集在尾巴，那就直接截取 tail 部分；如果是比较平均分布，根本没确定规律，那就果断滑动窗口，或是根据 How to Fine-Tune BERT for Text Classification 中采用的三种方法依次进行尝试:

* head-only： 保存前 510 个 token （留两个位置给 [CLS] 和 [SEP] ）
* tail-only： 保存最后 510 个 token
* head + tail ： 选择前 128 个 token 和最后 382 个 token

作者在 IMDB 和 Sogou 数据集上测试，发现 head + tail 效果最好。

#### 相对位置编码

另外，处理长文本的另一个必备 trick 是使用相对位置编码的预训练模型，比如 XLNet（BERT 是使用绝对位置进行编码）。

对于超长文本往往需要进行分段。在分段的情况下，如果仅仅对于每个段仍直接使用 Transformer 中的位置编码，即每个不同段的同一个位置上仍然使用相同的位置编码，就会出现问题。比如，第 $i−2$  段和第 $i−1$  段的第一个位置将具有相同的位置编码，但它们对于第 i 段的建模重要性显然并不相同（例如第 $i−2$ 段中的第一个位置重要性可能要低一些）。因此，我们需要对它们进行区分。相对位置编码的具体细节和代码实现，我们将在下一章 BERT 源码解读中为大家讲解。

#### 短文本处理

对于句子长度较短的文本，Config（models/bert.py） 中的 self.pad_size 大可不必设置为 512，过多的无用的 pad 会影响模型的处理速度，以及占用更多的内存。
对于本节中所提到的 THUCNews 数据集，首先我们先对文本长度进行分析：

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# 设置显示风格
plt.style.use('fivethirtyeight')
# 读数据
train_data = pd.read_csv("./THUCNews/data/train.txt", sep="\t", names=['sentence','label'])`
```

对文本长度进行可视化处理：

```python
# 在训练数据中添加新的句子长度列, 每个元素的值都是对应的句子列的长度
train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))
# 绘制句子长度列的数量分布图
sns.countplot("sentence_length", data=train_data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
plt.show()
# 绘制dist长度分布图
sns.distplot(train_data["sentence_length"])
#主要关注dist长度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()
```

可以发现，文本长度基本符合正态分布。

我们通常采用分位值对数据进行补全/截断。例如采用超过 95% 文本长度的 sent_len 作为模型的最大输入长度。这种方法可以有效避免极少数超长数据对整个模型的影响，获取超过 95% 文本长度的 sent_len 的代码如下：

```python
sent_len = sorted(train_data["sentence_length"])[int(len(train_data) * 0.95)]
```



## 模型运行

众所周知，神经网络有许多常规超参数，如 epoch、batch size、learning rate 等，这类超参数的重要性不言而喻，大到直接影响到我们训练网络能否收敛，小到影响到最终的学习效果。所以，这一节来介绍这类参数如何调节的上分技巧。

#### Epoch

epoch：整个神经网络的训练迭代次数。一般该参数受训练任务难度的上升而增多。就拿基于 BERT 的 fine-tuning 来说，对于简单的分类任务来，通常仅需要 2-4 个 epoch 基本就可以收敛，而对于复杂的对话生成、阅读理解等任务，则往往需要 10 个以上。最简单的判断标准是根据损失函数值有没有持续的下降，来确定最合适的迭代次数。

如果不能预估模型收敛需要的 epoch，可以将 epoch 设置较大的值。然后通过提前终止训练的方式来节省不必要的计算开销。比如采用一个贪心的策略：当连续训练多个 batch 后，模型的 loss 并没有下降，这时候我们就无须关心模型还剩多少 epoch 没有训练，而直接终止模型，并将最优结果的模型参数保存下来。具体代码片段（models/bert.py）如下：

```python
for i in range(epochs):
  ...模型训练...
    if dev_loss < dev_best_loss:  
    dev_best_loss = dev_loss
    torch.save(model.state_dict(), config.save_path)
    last_improve = total_batch
  if total_batch - last_improve > 1000:
    print("No optimization for a long time, auto-stopping...")
    break
```


上述代码中。dev_loss 用来保存当前验证集的 loss， dev_best_loss 用来保存模型在验证集上的最优 loss，last_improve 保存的是获得dev_best_loss 时的 batch数，total_batch 表示当前训练的 batch 数。这段代码的意义在于，当验证集 loss 超过1000 batch 没下降时，结束训练。

### Batch size

batch size：用来更新梯度的批数据大小。这里，直接给出经验性的结论：在合理范围内，batch size 设置的得越大越好。

bach size （models/bert.py）的选择需要根据机器内存、句子长度、模型规模进行动态调整。这里给出一个经验基准，笔者使用的 GPU 是一台 16G 显存的 TITAN Xp，在内存不爆的情况下，对 batch size 的选取一般遵循：

* 1w 左右小数据 32
* 10w 左右中规模数据 64
* 100w 以上大规模数据 128
  另外，在实际运用中，往往需要根据实际场景的需求和模型表现来调整 bach size。对于重召回的场景，可能是上不封顶，希望开大 bach size。而对于数据少而干净、模型又小（glue benchmark）的情况，自然是 bach size 较小反而会更优（16 为佳，32 勉强，很少超过 64）。如果发现 bach size 不敏感，GPU 利用率又没打满，那么可以开大 bach size 加速训练。如果发现对 bach size 敏感，那就需要向着它敏感的正向方向去调优。

另外，batch size 设置大的好处：

较大的 batch size 有助于提高内存的利用率，使得大矩阵乘法的并行化效率提高。
跑完一次 epoch 所需的迭代次数减少，对于相同数据量的处理速度进一步加快。
在一定范围内，一般来说 batch size 越大，其确定的下降方向越准，引起训练震荡越小。
由于最终收敛精度会陷入不同的局部极值，因此 batch size 增大时更容易达到最终收敛精度上的最优。
注意这里说的是合理范围，如果盲目增大 batch size 则也会带来负面影响：

batch size 增大到一定程度，其确定的梯度下降方向已经基本不再变化。
太大的 batch size 容易陷入 sharp minima，泛化性不好。
由于模型规模、句子长度都是事先定义好的，这时候再调节 batch size 大小时，需要注意 GPU 的占用率，防止爆内存的情况出现。我们在命令行输入nvidia-smi 可以查看GPU的占有率。 Batch size 越大，GPU 占用率也就越高，如下图所示，由于训练文本较短，本模型设置的 batch size 为128，当前 GPU 占有率为 90%。

### Learning rate

learning rate ：学习率是直接调整网络权重的超参数。学习率越低，网络参数的变化速度就越慢。学但是习率设置过大时，模型可能会在局部最优点之间来回震荡导致模型无法收敛。学习率设置过大时，虽然可以确保模型不会错过任何局部极小值，但也意味着我们将花费更长的时间来进行收敛，费时费力。

### 基本学习率

和从零训练模型不同，由于 BERT 模型本身已经得到充分训练，对指定任务进行 fine-tune 时，学习率设置不应该过大。《How to Fine-Tune BERT for Text Classification？》一文中提到，BERT 的 fine-tune 学习率设置为 [5e-5, 3e-5, 2e-5] 通常可以取得较好的学习效果。可以看到，本模型中设置的学习率为 5e-5，在训练到第 2 个 epoch 的时候，模型就已经收敛了（models/bert.py）。

### 设置不同学习率

对于一些较难的 NLP 任务，在利用 BERT 的基础上，往往需要添加其他模型作为顶层结构（如序列标注任务中的典型结构是：BERT + BiLSTM + CRF；文本生成任务中的典型结构是：BERT + seq2seq）。

由于 BERT 是已经训练好的模型，所以它的学习率无须设置的很大，只进行微调即可。但是上层的模型结构之前未被训练，所有参数权重是随机初始化的，如果采用同样的 learning rate，那么模型收敛的速度将很慢。为此，在实践中，通常将上层结构的学习率调大，争取使两者在训练结束的时候同步：当 BERT 训练充分时，上层结构也训练充分了，最终的模型性能当然是最好的。

Pytorch 允许我们针对网络不同的层，设置不同的学习率。首先，我们来看一个简单的例子。

```python
class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.baseConv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 128, 3, 1, 1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(128, 128, 3, 1, 1)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(128, 128, 3, 1, 1)),
            ('relu3', nn.ReLU())])
        )
        self.conv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 3, 3, 1, 1)
    def forward(self, x):
        h = self.baseConv(x)
        h = self.conv4(h)
        result = self.conv5(h)
        return result
```


这是一个五层的卷积网络，假如此处对 baseConv 中前三层卷积层设置学习率为 lr=0.001，conv4 与 conv5 的学习率为 0.0001。

(1) 直接指定网络层参数的学习率

```python
net = Net5()
params = []
params += [{'param': net.baseConv.parameters(), 'lr': 0.001}]
params += [{'param': net.conv4.parameters(), 'lr': 0.0001}]
params += [{'param': net.conv5.parameters(), 'lr': 0.0001}]
solver = Adam(params=params)
```


(2) 根据网络参数名进行指定学习率

例如将 Net5 卷积层的 weight 的学习率设为 0.001，bias 的学习率为 0.0001

```python
lr = 0.001
params = []
base_params = list(net.named_parameters())
for name, value in base_params.items():
  if 'bias' not in name:
    params += [{'param': [value], 'lr': lr}]
  else:
    params += [{'param': [value], 'lr': lr * 0.1}]
solver = Adam(params=params)
```


base_params 存储的网络参数的列表，形式是[(tensor1_name, tensor1), (tensor2_name, tensor2)...]。

根据参数名字中是否有包含 bias 来设置不同参数类型的学习率。

本模型中同样采用了这种策略（train_eval.py），具体的代码是：

```python
optimizer_grouped_parameters += [{'params': [p for n, p in param_optimizer if 'BERT' not in n], 'lr': 1e-4}]
```


上层模型的学习率是 BERT 学习率的 2 倍。下面是在一个 epoch 的对比图：

￼

统一学习率模型收敛情况
￼

不同学习率模型收敛情况
可以看到，在预训练模型和上层模型的参数设置不同学习率的情况下，模型收敛的更快 。

### weight_decay

权重衰减（weight_decay[2]）等价于 L2 范数正则化。正则化通过在模型损失函数中添加惩罚项，使得模型的网络参数值不能太大。是常用的防止过拟合的手段。 L2 范数正则化是在模型原损失函数基础上添加 L2 范数惩罚项，其中 L2 范数惩罚项指的是模型权重的平方和与一个正的常数的乘积。比如，对于线性回归损失函数：

$$\iota(w_1, w_2, b) = \frac{1}{2}(x_1^{(i)}w_1+x_2^{(i)}w_2+b-y^{(i)})^2$$


其中 $w_1, w_2$ 为权重参数，样本数为 n, 将权重参数用向量 $w = [w_1, w_2]$ 表示，带有 L2 范数惩罚项的新的损失函数为:

$$\iota(w_1, w_2, b) + \frac{\lambda}{2n}\Vert w \Vert ^2$$


上式中 L2 范数的 $\Vert w \Vert ^2$ 展开后得到 $w_1^2+w_2^2$  。

那让我们一起看看，模型中该如何加入权重衰减呢？

对网络的正则化一般发生在与数据直接做乘法的参数中，所以在 BERT 中， LN 层和 bias 并不需要做。

```python
param_optimizer = list(model.named_parameters())
```

首先，我们先看看 BERT 参数名都是什么样子的

```pyhon
'BERT.embeddings.position_embeddings.weight'
'BERT.embeddings.token_type_embeddings.weight'
'BERT.embeddings.LayerNorm.weight'
'BERT.embeddings.LayerNorm.bias'
'BERT.encoder.layer.0.attention.self.query.weight'
'BERT.encoder.layer.0.attention.self.query.bias'
'BERT.encoder.layer.0.attention.self.key.weight'
'BERT.encoder.layer.0.attention.self.key.bias'
'BERT.encoder.layer.0.attention.self.value.weight'
'BERT.encoder.layer.0.attention.self.value.bias'
'BERT.encoder.layer.0.attention.output.dense.weight'
'BERT.encoder.layer.0.attention.output.dense.bias'
......
```


可以看到，LN层的的参数包括 BERT.embeddings.LayerNorm.weight、BERT.embeddings.LayerNorm.bias。普通的偏置权重是以bias结尾的，所以有如下代码（train_eval.py）：

```python
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
```


将模型中除了 LN 层和 bias 的参数都设置权重衰减率为 0.01，之后将所有参数送入优化器中：

```python
optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate)
```


结果如下，可以看到收敛速度又有了显著提升：

￼

权重衰减率的大小的选择，目前也是众说纷纭。有些人喜欢设置成较大的衰减率如 1e-2，有些也比较喜欢较小的衰减率如 1e-4，没有实验表明哪种更好，但可以肯定的是：对网络进行正则化能控制模型的复杂度，降低参数量级，提高模型泛化性能。所以在特定任务中需要多次尝试。

### warmup

warmup 是一种优化器学习率优化的方法，也是使用 BERT 这类预训练模型必不可少的重要参数。如图所示，简而言之 warmup 是一种要求模型先使用一个较小的学习率去更新模型的参数，然后随着训练过程的深入，逐渐增大至指定的训练参数的一种方法，这样做会使模型的最终收敛效果更好。

原理解释：

* BERT 作为已经通过海量数据训练好的预训练模型，其参数的本身已经经过了“精雕细琢”，如果训练模型初期，喂给模型的样本照成了网络的严重震荡，那么辛苦训练得到参数将前功尽弃

* .更糟糕的情况时，如果我们的学习率是随训练衰减的，那模型在一开始由于学习率很大，将“用力过猛”，可能会用很大的学习率学习到一些无关特征，后来逐渐变小的学习率可能再难将模型“拽”回到正道上。
  使用方法也很简单，只需要在优化器上设置如下两个参数（train_eval.py）。

  

  ```python
  optimizer = BERTAdam(optimizer_grouped_parameters,
                           lr=config.learning_rate,
                           warmup=0.05,
                           t_total=len(train_iter) * config.num_epochs)
  ```

  


warmup：是指启动系数

t_total: warmup作用的轮次，这里设置了全阶段启用（因为我们的 epoch 很小）

可以看到，模型收敛的效果也是非常明显。

### 输出阶段

而对于超长文本，《How to Fine-Tune BERT for Text Classification？》论文中还提出了一种多层方法：首先将文本划分为  段， 然后分别对每一段进行编码，最后对于每一段的编码结果通过以下三种方式进行多层融合：

* mean pooling
* max pooling
* attention pooling
  池化操作常常见于卷积层之后，通过池化来降低卷积层输出的特征向量，以防止模型出现过拟合现象。而这里我们可以理解为，对超长文本分段之后，每一段有一个所属类别。对于 mean pooling 的理解，可以认为是对每个段落的类比做一个平均；max pooling 则可以认为取所有段落中置信度最高的类别作为文章的类别；Attention pooling 则可以看做是句子各个词或词组的权重和，每个词的权重代表了该词对句子意思的贡献。

因为本例中不存在超长文本信息这种情况，我们用使用伪代码说明这种情况的处理方式：

```python
import torch.nn.functional as F
import torch.nn as F
# BERT的输出
context_outputs = BERT_encoder(input_text, attention_mask)
# 获取最后一层输出: (batch_size, seq_len, hidden_size)
last_hidden_state = context_outputs[0]
#对最后一层的输出做mean pooling
res = F.adaptive_avg_pool2d(last_hidden_state, (1,1)) 
#对最后一层的输出做max_pooling
MP = nn.MaxPool1d(3, stride=2)   #构建一个卷积核大小为1x3，步长为2的池化层
res = m(last_hidden_state)
```

本节课首先讲解了在使用 BERT 处理 NLP 问题时，预处理处理阶段所需要注意的一些问题。当然对于文本数据的预处理的工作远不止于此，比如：数据清洗、格式编码、去停用词、类别均衡等等。这些工作如果处理得当，都可以直接或间接的提升模型的性能。

其次讲解了在 BERT 运行阶段需要精调的参数。所有的精调都可以遵循从大到小的调整方向，即先通过较大的数值确定模型大致的收敛范围，然后启用较小的参数细化范围。当然还有一些更加具备理论性的调优算法：网格搜索、随机搜索、贝叶斯优化算法等。

最后讲解了在输出阶段通过更改网络的结构的方式，对 BERT 进行更加细粒度的调整，使得模型效果更优。增加隐含层数目以加深网络深度，会在一定程度上改善网络性能，但是当测试错误率不再下降时，就需要寻求其他的改良方法。本章代码更是为大家准备了 BERT-CNN、BERT-RNN等进阶结构，有兴趣的同学可以在 /models 文件下自行查阅。

好了，截止到现在 BERT 的精调相信大家也都基本掌握，但是我们做学问要知其然，还要知其所以然。下一章中，我们将对BERT的源码进行解读，一起学习吧！



在前面几节课的内容中，我们已经学会了如何使用 BERT。但是 BERT 作为预训练模型并不适用于所有场合。所以，我们要做到知其然还要知其所以然，才能很好的将 BERT 这把屠龙刀为我们所用。本节课将一起和大家学习 BERT 的源码部分，有助于后续对 BERT 代码的进一步加工和上层模型的搭建。

为方便读者更清楚理解 BERT 的运行流程，在源码解读方便，本章去掉了多余的配置和冗余校验等工作，只为读者展示最核心或可能需要用户自定义更改的代码部分。



## 系统整体架构

首先是加载不同任务的数据载入与处理模块

```python
processor = processors[args.task_name]()
```

以第二章的三个任务为例，不同的任务的对应不同的处理模块。

```python
processors = {
    "afqmc": AFQMCProcessor,
    "tnews": TNewsProcessor,
    "weiboner": WeiboNerProcessor
    }
```


以新闻分类任务为例，TNewsProcessor类 位于 prepro.py 文件中，其主要工作是训练样本生成、测试样本生成、标签获取等，这里需要用户根据自己的数据集，定制化的修改。

```python
class TNewsProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples
    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples
    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels
```


接着是模型的加载工作：通过终端传入的参数，加载相应的model对象。huggingface 提供了 BERT、gpt2、XLNet、t5 等众多模型的代码可供用户直接调用。

```python
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
```

以传入 bert 为例，系统将加载 BERT基本配置、BERT模型结构、 Tokenizer，

```python
MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "sentence_bert": (BertConfig, SentenceBERT, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
}
```


下面给出 BERT-Base 的配置信息。BERT-Base 是 Google 发布的众多版本中的一个版本（如下所示），不同版本的 BERT 结构类似，只是在层数和隐藏层神经元上略有不同。

￼

```python
# BERT基本配置
def __init__(
    self,
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=0,
    **kwargs
)
```

之后，便可以通过调用对应对象的 from_pretrained 函数，来进行相关配置的修改。

```python
# BERT基本配置
config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=args.task_name,
    cache_dir=args.cache_dir if args.cache_dir else None,
)
tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
)
model = model_class.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    cache_dir=args.cache_dir if args.cache_dir else None,
)
```
接着是数据加载工作，主要由 load_and_cache_examples 实现

```python
train_dataset, train_examples = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
```


这个函数中最重要的函数是 convert_examples_to_features ，其主要功能是将自然文本数据转化为 BERT 可训练的 tensor，而该函数内部最核心的函数又是 convert_text_to_ids ，该函数做了如下工作：

```python
#利用tokenizer对原文本切词后转为对应的token
tokens = tokenizer.tokenize(text, add_special_tokens=True) 
#前后补[CLS]、[SEP]
tokens = ["[CLS]"]+tokens[:max_length-2]+["[SEP]"]    
text_len = len(tokens)  
#截长补短，并将文本转化为对应的数字下标
input_ids = tokenizer.convert_tokens_to_ids(tokens+["[PAD]"]*(max_length-text_len))
attention_mask = [1]*text_len+[0]*(max_length-text_len)
token_type_ids = [0]*max_length
```


下一步，我们进入 train 函数：

```python
global_step, tr_loss = train(args, train_dataset, model, tokenizer)
# train 函数中，首先对 DataLoader 的加载，供模型批次读入数据：

train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
```

之后是配置优化器，设置热启动(warmup功能可参见第三课)：

```python
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
)
```


接下来，会对模型进行一些加速处理，使得模型可以自动打在多个GPU上训练或进行多机分布式训练。

```python
# multi-gpu training
if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)
# Distributed training
if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )
```


最后就是流程化的训练步骤了：

```python
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        model.train()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": batch[3]}
        outputs = model(**inputs)
        loss = outputs[0]  # 取模型输出
        ...
        optimizer.step()
        scheduler.step()   # 更新学习率
        model.zero_grad()
        global_step += 1
```



首先，我们先来看看 model.train() 到底是如何运行的。

模型的初始化主要由下面的代码决定，MODEL_CLASSES 返回了模型配置、模型结构、tokenizer 三个对象。其中 config_class 负责整体模型超参数的一些配置，tokenizer_class 负责将原文本切成 BERT 可识别的 token 数据。接下来本节主要对 model_class 模型结果进行分析。

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
查看代码得知，model_class 调用的是 BertForSequenceClassification 类，该类实现于 transformers 包的 modeling_bert.py 文件中，具体代码如下:

```python
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
```


可以看到，初始化函数中定义的结构并不多。BertModel 即是 BERT 本尊了，之后定义了一个 Dropout 层和一个 Linear 线性分类层。所以对于句子分类模型的前向传播结构，我们可以大胆的猜测一下，其仅仅是将 BERT 的输出向量取出之后， 添加一个线性分类层即可。BertForSequenceClassification 类的前向传播流程如下：

```python
@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
):
      #取BERT输出向量
    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
    )
    pooled_output = outputs[1]
    #经过dropout层
    pooled_output = self.dropout(pooled_output)
    #经过线性分类层
    logits = self.classifier(pooled_output)
        # 拼接句子隐藏层状态和attention结构
    outputs = (logits,) + outputs[2:]  
    # 如果存在labels，计算输出损失
    if labels is not None:
        if self.num_labels == 1:
            #如果labels只有一维，这里使用均方差损失
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            #否则使用交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs
    return outputs  # (loss), logits, (hidden_states), (attentions)
```


前向传播的网络结构如上，和我们预测的基本一致。

接下来，我们就需要看看 BertModel 是如何对我们的输入向量进行编码的。

下面给出 BertModel 保留核心部分后的代码。从前向函数中我们可以看到， BertModel 类主要做了三个工作：

对输入文本进行 embedding 操作，将词汇映射成 BERT 可处理的空间向量。
经过以 transformer 为核心结构的编码器模块，提取文本语义特征。
对上一步提取的语义特征和模型中间层所产生的附加文本特征进行进一步的处理和汇总工作

```python
class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()
   def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
 # 步骤一
    embedding_output = self.embeddings(
    input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
# 步骤二
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
# 步骤三
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
```

截止到目前为止，BERT 内部的运行流程我们已经梳理完毕了，下面开始梳理具体的代码细节。















