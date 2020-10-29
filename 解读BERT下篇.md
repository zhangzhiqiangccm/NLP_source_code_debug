## BERT内部细节

### Embedding

Bert中的embedding层由 word embedding, token_type_embeddings 和 position embeddings 三者相加得到，前两者都比较普遍，关键是position_embeddings。

Bert结构不包含递归和循环结构，为了使模型能够有效利用序列的顺序特征，我们需要加入序列中各个Token间相对位置或者Token在序列中绝对位置的信息。Bert模型使用的是Learned Positional Embedding编码绝对位置。直接对不同的位置随机初始化一个position embedding，将其加到token embedding上输入模型，作为参数进行训练。

```python
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
```

三块embedding相加，并作层归一化：

```python
def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
    embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings
```



### Encoder		

Bert的核心编码层包括一个12层的transformer结构。比较核心的是Self_attention的计算，宏观上分为attention和Forward两个子层，每个子层之间，Bert都采用了残差结构并进行了层级归一化操作。残差块有助于解决梯度弥散问题。

```python
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # attention结构
        self.attention = BertAttention(config)
        # 解码结构
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        # 中间层的目的是为了对齐维度
        self.intermediate = BertIntermediate(config)
        # 每一层的输出连接结构
        self.output = BertOutput(config)
```

层级连接：

```python
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # attention结构
        self.attention = BertAttention(config)
        # 解码结构
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        # 中间层的目的是为了对齐维度
        self.intermediate = BertIntermediate(config)
        # 每一层的输出连接结构
        self.output = BertOutput(config)
首先看一下前向传播中的流程，BERT 中的每一层都实现了如下的结构
```

经典的BertAttention结构：

```python
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
```

SelfAttention结构：

```python
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        # 多头注意力中，每个头的size
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 所有头的累计size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 实现Q、K、V的多头线性映射
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # Dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
```

Attention 层可以将 Query 和一组 Key-Value 对映射到输出，其中 Query、Key、Value 和输出都是向量形式。 输出是值的加权和，其中分配给每个 Value 的权重由 Query 与相应 Key 计算得来。我们称这种特殊的 attention 机制为”Scaled Dot-Product Attention”。输入包含维度为 $d_k$ 的 Query 和 Key，以及维度为 $d_v$ 的 Value。 我们首先分别计算 Query 与各个 Key 做点积运算，然后将每个点积结果除以 $\sqrt[]{(d_k)}$，最后使用 Softmax 函数来获得 Key 的权重。

在具体实现时，我们可以以矩阵的形式进行并行运算，这样能充分利用 GPU 加速运算过程。（多条数据为一个 batch，Query、Key 和 Value 分别组合为矩阵）具体来说，将所有的 Query、Key 和 Value 向量分别组合成矩阵 Q、K 和 V，这样输出矩阵可以表示为：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

这部分在前向传播中的代码：

```python
def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)
        # 分别对Q、K、V的实现多头线性映射
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # 计算query和key做点乘注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        # 利用softmax将注意力分数正则化为概率
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        # 对value进行注意力得分乘法
        context_layer = torch.matmul(attention_probs, value_layer)
        
       #forward 函数中继续执行了多头注意力的解码
       # 多头注意力concat
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
```

残差连接与层归一化代码：

```python
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 残差 + 层级归一化操作
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
 class BertLayerNorm(nn.Module):
    # "构建LN层"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)   #求当前层均值
        std = x.std(-1, keepdim=True)     #求当前层方差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2   #对当前层进行LN
```

虽然线性变换在不同位置上是相同的，但它们在层与层之间使用不同的参数。这其实是相当于使用了两个内核大小为 1 的卷积。

```python
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
```


上述代码中，intermediate_act_fn 正是公式中 $$\max(0, xW_1 + b_1)) $$的体现。

Masked部分

mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。其中，padding mask 在所有的 scaled dot-product attention 里面都需要用到，而 sequence mask 只有在 decoder 的 self-attention 里面用到。

```python
encoder_outputs = self.encoder(
    embedding_output,
    attention_mask=extended_attention_mask,
    head_mask=head_mask,
    encoder_hidden_states=encoder_hidden_states,
    encoder_attention_mask=encoder_extended_attention_mask,
)
def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: tuple, device: device):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # 解码过程中的sequence mask,上三角矩阵的形成
        if self.config.is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            causal_mask = causal_mask.to(attention_mask.dtype)
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    #scaled dot-product attention中的padding mask
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
```

Pooler

取决于不同的任务。

## BERT模型的改进

改进可以从两个预训练任务入手：1.Mask Language Model 任务；2. Next Sentence Prediction (NSP) 任务。

对于MLM任务，可以有多种改进思路，一种是继续做生成式的语言模型，不止需要生成一个字，而是要生成一个词或者短语；另一种思路是用判别式任务代替语言模型，可以得到精度更高的模型，但由于判别式任务的分类边界少，一定程度上降低了模型的表达能力。

### 调参

#### 混合精度训练——更多的调参时间，更精细的超参数调节

哪怕仅仅是在下游任务上微调，训练 BERT 模型也往往会消耗大量的计算资源。假如具备了一定的调参能力，那么在时间和算力确定的情况下，让每次的实验时间尽可能短，就有助于争取到更多的调参次数，自然就有望找到更优的超参数，提升模型的表现。因此，加速 BERT 模型的训练虽然不会直接带来任务精度上的收益，但是节省下来的时间却允许你进行更多的实验，在打比赛等时间有限的时候，这些更多次的实验就容易创造奇迹，带来最后一公里的性能提升。

于是问题聚焦在训练速度上，该怎样快速提升大模型训练速度呢？

百度研究院和英伟达研究院在 2017 年的深度学习顶会 ICLR 上发表了一篇论文《Mixed Precision Training》，即混合精度训练。该方法如今已成为各大深度学习框架的内置技能，仅需寥寥数行代码，就能开启混合精度训练模式。

在混合精度训练的过程中，前向计算会临时从默认的f单精度浮点类型 float32 降到半精度浮点类型 float16，运算精度虽然有所下降，但是运算速度却大大加快了，从而达到了加速模型训练的目的。当然了，虽然运算精度下降了，但是混合精度训练算法会通过参数拷贝、loss 缩放等机制来对抗运算精度下降可能带来的模型性能下降。经验上在 BERT 上使用混合精度训练会带来约 1.7X 的训练速度提升，且一般不会影响模型泛化能力。

此外，如果是比赛或模型部署场景对推理速度有较高的要求，将前向计算的运算精度降低到 float16 还是一种非常容易实现的加速方法，且这种推理加速方法一般不会造成推理结果的改变。

在 Pytorch 中实现混合精度训练也非常容易，英伟达提供的 Apex 库中集成了自动混合精度的实现（AMP），基于 AMP，往往只需要在 BERT 代码的基础上额外增加两三行代码即可实现。

不过需要注意的是，混合精度训练在具备 Tensor Core 的显卡上才能发挥最大功效。比较旧的显卡架构（如 Kepler、Maxwell 架构）一般没有 FP16 单元，因此无法应用混合精度训练；而 Pascal 架构的显卡（如 Titan X，P40）虽然支持 FP16，但是由于没有 Tensor Core，可能带来的加速效果不够显著。要发挥混合精度训练的最佳威力，建议在 Volta 架构（如 V100）或其他注明有 Tensor Core 的显卡上尝试。

#### 显存重计算

使用更深更大更先进的预训练模型，就意味着模型天然拥有更强大的数据拟合能力与高层特征抽取能力，越有望解决困难的自然语言处理任务，以及普通任务中的困难样本。

无论是 BERT，还是 XLNet、RoBERTa、ALBERT 等后续出现的预训练模型，官方一般都会放出多种尺寸的预训练模型。一般来说，更宽、更深的预训练模型往往在预训练阶段记住了更多的先验知识，拥有了更上层的特征抽取和表示能力，因此往往在较为困难的下游任务中比尺寸较小的预训练模型表现出更强的学习能力。

然而，更大尺寸的预训练模型，不仅会训练更慢，而且显存开销也会急剧增加。以 BERT Large为例，如果文本长度达到 512，那么训练 BERT Large 的话，哪怕用上当今最强计算显卡V100（单卡高达 32GB 显存），batch size 也往往只能开到 8 左右。对显存的开销，使得很多时候我们都无法尝试更大的预训练模型到底能带来多少额外的性能增益。

于是问题聚焦在显存开销上，该怎样打破显存不足的瓶颈呢？

显存重计算，或者说梯度检查点（gradient checkpoint）技术就是专门为训练大模型、解决显存不足而诞生的。

显存重计算方法由陈天奇等人于 2016 年提出的，这篇文章《Training deep nets with sublinear memory cost》用时间换空间的思想，在前向时只保存部分中间节点，在反向时重新计算没保存的部分。论文通过这种机制，在每个 batch 只多计算一次前向的情况下，把 n 层网络的占用显存优化到了 O(√n)。在极端情况下，仍可用O(nlogn) 的计算时间换取到 O(logn) 的显存占用。在论文的实验中，他们成功将将 1000 层的残差网络从 48G 优化到了 7G。而且，这种方法同样可以直接应用于 RNN 结构中。

我们知道，神经网络的一次训练包含前向计算、后向计算和优化三个步骤。在这个过程中，前向计算会输出大量的隐层变量 Tensor，当模型层数加深时，Tensor 数量可达成千上万个。如BERT Large 模型，单个 Tensor 可达到 1GB，这些 Tensor 在显存中累积，显存很快就爆掉了，这是大型预训练吃显存的最关键原因。

￼因此，这个方法就是在每个训练 step 的前向过程中保存若干检查点，计算到 loss 层时模型的前向部分只保存了检查点，而不是保存全部中间变量，这样就省下了大量的用于存储中间变量的显存。具体来说，重计算先将前向计算分割成多个段，将每个段的起始 Tensor 作为这个段的检查点（checkpoints）。前向计算时，除了检查点以外的其他隐层 Tensor 占有的显存可以及时释放。反向计算用到这些隐层 Tensor 时，从前一个检查点开始，重新进行这个段的前向计算，就可以重新获得隐层 Tensor。

经验上来讲，在 BERT 上面应用重计算技术后，显存开销可以降低约 4 倍，这就意味着我们甚至可以在 K40 这种古老的显卡上跑起来 BERT Large 这种大型预训练模型，进而获得更好的下游任务表现。

如今显存重计算技术也已被 Pytorch、PaddlePaddle 等深度学习框架纳入标配。

