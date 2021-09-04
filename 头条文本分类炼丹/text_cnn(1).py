# 词向量！
# 自己训练

import torch
from torch import nn
import torch.nn.functional as F
import jieba
from embedding import get_embedding


class TextCNN(nn.Module):
    def __init__(self, word_embedding, each_filter_num, filter_heights, drop_out, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_embedding, freeze=True)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=each_filter_num,
                      kernel_size=(h, word_embedding.shape[0]))
            for h in filter_heights
        ])

        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(each_filter_num * len(filter_heights), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, input_ids=None):
        word_embeddings = self.embedding(input_ids)
        sentence_embedding = word_embeddings.unsqueeze(1)

        out = torch.cat([self.conv_and_pool(sentence_embedding, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)

        outputs = (out, )

        return outputs


if __name__ == '__main__':
    some_text_sentence = '今天股市大跌'
    words = list(jieba.cut(some_text_sentence))
    embedding, token2id, _ = get_embedding(set(words))

    text_cnn_model = TextCNN(embedding, each_filter_num=128, filter_heights=[2, 3, 5], drop_out=0.3,
                             num_classes=15)

    ids =[token2id[w] for w in words]

    some_text_sentence = '测试一个新句子'
    words = list(jieba.cut(some_text_sentence))
    embedding, token2id, _ = get_embedding(set(words))

    # out = text_cnn_model(torch.tensor([ids]))

    # print(out)