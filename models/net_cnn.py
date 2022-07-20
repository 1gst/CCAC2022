import torch
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from transformers import AutoModel


# 参考https://blog.csdn.net/Delusional/article/details/114549281
# https://blog.csdn.net/weixin_46425692/article/details/108397005
# config.vocab_size  ## 已知词库大小
# config.embedding_size  ##每个词向量长度
# config.num_clas  ##类别数
# config.out_channels = 16  ## 输出卷积核的个数
# args.kernel_sizes  ## 卷积核list，形如[3,4,5]


class TextCNN(nn.Module):
    def __init__(self, max_len, embedding_size=768, out_channels=16, kernel_sizes=None):
        super(TextCNN, self).__init__()
        # self.dropout_rate = config.dropout_rate
        # self.num_class = config.num_clas

        # self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
        #                               embedding_dim=config.embedding_size)
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        if kernel_sizes is None:
            self.kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.embedding_size,
                          out_channels=self.out_channels,
                          kernel_size=ks),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.max_len - ks + 1))
            for ks in
            self.kernel_sizes])  # 创建3个nn.Sequential，包含了 图中的convolution层、activation function层 和 maxPooling层, 其中每个层的参数都有变化
        # self.fc = nn.Linear(in_features=config.out_channels * len(config.kernel_size),
        #                     out_features=config.num_class)  # 每种类别的卷积核个数相乘，得到的长度就是全连接层输入的长度

    def forward(self, inputs):
        # embed_x = self.embedding(x)  # b x src_len
        inputs = inputs.permute(0, 2, 1)
        # b x src_len x embed_size --> b x embed_size x src_lem
        out = [conv(inputs) for conv in self.convs]  # 计算每层卷积的结果，这里输出的结果已经经过池化层处理了
        out = torch.cat(out, dim=1)  # 对池化后的向量进行拼接
        out = out.view(-1, out.size(1))  # 拉成一竖条作为全连接层的输入
        # out = F.dropout(input=out,
        #                 p=self.dropout_rate)  # 这里也没有在图中的表现出来，这里是随机让一部分的神经元失活，避免过拟合。它只会在train的状态下才会生效。进入train状态可查看nn.Module。train()方法
        # out = self.fc(out)
        return out

    def get_outsize(self):
        return self.out_channels * len(self.kernel_sizes)


class CCACNet(PreTrainedModel):
    def __init__(self, model_path, config, max_len, dropout_rate=0.2, class_num=2):
        super(CCACNet, self).__init__(config)
        self.max_len = max_len
        self.config = config
        # 加载预训练模型
        self.bert = AutoModel.from_pretrained(model_path, config=self.config)
        # textcnn
        self.cnn = TextCNN(max_len=self.max_len, embedding_size=self.config.hidden_size, out_channels=16,
                           kernel_sizes=[3, 4, 5])
        # dropout与分类网络
        self.fc_dropout = nn.Dropout(dropout_rate)
        out_num = self.cnn.get_outsize()
        self.fc = nn.Linear(self.config.hidden_size + out_num, class_num)

    def forward(self, input_ids, attention_mask):
        """
        前向计算
        :param input_ids:[batch_size,seq_len]
        :param token_type_ids:[batch_size,seq_len]
        :param attention_mask:[batch_size,seq_len]
        :return:
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                               return_dict=True)
        # 得到模型的最后一层没有softmax的隐藏层输出
        cnn_output = self.cnn(outputs.get('last_hidden_state'))
        # 得到模型的最后一层没有softmax的cls输出
        cls_output = outputs.get('pooler_output')
        # 把输出cat起来
        output = torch.cat([cnn_output, cls_output], dim=-1)
        logits = self.fc(self.fc_dropout(output))
        return logits
