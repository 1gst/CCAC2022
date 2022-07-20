import torch
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM


class CCACNet(PreTrainedModel):
    def __init__(self, model_path, config, max_len, dropout_rate=0.2, class_num=2):
        super(CCACNet, self).__init__(config)
        self.max_len = max_len
        self.config = config

        # 加载预训练模型
        self.bert = AutoModelForMaskedLM.from_pretrained(model_path, config=self.config)
        # dropout与分类网络
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(30000, class_num)

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
        # hidden_state = torch.mean(outputs.get('last_hidden_state'), 1)
        # 得到模型的最后一层没有softmax的cls输出
        # cls_output = outputs.get('pooler_output')
        # 把输出cat起来
        # output = torch.cat([hidden_state, cls_output], dim=-1)
        logits = self.fc(self.fc_dropout(torch.mean(outputs.get('logits'), 1)))
        return logits
