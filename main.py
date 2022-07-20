# 任务器类:封装方法，可训练模型，测试模型
import json
import os

import numpy as np
import torch
from datasets import load_metric
from transformers import TrainingArguments, DataCollatorWithPadding, AutoConfig, AutoModel
from dataset.ccac_dataset import CCACDataset
from models.net import CCACNet
from utils.my_trainer import MyTrainer
import torch.nn.functional as F


class Tasker():
    def __init__(self):
        # 初始化模型所有路径
        # self.init_path(model="albert-base-v2")
        self.init_path(model="albert-base-v2")
        self.trainer = None
        self.max_len=450

    def init_path(self, model='roberta-base'):
        # 训练过程输出文件存放路径
        self.output_dir_path = "output/" + model
        # 日志存放路径
        self.logging_dir_path = "log/" + model
        # 读取模型路径
        self.model_path = "models/" + model
        self.best_model_path = "best_models/" + model
        # 配置文件读取路径
        self.config_path = "models/" + model + "/config.json"
        self.best_config_path = "best_models/" + model + "/config.json"
        # 分词器读取路径--注意路径最后有反斜杆
        self.tokenizer_path = "models/" + model + "/"
        self.best_tokenizer_path = "best_models/" + model + "/"
        # 保存最优模型位置
        self.save_model_path = "best_models/" + model
        # 训练数据集的路径
        self.train_data_path = "corpus/train.csv"
        # 验证数据集的路径
        self.dev_data_path = "corpus/dev.csv"
        # 测试数据集的路径
        self.test_data_path = "corpus/test.csv"

    # 设置训练器的超参数
    # 详细参数说明见https://zhuanlan.zhihu.com/p/363670628
    def get_training_args(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir_path,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=5,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_dir=self.logging_dir_path,
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            # 如果网络前向计算的时候没用到labels则需要指定标签，否则trainer会在网络需要的参数中找不到labels，从而排除labels
            label_names=['labels'],
            seed=123
        )
        return training_args

    def train(self, best_model=False, use_fgm=False):
        """
        训练模型
        :param best_model: 是否加载最优的模型，默认只加载最原始的模型
        :return:
        """
        if best_model:
            Datasets = CCACDataset(tokenizer_path=self.best_tokenizer_path, train_data_path=self.train_data_path,
                                   dev_data_path=self.dev_data_path, test_data_path=self.test_data_path
                                   )
        else:
            Datasets = CCACDataset(tokenizer_path=self.tokenizer_path, train_data_path=self.train_data_path,
                                   dev_data_path=self.dev_data_path, test_data_path=self.test_data_path
                                   )
        # 得到训练器的超参数
        training_args = self.get_training_args()
        tokenizer = Datasets.get_tokenizer()
        # 加载训练数据
        train_datasets = Datasets.LoadTrainDataset()
        # 加载验证数据
        dev_datasets = Datasets.LoadDevDataset()
        # 动态填充，即将每个批次的输入序列填充到一样的长度
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding='max_length',max_length=self.max_len)
        # 加载模型
        # 判断是否加载最优的模型
        if best_model:
            # 加载模型配置 output_hidden_states是否获取所有隐藏层的输出
            config = AutoConfig.from_pretrained(self.best_config_path, output_hidden_states=True)
            model = CCACNet(model_path=self.best_model_path, config=config,max_len=self.max_len)
        else:
            # 加载模型配置 output_hidden_states是否获取所有隐藏层的输出
            config = AutoConfig.from_pretrained(self.config_path, output_hidden_states=True)
            model = CCACNet(model_path=self.model_path, config=config,max_len=self.max_len)
        # 构造训练器
        trainer = MyTrainer(
            model,
            training_args,
            train_dataset=train_datasets,
            eval_dataset=dev_datasets,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            use_fgm=use_fgm
        )
        # 训练模型
        trainer.train()
        #加载最优模型评估
        trainer.evaluate(dev_datasets)
        # 保存模型
        trainer.save_model(self.save_model_path)
        # trainer.model.save_pretrained(self.save_model_path)
        # trainer.tokenizer.save_pretrained(self.save_model_path)
        # 保存一下trainer,方便训练完之后紧接着测试
        self.trainer = trainer

    def test(self):
        Datasets = CCACDataset(tokenizer_path=self.best_tokenizer_path, train_data_path=self.train_data_path,
                               dev_data_path=self.dev_data_path, test_data_path=self.test_data_path
                               )
        # 加载测试数据
        test_datasets = Datasets.LoadTestDataset()
        if self.trainer == None:
            # 得到训练器的超参数
            training_args = self.get_training_args()
            tokenizer = Datasets.get_tokenizer()
            # 加载训练数据
            train_datasets = Datasets.LoadTrainDataset()
            # 加载验证数据
            dev_datasets = Datasets.LoadDevDataset()
            # 动态填充，即将每个批次的输入序列填充到一样的长度
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding='max_length',max_length=self.max_len)
            # 加载训练过的最优模型
            # 加载模型配置 output_hidden_states是否获取所有隐藏层的输出
            config = AutoConfig.from_pretrained(self.best_config_path, output_hidden_states=True)
            model = CCACNet(model_path=self.best_model_path, config=config,max_len=self.max_len)
            # 构造训练器
            trainer = MyTrainer(
                model,
                training_args,
                train_dataset=train_datasets,
                eval_dataset=dev_datasets,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
            # 使用训练器预测模型
            logits, label_ids, metrics = trainer.predict(test_datasets)
            print(metrics)
            # 生成提交结果的文件
            # ids=tokenized_test_datasets["test"]["id"]
            # save_submition(ids,logits)
        else:
            # 使用训练器预测模型
            logits, label_ids, metrics = self.trainer.predict(test_datasets)
            print(metrics)
            # 生成提交结果的文件
            # ids = tokenized_test_datasets["test"]["id"]
            # save_submition(ids, logits)

    def print_model(self):
        f = open('log.txt', 'w')
        # 加载模型配置 output_hidden_states是否获取所有隐藏层的输出
        config = AutoConfig.from_pretrained(self.best_config_path, output_hidden_states=True)
        model = CCACNet(model_path=self.best_model_path, config=config, max_len=self.max_len)
        # model=AutoModel.from_pretrained(self.best_model_path, config=config)
        # print(model, file=f)
        for parameters in model.parameters():#打印出参数矩阵及值
                print(parameters,file=f)

# 指标评估
def compute_metrics(eval_preds):
    # metric = load_metric("glue", "mrpc")
    # 没网络时从本地加载
    metric = load_metric("utils/glue.py", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 生成提交数据
def save_submition(ids, logits):
    # 转为tensor
    logits = torch.from_numpy(logits)
    filename = r'corpus\submit.json'
    # 按行进行softmax，得到预测的概率
    predictions = F.softmax(logits, dim=1)
    # 以第二维度为标准，获取该维度最大值的下标
    pre_label = torch.argmax(predictions, dim=1).numpy()
    # 将预测结果写入文件
    with open(filename, 'w') as f:
        for id in ids:
            result = {}
            result["id"] = id
            result["label"] = str(pre_label[id])
            line = json.dumps(result)
            f.write(line + '\n')



if __name__ == '__main__':
    tasker = Tasker()
    # 训练
    tasker.train(best_model=True, use_fgm=False)
    # 测试
    # tasker.test()
    # tasker.print_model()