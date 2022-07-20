from datasets import load_dataset
from transformers import AutoTokenizer
import re


class CCACDataset():
    def __init__(self, tokenizer_path, train_data_path, dev_data_path, test_data_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.train_data_path = train_data_path
        self.dev_data_path = dev_data_path
        self.test_data_path = test_data_path

        # 加载训练数据集

    def LoadTrainDataset(self):
        # 加载训练和验证的原始数据，成为一个DatasetDict对象，里面的每一个数据集为Dataset可迭代对象
        train_datasets = load_dataset("csv",data_files={"train":self.train_data_path}, split='train[:]')
        # tokenizer映射到原始数据上
        tokenized_train_datasets = train_datasets.map(self.tokenize_function, remove_columns="label", batched=True)
        return tokenized_train_datasets

        # 加载验证数据集
    def LoadDevDataset(self):
        # 加载训练和验证的原始数据，成为一个DatasetDict对象，里面的每一个数据集为Dataset可迭代对象
        dev_datasets = load_dataset("csv",data_files={"validation":self.dev_data_path}, split='validation[:]')
        # tokenizer映射到原始数据上
        tokenized_dev_datasets = dev_datasets.map(self.tokenize_function, remove_columns="label", batched=True)
        return tokenized_dev_datasets

        # 加载测试数据集

    def LoadTestDataset(self):
        # 加载测试的原始数据，成为一个DatasetDict对象，里面的每一个数据集为Dataset可迭代对象
        test_datasets = load_dataset("csv", data_files=self.test_data_path)
        # tokenizer映射到原始数据上
        tokenized_test_datasets = test_datasets.map(self.tokenize_function, batched=True)
        return tokenized_test_datasets

        # 获取分词器

    def get_tokenizer(self):
        return self.tokenizer

    def tokenize_function(self, example):
        if isinstance(example["sentence1"], list):
            sentence1_list = []
            for item in example["sentence1"]:
                sentence1_list.append(re.sub('\[(.*?)]', '', str(item)))
            example["sentence1"] = sentence1_list
        else:
            example["sentence1"] = re.sub('\[(.*?)]', '', str(example["sentence1"]))
        if isinstance(example["sentence2"], list):
            sentence2_list = []
            for item in example["sentence2"]:
                sentence2_list.append(re.sub('\[(.*?)]', '', str(item)))
            example["sentence2"] = sentence2_list
        else:
            example["sentence2"] = re.sub('\[(.*?)]', '', str(example["sentence2"]))

        tokenized_example = self.tokenizer(example["sentence1"], example["sentence2"], padding=True, truncation=True)
        if 'label' in example.keys():
            tokenized_example['labels'] = example['label']
        return tokenized_example


if __name__ == '__main__':
    dataset = CCACDataset('../models/roberta-wwm-ext/')
    dataset.LoadTrainDataset()
