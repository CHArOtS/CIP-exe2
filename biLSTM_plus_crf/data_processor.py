import os
import json
import pickle

import torch
from torch.utils.data import Dataset


def get_label_map(data_path=''):
    # 标签字典保存路径
    label_map_path = '../data/label_map.json'

    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as fp:
            label_map = json.load(fp)
    else:
        # datapath目前没有使用，数据集中的标签数目较少且固定的，对复杂的语料需要另外编写代码读取
        """
        # 读取json数据
        json_data = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                json_data.append(json.loads(line))
        '''
        json_data[0]数据为该格式：

        {'text': '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，',
         'label': {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}}
        '''
        # 统计共有多少类别
        n_classes = []
        for data in json_data:
            for label in data['label'].keys():  # 获取实体标签，如'name'，'company'
                if label not in n_classes:  # 将新的标签加入到列表中
                    n_classes.append(label)
        n_classes.sort()
        """
        n_classes = ['PER', 'LOC', 'ORG']
        # 设计label_map字典，对每个标签设计两种并设置ID值
        label_map = {}
        for n_class in n_classes:
            label_map['B-' + n_class] = len(label_map)
            label_map['I-' + n_class] = len(label_map)
        label_map['O'] = len(label_map)
        # 对BiLSTM+CRF网络，增加除BIO标签外的开始和结束标签，以增强其标签约束能力
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        label_map[START_TAG] = len(label_map)
        label_map[STOP_TAG] = len(label_map)

        # 将label_map字典存储成json文件
        with open(label_map_path, 'w', encoding='utf-8') as fp:
            json.dump(label_map, fp, indent=4)

    # 翻转标签字典，以预测时输出的序列为索引，方便转换成中文汉字
    label_map_inv = {v: k for k, v in label_map.items()}
    return label_map, label_map_inv


def get_vocab(data_path=''):
    # 词表保存路径
    vocab_path = '../data/vocab.pkl'
    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as fp:
            vocab = pickle.load(fp)
    else:
        text_data = []
        # 加载数据集
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                if len(line) != 2:
                    continue
                text_data.append(line.split()[0])

        # 建立词表字典，提前加入'PAD'和'UNK'
        # 'PAD'：在一个batch中不同长度的序列用该字符补齐
        # 'UNK'：当验证集或测试集出现词表以外的词时，用该字符代替
        vocab = {'PAD': 0, 'UNK': 1}
        # 遍历数据集，不重复取出所有字符，并记录索引
        for character in text_data:
            if character not in vocab:
                vocab[character] = len(vocab)
        # vocab：{'PAD': 0, 'UNK': 1, '浙': 2, '商': 3, '银': 4, '行': 5...}
        # 保存成pkl文件
        with open(vocab_path, 'wb') as fp:
            pickle.dump(vocab, fp)

    # 翻转字表，预测时输出的序列为索引，方便转换成中文汉字
    # vocab_inv：{0: 'PAD', 1: 'UNK', 2: '浙', 3: '商', 4: '银', 5: '行'...}
    vocab_inv = {v: k for k, v in vocab.items()}
    return vocab, vocab_inv


def data_read(path):
    # 将预处理过的res语料文件处理成text和label两个列表的格式 类似HMM模型中的操作
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read()
    lines = lines.split('\n')
    data = []
    text = []
    label = []
    for line in lines:
        if line == '':
            if len(text) > 0:
                data.append([text, label])
                text = []
                label = []
                continue
            else:
                continue
        text.append(line.split()[0])
        label.append(line.split()[1])
    return data


class Model_Dataset(Dataset):
    def __init__(self, file_path, vocab, label_map):
        self.file_path = file_path
        # 数据预处理
        self.data = data_read(self.file_path)
        self.label_map, self.label_map_inv = label_map
        self.vocab, self.vocab_inv = vocab
        # self.data为中文汉字和英文标签，将其转化为索引形式
        self.examples = []
        for text, label in self.data:
            t = [self.vocab.get(t, self.vocab['UNK']) for t in text]
            l = [self.label_map[l] for l in label]
            self.examples.append([t, l])

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.data)

    def collect_fn(self, batch):
        # 取出一个batch中的文本和标签，将其单独放到变量中处理
        # 长度为batch_size，每个序列长度为原始长度
        text = [t for t, l in batch]
        label = [l for t, l in batch]
        # 获取一个batch内所有序列的长度，长度为batch_size
        seq_len = [len(i) for i in text]
        # 提取出最大长度用于填充
        max_len = max(seq_len)

        # 填充到最大长度，文本用'PAD'补齐，标签用'O'补齐
        text = [t + [self.vocab['PAD']] * (max_len - len(t)) for t in text]
        label = [l + [self.label_map['O']] * (max_len - len(l)) for l in label]

        # 将其转化成tensor，再输入到模型中，这里的dtype必须是long否则报错
        # text 和 label shape：(batch_size, max_len)
        # seq_len shape：(batch_size,)
        text = torch.tensor(text, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)

        return text, label, seq_len