import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn import metrics
from itertools import chain


# 设计tag2idx字典，对每个标签设计两种，如B-name、I-name，并设置其ID值
n_classes = ['PER', 'LOC', 'ORG']
tag2idx = defaultdict()
tag2idx['O'] = 0
count = 1
for n_class in n_classes:
    tag2idx['B-' + n_class] = count
    count += 1
    tag2idx['I-' + n_class] = count
    count += 1


def data_read(path):
    # 将预处理过的res语料文件处理成text和label两个列表的格式
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


class HMM_model:
    def __init__(self, tag2idx):
        self.tag2idx = tag2idx  # tag2idx字典
        self.n_tag = len(self.tag2idx)  # 标签个数
        self.n_char = 65535  # 所有字符的Unicode编码个数，包括汉字
        self.epsilon = 1e-100  # 无穷小量，防止归一化时分母为0
        self.idx2tag = dict(zip(self.tag2idx.values(), self.tag2idx.keys()))  # idx2tag字典
        self.A = np.zeros((self.n_tag, self.n_tag))  # 状态转移概率矩阵, shape:(21, 21)
        self.B = np.zeros((self.n_tag, self.n_char))  # 观测概率矩阵, shape:(21, 65535)
        self.pi = np.zeros(self.n_tag)  # 初始隐状态概率,shape：(21,)

    def train(self, train_data):
        print('开始训练数据：')
        for i in tqdm(range(len(train_data))):  # 几组数据
            for j in range(len(train_data[i][0])):  # 每组数据中几个字符
                cur_char = train_data[i][0][j]  # 取出当前字符
                cur_tag = train_data[i][1][j]  # 取出当前标签
                self.B[self.tag2idx[cur_tag]][ord(cur_char)] += 1  # 对B矩阵中标签->字符的位置加一
                if j == 0:
                    # 若是文本段的第一个字符，统计pi矩阵
                    self.pi[self.tag2idx[cur_tag]] += 1
                    continue
                pre_tag = train_data[i][1][j - 1]  # 记录前一个字符的标签
                self.A[self.tag2idx[pre_tag]][self.tag2idx[cur_tag]] += 1  # 对A矩阵中前一个标签->当前标签的位置加一

        # 防止数据下溢,对数据进行对数归一化
        self.A[self.A == 0] = self.epsilon
        self.A = np.log(self.A) - np.log(np.sum(self.A, axis=1, keepdims=True))
        self.B[self.B == 0] = self.epsilon
        self.B = np.log(self.B) - np.log(np.sum(self.B, axis=1, keepdims=True))
        self.pi[self.pi == 0] = self.epsilon
        self.pi = np.log(self.pi) - np.log(np.sum(self.pi))

        # 将A，B，pi矩阵保存到本地
        np.savetxt('./hmm/A.txt', self.A)
        np.savetxt('./hmm/B.txt', self.B)
        np.savetxt('./hmm/pi.txt', self.pi)
        print('训练完毕！')

    # 载入A，B，pi矩阵参数
    def load_paramters(self, A='./hmm/A.txt', B='./hmm/B.txt', pi='./hmm/pi.txt'):
        self.A = np.loadtxt(A)
        self.B = np.loadtxt(B)
        self.pi = np.loadtxt(pi)

    # 使用维特比算法进行解码
    def viterbi(self, s):
        # 计算初始概率，pi矩阵+第一个字符对应各标签概率
        delta = self.pi + self.B[:, ord(s[0])]
        # 前向传播记录路径
        path = []
        for i in range(1, len(s)):
            # 广播机制，重复加到A矩阵每一列
            tmp = delta.reshape(-1, 1) + self.A
            # 取最大值作为节点值，并加上B矩阵
            delta = np.max(tmp, axis=0) + self.B[:, ord(s[i])]
            # 记录当前层每一个节点的最大值来自前一层哪个节点
            path.append(np.argmax(tmp, axis=0))

        # 回溯，先找到最后一层概率最大的索引
        index = np.argmax(delta)
        results = [self.idx2tag[index]]
        # 逐层回溯，沿着path找到起点
        while path:
            tmp = path.pop()
            index = tmp[index]
            results.append(self.idx2tag[index])
        # 序列翻转
        results.reverse()
        return results

    def predict(self, s):
        results = self.viterbi(s)
        for i in range(len(s)):
            print(s[i] + results[i], end=' | ')

    def predict_to_List(self, s):
        results = self.viterbi(s)
        char_list = []
        tag_list = []
        for i in range(len(s)):
            char_list.append(s[i])
            tag_list.append(results[i])
        return char_list, tag_list

    def valid(self, valid_data):
        y_pred = []
        # 遍历验证集每一条数据，使用维特比算法得到预测序列，并加到列表中
        for i in range(len(valid_data)):
            y_pred.append(self.viterbi(valid_data[i][0]))
        return y_pred


train_data = data_read('./data/train_process.txt')
valid_data = data_read('./data/test_process.txt')
print("=====================================")
print('训练集长度:', len(train_data))
print('验证集长度:', len(valid_data))
print("=====================================")

model = HMM_model(tag2idx)
model.train(train_data)
model.load_paramters()


y_pred = model.valid(valid_data)
y_true = [data[1] for data in valid_data]

# 排好标签顺序输入，否则默认按标签出现顺序进行排列
sort_labels = [k for k in tag2idx.keys()]

y_true = list(chain.from_iterable(y_true))
y_pred = list(chain.from_iterable(y_pred))

# 打印详细分数报告，包括precision(精确率)，recall(召回率)，f1-score(f1分数)，support(个数)，digits=3代表保留3位小数
print(metrics.classification_report(
    y_true, y_pred, labels=sort_labels[1:]
))

# 样例测试
model.predict('张三是我的好朋友。')
print("\n")
model.predict('我在哈尔滨工业大学上学。')
print("\n")
model.predict('“怎么会有视觉中国这么不要脸的媒体啊。”周志辰在微博上骂了起来。')
print("\n")
model.predict('张三在北京的微软公司工作。')
print("\n")
model.predict('马云在杭州创办了阿里巴巴。')

