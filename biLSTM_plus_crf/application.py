import time
import datetime
from itertools import chain

import torch
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from biLSTM_plus_crf.data_processor import Model_Dataset, get_vocab, get_label_map
from biLSTM_plus_crf.model import BiLSTM_CRF

# 设置torch随机种子
torch.manual_seed(4396)

embedding_size = 32  # 词向量维度
hidden_dim = 128  # 隐层维度
epochs = 10  # 训练周期
batch_size = 512  # 批次大小
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# 建立中文词表，扫描训练集所有字符得到，'PAD'在batch填充时使用，'UNK'用于替换字表以外的新字符
vocab = get_vocab('../data/train_process.txt')
# 建立标签字典，扫描训练集所有字符得到
label_map = get_label_map('../data/train_process.txt')

train_dataset = Model_Dataset('../data/train_process.txt', vocab, label_map)
valid_dataset = Model_Dataset('../data/test_process.txt', vocab, label_map)
print("====================================")
print('训练集长度:', len(train_dataset))
print('验证集长度:', len(valid_dataset))
print("====================================")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True,
                              collate_fn=train_dataset.collect_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False,
                              collate_fn=valid_dataset.collect_fn)
model = BiLSTM_CRF(embedding_size, hidden_dim, train_dataset.vocab, train_dataset.label_map, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def evaluate():
    all_label = []
    all_pred = []
    model.eval()
    model.state = 'eval'
    with torch.no_grad():
        for text, label, seq_len in tqdm(valid_dataloader, desc='eval: '):
            text = text.to(device)
            seq_len = seq_len.to(device)
            batch_tag = model(text, seq_len, label)
            all_label.extend([[train_dataset.label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)])
            all_pred.extend([[train_dataset.label_map_inv[t] for t in l] for l in batch_tag])

    all_label = list(chain.from_iterable(all_label))
    all_pred = list(chain.from_iterable(all_pred))
    sort_labels = [k for k in train_dataset.label_map.keys()]
    # 使用sklearn库得到F1分数
    f1 = metrics.f1_score(all_label, all_pred, average='macro', labels=sort_labels[:-3])

    print(metrics.classification_report(
        all_label, all_pred, labels=sort_labels[:-3], digits=3
    ))
    return f1


def train():
    total_start = time.time()
    best_score = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        model.state = 'train'
        for step, (text, label, seq_len) in enumerate(train_dataloader, start=1):
            start = time.time()
            text = text.to(device)
            label = label.to(device)
            seq_len = seq_len.to(device)

            loss = model(text, seq_len, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("====================================")
            print('Epoch: [', (epoch + 1)/epochs, ']')
            print('cur_epoch_finished:', step * batch_size / len(train_dataset) * 100, '%')
            print('loss:', loss.item())
            print('cur_step_time:', time.time() - start)
            print('cur_epoch_remaining_time:', datetime.timedelta(seconds=int((len(train_dataloader) - step) / step * (time.time() - epoch_start))))
            print('total_remaining_time:', datetime.timedelta(seconds=int((len(train_dataloader) * epochs - (len(train_dataloader) * epoch + step)) / (len(train_dataloader) * epoch + step) * (time.time() - total_start))))
            print("====================================")

        # 每周期验证一次，保存最优参数
        score = evaluate()
        if score > best_score:
            print("===============!!!!=================")
            print('score increase:', best_score, '->', score)
            print("====================================")
            best_score = score
            torch.save(model.state_dict(), './model.bin')
        print('current best score:', best_score)




train()
