import re


# 初始化参数
fp = 'data/train.txt'  # 填写数据集路径
op = 'data/train_process.txt'  # 填写输出路径
# 上述路径修改为test.txt和test_process.txt即可处理测试集

# 读取数据集
def read_data(filepath):
    print('正在从', filepath, '读取数据...')
    with open(filepath, 'r', encoding='GBK') as f:
        data = f.read()
    data = data.split('\n')
    return data


# 单行语料处理
def pre_process(text):
    if not text:
        return []
    valid_content = text.split(' ', 1)[1]
    # 使用正则表达式分割文本
    words = re.findall(r'\[[^]]*]\w+|\b\w+/\w+|./w', valid_content)
    # 初始化BIO标注列表
    bio_tags = []
    for word in words:
        # 检查是否是嵌套实体
        if word.startswith('[') and (word[-3] == ']' or word[-2] == ']'):
            # 依据尾字符标记嵌套实体类型
            word_end = word[-1]
            # 去除括号
            word = word[1:-3]
            # 分割单词和词性标注
            nested_words = re.findall(r'\b\w+/\w+|./w', word)
            # 去除嵌套实体内原始词性标注
            nested_single_words = []
            for nested_word in nested_words:
                nested_single_words.append(nested_word.split('/')[0])
            # 嵌套实体拼接 并生成BIO标注
            nested_word = ''.join(nested_single_words)
            if word_end == 'r':
                bio_tags.append((nested_word[0], 'B_PER'))
                for c in nested_word[1:]:
                    bio_tags.append((c, 'I-PER'))
            elif word_end == 's':
                bio_tags.append((nested_word[0], 'B-LOC'))
                for c in nested_word[1:]:
                    bio_tags.append((c, 'I-LOC'))
            else:
                bio_tags.append((nested_word[0], 'B-ORG'))
                for c in nested_word[1:]:
                    bio_tags.append((c, 'I-ORG'))
        else:
            # 分割单词和词性标注
            # print(word.split('/'))
            word, tag = word.split('/')
            # 根据词性标注生成BIO标注
            if tag in ['nr', 'ns', 'nt']:
                if tag == 'nr':
                    bio_tags.append((word[0], 'B-PER'))
                    for c in word[1:]:
                        bio_tags.append((c, 'I-PER'))
                elif tag == 'ns':
                    bio_tags.append((word[0], 'B-LOC'))
                    for c in word[1:]:
                        bio_tags.append((c, 'I-LOC'))
                else:
                    bio_tags.append((word[0], 'B-ORG'))
                    for c in word[1:]:
                        bio_tags.append((c, 'I-ORG'))
            else:
                for c in word:
                    bio_tags.append((c, 'O'))
    return bio_tags


def BIO_merge(BIO_origin_list):
    BIO_list_new = []
    print("开始合并命名实体中重复的B标签")
    for i in range(len(BIO_origin_list)-1, 1, -1):
        if BIO_origin_list[i][1][0] == 'B' and BIO_origin_list[i-1][1][0] == 'B':
            BIO_list_new.append((BIO_origin_list[i][0], 'I'+BIO_origin_list[i][1][1:]))
        else:
            BIO_list_new.append(BIO_origin_list[i])
    BIO_list_new.reverse()
    print("合并完成")
    return BIO_list_new


def output_data(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        for c in data:
            if c[0] == '':
                f.write('\n')
            else:
                f.write(c[0] + '  ' + c[1] + '\n')
    print('数据已输出至', output_path)


text = read_data(fp)
BIO_list = []
for line in text:
    for char in pre_process(line):
        BIO_list.append(char)
    BIO_list.append(('', 'O'))
    # 输出一行空行
BIO_list = BIO_merge(BIO_list)
print(BIO_list)
output_data(op, BIO_list)