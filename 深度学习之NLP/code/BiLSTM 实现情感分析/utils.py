"""
:author: Qizhi Li
"""
import os


def read_data(file_path):
    """
    读取数据
    :param file_path: str
            文件路径
    :return texts: list
            文本列表
    :return labels: list
            标签列表
    """
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        clean_line = line.rstrip('\n').split('\t')
        texts.append(clean_line[0])
        labels.append(clean_line[1])

    return texts, labels