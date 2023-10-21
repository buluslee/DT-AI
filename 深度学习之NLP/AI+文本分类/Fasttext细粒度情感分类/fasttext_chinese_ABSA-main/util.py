#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

stop_words = []


def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def seg_words(contents):
    contents_segs = list()
    for i, content in enumerate(contents):
        print(i + 1, len(contents))
        rcontent = content.replace("\r\n", " ").replace("\n", " ")
        segs = [word for word in jieba.cut(rcontent) if word not in stop_words]
        contents_segs.append(" ".join(segs))
    return contents_segs


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[1, 0, -1, -2], average='macro')


def get_report(y_true, y_pred):
    return classification_report(y_true, y_pred, labels=[1, 0, -1, -2])
