#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import config
import logging

import numpy as np

import joblib
from util import load_data_from_csv, seg_words, get_report


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

logger.info("start compute report for validata model")
train_data_df = load_data_from_csv(config.train_data_path)
validate_data_df = load_data_from_csv(config.validate_data_path)
columns = train_data_df.columns.values.tolist()
# validata model
content_validata = validate_data_df.iloc[:, 1]

logger.info("start seg validata data")
content_validata = seg_words(content_validata)
logger.info("complet seg validata data")

logger.info("prepare valid format")
validata_data_format = np.asarray([content_validata]).T
logger.info("complete formate train data")

model_path = config.model_path
model_name = "fasttext_model.pkl"

classifier_dict = joblib.load(model_path + model_name)
logger.info("complete load model")

logger.info("start compute report for validata model")
f1_score_dict = dict()
for column in columns[2:]:
    true_label = np.asarray(validate_data_df[column])
    classifier = classifier_dict[column]
    pred_label = classifier.predict(validata_data_format).astype(int)
    report = get_report(true_label, pred_label)
    print(report)