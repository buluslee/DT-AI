#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import config
import logging
import os

import numpy as np

from skift_core import FirstColFtClassifier
import joblib
from util import load_data_from_csv, seg_words, get_f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, nargs='?',
                        default='fasttext_model.pkl',
                        help='the name of model')
    parser.add_argument('-lr', '--learning_rate', type=float, nargs='?',
                        default=1.0)
    parser.add_argument('-ep', '--epoch', type=int, nargs='?',
                        default=10)
    parser.add_argument('-wn', '--word_ngrams', type=int, nargs='?',
                        default=1)
    parser.add_argument('-mc', '--min_count', type=int, nargs='?',
                        default=1)

    args = parser.parse_args()
    model_name = args.model_name
    learning_rate = args.learning_rate
    epoch = args.epoch
    word_ngrams = args.word_ngrams
    min_count = args.min_count

    # load train data
    logger.info("start load load")
    train_data_df = load_data_from_csv(config.train_data_path)
    validate_data_df = load_data_from_csv(config.validate_data_path)

    content_train = train_data_df.iloc[:, 1]

    logger.info("start seg train data")
    content_train = seg_words(content_train)
    logger.info("complete seg train data")

    logger.info("prepare train format")
    train_data_format = np.asarray([content_train]).T
    logger.info("complete formate train data")

    columns = train_data_df.columns.values.tolist()

    # model train
    logger.info("start train model")
    classifier_dict = dict()
    for column in columns[2:]:
        train_label = train_data_df[column]
        logger.info("start train %s model" % column)
        sk_clf = FirstColFtClassifier(lr=learning_rate, epoch=epoch,
                                      wordNgrams=word_ngrams,
                                      minCount=min_count, verbose=2)
        sk_clf.fit(train_data_format, train_label)
        logger.info("complete train %s model" % column)
        classifier_dict[column] = sk_clf

    logger.info("complete train model")
    logger.info("start save model")
    model_path = config.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    joblib.dump(classifier_dict, model_path + model_name)
    logger.info("complete save model")

    # validata model
    content_validata = validate_data_df.iloc[:, 1]

    logger.info("start seg validata data")
    content_validata = seg_words(content_validata)
    logger.info("complet seg validata data")

    logger.info("prepare valid format")
    validata_data_format = np.asarray([content_validata]).T
    logger.info("complete formate train data")

    logger.info("start compute f1 score for validata model")
    f1_score_dict = dict()
    for column in columns[2:]:
        true_label = np.asarray(validate_data_df[column])
        classifier = classifier_dict[column]
        pred_label = classifier.predict(validata_data_format).astype(int)
        f1_score = get_f1_score(true_label, pred_label)
        f1_score_dict[column] = f1_score

    f1_score = np.mean(list(f1_score_dict.values()))
    str_score = "\n"
    for column in columns[2:]:
        str_score += column + ":" + str(f1_score_dict[column]) + "\n"

    logger.info("f1_scores: %s\n" % str_score)
    logger.info("f1_score: %s" % f1_score)
    logger.info("complete compute f1 score for validate model")
