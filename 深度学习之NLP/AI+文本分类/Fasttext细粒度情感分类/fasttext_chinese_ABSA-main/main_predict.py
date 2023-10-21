#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import config
import logging

import numpy as np

import joblib
from util import load_data_from_csv, seg_words


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, nargs='?',
                        default='fasttext_model.pkl',
                        help='the name of model')

    args = parser.parse_args()
    model_name = args.model_name

    # load data
    logger.info("start load load")
    test_data_df = load_data_from_csv(config.test_data_path)

    # load model
    logger.info("start load model")
    classifier_dict = joblib.load(config.model_path + model_name)

    content_test = test_data_df['review']
    logger.info("start seg train data")
    content_test = seg_words(content_test)
    logger.info("complete seg train data")

    logger.info("prepare predict data format")
    test_data_format = np.asarray([content_test]).T
    logger.info("complete prepare predict formate data")

    columns = test_data_df.columns.values.tolist()

    # model predict
    logger.info("start predict test data")
    for column in columns[2:]:
        test_data_df[column] = classifier_dict[column].predict(
                test_data_format).astype(int)
        logger.info("complete %s predict" % column)

    test_data_df.to_csv(config.test_data_predict_output_path,
                        encoding="utf-8", index=False)
    logger.info("complete predict test data")
