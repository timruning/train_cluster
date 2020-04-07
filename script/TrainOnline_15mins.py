# coding = utf-8

import collections
import argparse
import sys

import time

import numpy as np
import tensorflow as tf
from sklearn import metrics

from script import Console
from script import LoadData
from script import NFM_15min

FLAGS = None

def train(_):
    #path params
    data_path=FLAGS.path
    model_path=FLAGS.model_path
    online_model_path=FLAGS.online_model_path
    log_path=FLAGS.summaries_dir

    #model params
    batch_size = 1
    loss_type = "square_loss"
    layers = [64]
    hidden_factor = 64
    # optimizer_type = 'AdamOptimizer'
    # optimizer_type = 'AdagradOptimizer'
    optimizer_type = 'lazyAdamOptimizer'
    # optimizer_type = 'GradientDescentOptimizer'
    lambda_bilinear = 1
    keep_prob=[0.8,0.5]
    random_seed = 2017
    batch_norm = 1
    lamda_bilinear = 0
    learning_rate = 0.0001

    feature_path=data_path+"/feature_set"

    features_M = LoadData.getFeatureSize(features_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--path",
        type=str,
        default="../data/",
        help='Input data path.'
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../model/test.cptk",
        help='Input data path.'
    )
    parser.add_argument(
        "--online_model_path",
        type=str,
        default="../related_model",
        help='Input data path.'
    )
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default='../log/first',
        help='log out'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)