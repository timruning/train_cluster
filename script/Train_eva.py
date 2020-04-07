# coding=utf-8
import argparse
import collections
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn import metrics

from script import LoadData
from script import NFM_data

FLAGS = None


def decodeCSV(line):
    defaults = collections.OrderedDict([
        ('label', [-1]),
        ('uid', ['abc']),
        ('newsid', ['123']),
        ('groupid', ['123'])
    ])
    for i in range(50):
        defaults['key%d' % i] = [0]

    items = tf.decode_csv(line, list(defaults.values()))
    pairs = zip(defaults.keys(), items)
    features_dic = dict(pairs)
    label = tf.cast([features_dic.pop('label')], dtype=tf.float32)
    print(label)
    uid = features_dic.pop('uid')
    newsid = features_dic.pop('newsid')
    groupid = features_dic.pop('groupid')
    features = []
    for key in features_dic.keys():
        print(features_dic[key])
        features.append(features_dic[key])
    # features = np.array(features, dtype=np.uint32)
    return features, label


def getDirFiles(path):
    result = []
    if not os.path.exists(path):
        return result
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            result.append(dirpath + '/' + file)
    return result


def readBatchs(train_files, batch_size=200, random_crop=False, random_clip=False, shuffle_batch=True):
    file_name_queue = tf.train.string_input_producer(train_files)
    reader = tf.TextLineReader()
    key, value = reader.read(file_name_queue)

    defaults = collections.OrderedDict([
        ('label', [-1]),
        ('uid', ['abc']),
        ('newsid', ['123']),
        ('groupid', ['123'])
    ])
    for i in range(50):
        defaults['key%d' % i] = [0]
    items = tf.decode_csv(value, list(defaults.values()))
    pairs = zip(defaults.keys(), items)
    features_dic = dict(pairs)
    label = tf.cast([features_dic.pop('label')], dtype=tf.float32)
    print(label)
    uid = features_dic.pop('uid')
    newsid = features_dic.pop('newsid')
    groupid = features_dic.pop('groupid')
    feature = []
    for key in features_dic.keys():
        print(features_dic[key])
        feature.append(features_dic[key])
    features = tf.cast(feature, dtype=tf.int32)
    if shuffle_batch:
        labels, features = tf.train.shuffle_batch([label, feature], batch_size=batch_size, capacity=80000,
                                                  num_threads=8, min_after_dequeue=2000)
    else:
        labels, features = tf.train.batch([label, feature], batch_size=batch_size, capacity=80000, num_threads=4)
    return labels, features


def train(_):
    log_path = FLAGS.summaries_dir
    print(log_path)
    # 控制一条样本实际特征数量
    FEATURE_LENGTH = 50
    # Create a cluster from the parameter server and worker hosts.
    path = FLAGS.path
    model_path = FLAGS.model_path
    # 数据地址
    train_path = path + "train"
    train_files = getDirFiles(train_path)
    print(train_files)

    features_path = path + "feature_set"

    # loss_type = "log_loss"
    batch_size = 1
    loss_type = "square_loss"
    layers = [64]
    hidden_factor = 64
    # optimizer_type = 'AdamOptimizer'
    optimizer_type = 'AdagradOptimizer'
    lambda_bilinear = 1

    print(train_path)
    # 在worker中加载数据

    train_y, train_x = readBatchs(train_files=train_files, batch_size=batch_size, shuffle_batch=False)

    features_M = LoadData.getFeatureSize(features_path)

    print("#################### abc")
    nfm = NFM_data.NMF_Net(train_features=train_x, train_labels=train_y, batch_size=batch_size,
                           hidden_factor=hidden_factor, layers=layers, loss_type=loss_type,
                           features_M=features_M, random_seed=2017, batch_norm=1,
                           lamda_bilinear=0, optimizer_type=optimizer_type, learning_rate=0.05,
                           lambda_bilinear=lambda_bilinear, keep_prob=[0.8, 0.5])
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    with tf.Session() as session:

        train_writer = tf.summary.FileWriter(log_path + "/train", session.graph)

        session.run(init_op)
        saver.restore(sess=session, save_path=model_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        max_iter = 400
        iter = 0
        print("#### nfm.optimizer\t", nfm.optimizer)

        labels = None
        outs = None
        auc = 0

        try:
            i = 0
            while i < 100:
                # feed_dic get 这里有点问题，回头改
                feed_dict = {nfm.dropout_keep: nfm.keep_prob,
                             nfm.train_phase: True}
                _nfm_out, _train_x, _train_y = session.run(
                    [nfm.out, train_x, train_y], feed_dict=feed_dict)
                if labels is None:
                    labels = _train_y
                else:
                    labels = np.concatenate((labels, _train_y), axis=0)

                if outs is None:
                    outs = _nfm_out
                else:
                    outs = np.concatenate((outs, _nfm_out), axis=0)
                i += 1
                print(i)
        except:
            auc = metrics.roc_auc_score(y_score=outs, y_true=labels)
            print(auc)
            print("a epoch end!!")
        auc = metrics.roc_auc_score(y_score=outs, y_true=labels)
        print(auc)

        coord.request_stop()  # queue需要关闭，否则报错 
        coord.join(threads)


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
        "--summaries_dir",
        type=str,
        default='../log/first',
        help='log out'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
