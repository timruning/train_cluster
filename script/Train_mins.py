# coding=utf-8
import argparse
import collections
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn import metrics

from script import Console
from script import LoadData
from script import NFM_15min

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
    train_files = Console.getDirFiles(train_path)
    features_path = path + "feature_set"

    # loss_type = "log_loss"
    batch_size = 2
    loss_type = "square_loss"
    layers = [64]
    hidden_factor = 64
    # optimizer_type = 'AdamOptimizer'
    # optimizer_type
    # 'AdagradOptimizer'
    optimizer_type = 'lazyAdamOptimizer'
    # optimizer_type = 'GradientDescentOptimizer'
    lambda_bilinear = 1

    print(train_path)
    # 在worker中加载数据
    random_seed = time.time()
    features_M = LoadData.getFeatureSize(features_path)

    nfm = NFM_15min.NMF_Net(batch_size=batch_size, hidden_factor=hidden_factor, layers=layers, loss_type=loss_type,
                            features_M=features_M, random_seed=random_seed, batch_norm=1,
                            lamda_bilinear=0, optimizer_type=optimizer_type, learning_rate=0.0001,
                            lambda_bilinear=lambda_bilinear, keep_prob=[0.8, 0.5])

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    saver = tf.train.Saver(nfm.params)
    merged = tf.summary.merge_all()
    builder=tf.saved_model.builder.SavedModelBuilder("model3")
    with tf.Session() as session:
        train_writer = tf.summary.FileWriter(log_path + "/train", session.graph)
        session.run(init_op)
        # saver.restore(sess=session, save_path=model_path)
        for file_num in range(1, len(train_files)):
            t1 = time.time()

            train_file = train_files[file_num - 1]
            test_file = train_files[file_num]

            TRAIN_DATA = LoadData.loadData_csv(train_file)
            TEST_DATA = LoadData.loadData_csv(test_file)

            total_train_batch_num = int(len(TRAIN_DATA['Y']) / batch_size)
            total_test_batch_num = int(len(TEST_DATA['Y']) / batch_size)

            train
            for i in range(total_train_batch_num):
                begin = i * batch_size
                batch_xs = LoadData.batch(TRAIN_DATA, batch_size=batch_size, begin=begin)
                feed_dict = {nfm.train_features: batch_xs['X'], nfm.train_labels: batch_xs['Y'],
                             nfm.dropout_keep: nfm.keep_prob,
                             nfm.train_phase: True}
                if i % 100 == 99:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _nfm_loss, _nfm_optimizer, _merged = session.run(
                        [nfm.loss, nfm.optimizer, merged], feed_dict=feed_dict, options=run_options,
                        run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, train_files[file_num] + '#step%03d' % i)
                    train_writer.add_summary(_merged, i)
                else:
                    _nfm_loss, _nfm_optimizer = session.run(
                        [nfm.loss, nfm.optimizer], feed_dict=feed_dict)
                    print(_nfm_loss)
            builder.add_meta_graph_and_variables(session,["serving"])
            builder.save()
            # eva
            eva_label = None
            eva_out = None
            for i in range(2):
                begin = i * batch_size
                batch_xs = LoadData.batch(TRAIN_DATA, batch_size=batch_size, begin=begin)

                # feed_dic get 这里有点问题，回头改
                feed_dict = {nfm.train_features: batch_xs['X'], nfm.train_labels: batch_xs['Y'],
                             nfm.dropout_keep: nfm.no_dropout,
                             nfm.train_phase: True}

                _nfm_out = session.run(
                    [nfm.out], feed_dict=feed_dict)
                print(_nfm_out)
                if eva_out is None:
                    eva_out = _nfm_out[0]
                else:
                    eva_out = np.concatenate((eva_out, _nfm_out[0]), axis=0)
                if eva_label is None:
                    eva_label = batch_xs['Y']
                else:
                    eva_label = np.concatenate((eva_label, batch_xs['Y']), axis=0)
            print(np.shape(eva_label))
            print(np.shape(eva_out))
            eva_auc = metrics.roc_auc_score(y_true=eva_label, y_score=eva_out)

            # test
            # test_label = None
            # test_out = None
            # for i in range(total_test_batch_num):
            #     begin = i * batch_size
            #     batch_xs = LoadData.batch(TEST_DATA, batch_size=batch_size, begin=begin)
            #
            #     # feed_dic get 这里有点问题，回头改
            #     feed_dict = {nfm.train_features: batch_xs['X'], nfm.train_labels: batch_xs['Y'],
            #                  nfm.dropout_keep: nfm.keep_prob,
            #                  nfm.train_phase: True}
            #
            #     _nfm_out = session.run(
            #         [nfm.out], feed_dict=feed_dict)
            #     if test_out is None:
            #         test_out = _nfm_out[0]
            #     else:
            #         test_out = np.concatenate((test_out, _nfm_out[0]), axis=0)
            #     if test_label is None:
            #         test_label = batch_xs['Y']
            #     else:
            #         test_label = np.concatenate((test_label, batch_xs['Y']), axis=0)
            # test_auc = metrics.roc_auc_score(y_true=test_label, y_score=test_out)
            # t2 = time.time() - t1
            # print("file_num=", train_files[file_num], "\teva_auc=", eva_auc, "\ttest_auc=", test_auc, "\ttime=", t2)
            # gd = session.graph.as_graph_def()
            #
            # for node in gd.node:
            #     # name1=node.op
            #     # print(name1)
            #     if node.op == 'bn_fm':
            #         node.op = 'bn_fm'
            #         for index in range(len(node.input)):
            #             if 'moving_' in node.input[index]:
            #                 node.input[index] = node.input[index] + '/read'
            #     elif node.op == 'AssignSub':
            #         node.op = 'Sub'
            #         if 'use_locking' in node.attr: del node.attr['use_locking']
            #
            # graph2 = tf.graph_util.convert_variables_to_constants(sess=session,
            #                                                       input_graph_def=gd,
            #                                                       output_node_names=["input_train_feature",
            #                                                                          "input_dropout",
            #                                                                          "input_phase",
            #                                                                          "output"])
            # tf.train.write_graph(graph2, './log', 'graph.pb', as_text=False)
            # saver.save(session, save_path="/opt/develop/workspace/sohu/NFM/Train_Cluster/script/log/test/hello.cptk")


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
