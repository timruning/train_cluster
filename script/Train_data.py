# coding=utf-8
import argparse
import collections
import os
import sys
import time

import tensorflow as tf

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

    features_path = path + "feature_set"

    # loss_type = "log_loss"
    batch_size = 100
    loss_type = "square_loss"
    layers = [64]
    hidden_factor = 64
    optimizer_type = 'AdamOptimizer'
    lambda_bilinear = 1

    print(train_path)
    # 在worker中加载数据
    TRAIN_DATA = tf.data.TextLineDataset(train_files)
    TRAIN_DATA = TRAIN_DATA.shuffle(buffer_size=100000, seed=2017)
    TRAIN_DATA = TRAIN_DATA.map(decodeCSV)
    TRAIN_DATA = TRAIN_DATA.batch(batch_size=batch_size)

    train_iter = TRAIN_DATA.make_initializable_iterator()
    train_x, train_y = train_iter.get_next()

    features_M = LoadData.getFeatureSize(features_path)

    # global_step = tf.contrib.framework.get_or_create_global_step()

    # global_step = tf.train.get_or_create_global_step()

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
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        max_iter = 400
        iter = 0
        print("#### nfm.optimizer\t", nfm.optimizer)

        while iter < max_iter:
            session.run(train_iter.initializer)
            t1 = time.time()

            # train

            try:
                i = 0
                while True:
                    # feed_dic get 这里有点问题，回头改
                    feed_dict = {nfm.dropout_keep: nfm.keep_prob,
                                 nfm.train_phase: True}

                    if i % 100 == 99:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _nfm_loss, _nfm_optimizer, _merged, _train_x, _train_y = session.run(
                            [nfm.loss, nfm.optimizer, merged, train_x, train_y], feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'epoch%03d' % iter + '_step%03d' % i)
                        train_writer.add_summary(_merged, i)
                        t2 = time.time() - t1
                        print(t2)
                        t1 = time.time()
                    else:
                        _nfm_loss, _nfm_optimizer, _train_x, _train_y = session.run(
                            [nfm.loss, nfm.optimizer, train_x, train_y], feed_dict=feed_dict)
                    i += 1
            except:
                print("a epoch end!!")

            # # eva
            # eva_label = None
            # eva_out = None
            # for i in range(total_train_batch_num):
            #     begin = i * batch_size
            #     batch_xs = LoadData.batch(TRAIN_DATA, batch_size=batch_size, begin=begin)
            #
            #     # feed_dic get 这里有点问题，回头改
            #     feed_dict = {nfm.train_features: batch_xs['X'], nfm.train_labels: batch_xs['Y'],
            #                  nfm.dropout_keep: nfm.keep_prob,
            #                  nfm.train_phase: True}
            #
            #     _nfm_out = session.run(
            #         [nfm.out], feed_dict=feed_dict)
            #     if eva_out is None:
            #         eva_out = _nfm_out[0]
            #     else:
            #         eva_out = np.concatenate((eva_out, _nfm_out[0]), axis=0)
            #     if eva_label is None:
            #         eva_label = batch_xs['Y']
            #     else:
            #         eva_label = np.concatenate((eva_label, batch_xs['Y']), axis=0)
            # print(np.shape(eva_label))
            # print(np.shape(eva_out))
            # eva_auc = metrics.roc_auc_score(y_true=eva_label, y_score=eva_out)
            #
            # # valid
            # valid_label = None
            # valid_out = None
            # for i in range(total_valid_batch_num):
            #     begin = i * batch_size
            #     batch_xs = LoadData.batch(VALID_DATA, batch_size=batch_size, begin=begin)
            #
            #     # feed_dic get 这里有点问题，回头改
            #     feed_dict = {nfm.train_features: batch_xs['X'], nfm.train_labels: batch_xs['Y'],
            #                  nfm.dropout_keep: nfm.keep_prob,
            #                  nfm.train_phase: True}
            #
            #     _nfm_out = session.run(
            #         [nfm.out], feed_dict=feed_dict)
            #     # valid_writer.add_summary(_merged, i)
            #     if valid_out is None:
            #         valid_out = _nfm_out[0]
            #     else:
            #         valid_out = np.concatenate((valid_out, _nfm_out[0]), axis=0)
            #     if valid_label is None:
            #         valid_label = batch_xs['Y']
            #     else:
            #         valid_label = np.concatenate((valid_label, batch_xs['Y']), axis=0)
            #
            # valid_auc = metrics.roc_auc_score(y_true=valid_label, y_score=valid_out)
            #
            # # test
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
            # print("epoch=", iter, "\teva_auc=", eva_auc, "\tvalid_auc=", valid_auc, "\ttest_auc=", test_auc,
            #       "\ttime=", t2)
            iter += 1
            saver.save(session, save_path=model_path)


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
        default="../model/",
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
