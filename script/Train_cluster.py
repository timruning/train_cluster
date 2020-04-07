# coding=utf-8
import argparse
import sys
import time

import Console
import LoadData
import NFM_feeddic
import numpy as np
import tensorflow as tf
from sklearn import metrics

FLAGS = None


def train_cluster(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    log_path = FLAGS.summaries_dir

    print(log_path)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            path = FLAGS.path

            # 数据地址
            train_path = path + "train"

            valid_path = path + "valid"

            test_path = path + "test"

            features_path = path + "feature_set"

            print(train_path)
            print(valid_path)
            print(test_path)
            # 在worker中加载数据
            TRAIN_DATA = LoadData.loadData(train_path)
            VALID_DATA = LoadData.loadData(valid_path)
            TEST_DATA = LoadData.loadData(test_path)
            features_M = LoadData.getFeatureSize(features_path)

            # loss_type = "log_loss"
            batch_size = 100
            loss_type = "square_loss"
            layers = [64]
            hidden_factor = 64
            optimizer_type = 'AdamOptimizer'
            lambda_bilinear = 1
            # global_step = tf.contrib.framework.get_or_create_global_step()
            global_step = tf.train.get_or_create_global_step()
            print("#################### abc")
            nfm = NFM_feeddic.NMF_Net(hidden_factor=hidden_factor, layers=layers, loss_type=loss_type,
                                      features_M=features_M, random_seed=2017, batch_norm=1,
                                      lamda_bilinear=0, optimizer_type=optimizer_type, learning_rate=0.05,
                                      lambda_bilinear=lambda_bilinear, keep_prob=[0.8, 0.5], global_step=global_step)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            # saver = tf.train.Saver()
            merged = tf.summary.merge_all()

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="../check_point/train_logs") as session:

            train_writer = tf.summary.FileWriter(log_path + "/train")
            session.run(init_op)
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord)
            max_iter = 400
            iter = 0
            print("#### nfm.optimizer\t", nfm.optimizer)

            while iter < max_iter:
                t1 = time.time()
                Console.shuffle_in_unison_scary(TRAIN_DATA['Y'], TRAIN_DATA['X'])
                total_train_batch_num = int(len(TRAIN_DATA['Y']) / batch_size)
                total_valid_batch_num = int(len(VALID_DATA['Y']) / batch_size)
                total_test_batch_num = int(len(TEST_DATA['Y']) / batch_size)
                print("---------------------------------\t", total_train_batch_num)
                print('---------------------------------\t', total_valid_batch_num)
                print('---------------------------------\t', total_test_batch_num)

                # train
                for i in range(total_train_batch_num):
                    batch_xs = LoadData.shuffle_batch(TRAIN_DATA, batch_size=batch_size)
                    # feed_dic get 这里有点问题，回头改
                    feed_dict = {nfm.train_features: batch_xs['X'], nfm.train_labels: batch_xs['Y'],
                                 nfm.dropout_keep: nfm.keep_prob,
                                 nfm.train_phase: True}

                    if i % 100 == 99:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _nfm_loss, _nfm_optimizer, _merged = session.run(
                            [nfm.loss, nfm.optimizer, merged], feed_dict=feed_dict, options=run_options,
                            run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'epoch%03d' % iter + 'step%03d' % i)
                        train_writer.add_summary(_merged, i)
                    else:
                        _nfm_loss, _nfm_optimizer = session.run(
                            [nfm.loss, nfm.optimizer], feed_dict=feed_dict)

                # eva
                eva_label = None
                eva_out = None
                for i in range(total_train_batch_num):
                    begin = i * batch_size
                    batch_xs = LoadData.batch(TRAIN_DATA, batch_size=batch_size, begin=begin)

                    # feed_dic get 这里有点问题，回头改
                    feed_dict = {nfm.train_features: batch_xs['X'], nfm.train_labels: batch_xs['Y'],
                                 nfm.dropout_keep: nfm.keep_prob,
                                 nfm.train_phase: True}

                    _nfm_out = session.run(
                        [nfm.out], feed_dict=feed_dict)
                    if eva_out is None:
                        eva_out = _nfm_out[0]
                    else:
                        eva_out = np.concatenate((eva_out, _nfm_out[0]), axis=0)
                    if eva_label is None:
                        eva_label = batch_xs['Y']
                    else:
                        eva_label = np.concatenate((eva_label, batch_xs['Y']), axis=0)

                eva_auc = metrics.roc_auc_score(y_true=eva_label, y_score=eva_out)

                # valid
                valid_label = None
                valid_out = None
                for i in range(total_valid_batch_num):
                    begin = i * batch_size
                    batch_xs = LoadData.batch(VALID_DATA, batch_size=batch_size, begin=begin)

                    # feed_dic get 这里有点问题，回头改
                    feed_dict = {nfm.train_features: batch_xs['X'], nfm.train_labels: batch_xs['Y'],
                                 nfm.dropout_keep: nfm.keep_prob,
                                 nfm.train_phase: True}

                    _nfm_out = session.run(
                        [nfm.out], feed_dict=feed_dict)
                    # valid_writer.add_summary(_merged, i)
                    if valid_out is None:
                        valid_out = _nfm_out[0]
                    else:
                        valid_out = np.concatenate((valid_out, _nfm_out[0]), axis=0)
                    if valid_label is None:
                        valid_label = batch_xs['Y']
                    else:
                        valid_label = np.concatenate((valid_label, batch_xs['Y']), axis=0)

                valid_auc = metrics.roc_auc_score(y_true=valid_label, y_score=valid_out)

                # test
                test_label = None
                test_out = None
                for i in range(total_test_batch_num):
                    begin = i * batch_size
                    batch_xs = LoadData.batch(TEST_DATA, batch_size=batch_size, begin=begin)

                    # feed_dic get 这里有点问题，回头改
                    feed_dict = {nfm.train_features: batch_xs['X'], nfm.train_labels: batch_xs['Y'],
                                 nfm.dropout_keep: nfm.keep_prob,
                                 nfm.train_phase: True}

                    _nfm_out = session.run(
                        [nfm.out], feed_dict=feed_dict)
                    if test_out is None:
                        test_out = _nfm_out[0]
                    else:
                        test_out = np.concatenate((test_out, _nfm_out[0]), axis=0)
                    if test_label is None:
                        test_label = batch_xs['Y']
                    else:
                        test_label = np.concatenate((test_label, batch_xs['Y']), axis=0)
                test_auc = metrics.roc_auc_score(y_true=test_label, y_score=test_out)
                t2 = time.time() - t1
                print("epoch=", iter, "\teva_auc=", eva_auc, "\tvalid_auc=", valid_auc, "\ttest_auc=", test_auc,
                      "\ttime=", t2)
                iter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="localhost:2222",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="localhost:2223,localhost:2224",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="worker",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="../data/",
        help='Input data path.'
    )
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default='../log/first',
        help='log out'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=train_cluster, argv=[sys.argv[0]] + unparsed)
