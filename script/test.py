import collections

import numpy as np
import tensorflow as tf

from script import LoadData


# a = [1, 2, 3, 661373]
# b = numpy.array(a, dtype=numpy.uint32)
# print(sys.getsizeof(b))
def testLoadData():
    path = "/opt/develop/workspace/sohu/NFM/Train_Cluster/data/xaa"
    LoadData.loadData(path, 80)


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
    label = features_dic.pop('label')
    uid = features_dic.pop('uid')
    newsid = features_dic.pop('newsid')
    groupid = features_dic.pop('groupid')
    features = []
    for key in features_dic.keys():
        print(features_dic[key])
        features.append(features_dic[key])
    # features = np.array(features, dtype=np.uint32)
    return features, label


def testLoadDataTensor():
    files = ['../data/onehot.csv', '../data/xaa.csv']
    dataset = tf.data.TextLineDataset(files).map(decodeCSV)
    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.batch(batch_size=100)
    iter = dataset.make_initializable_iterator()
    x, y = iter.get_next()
    print(x)
    with tf.Session() as sess:
        sess.run(iter.initializer)
        index = 0
        while True:
            _x, _y = sess.run([x, y])
            print(_x)
            print(_y)
            index += 1


if __name__ == '__main__':
    # testLoadData()
    # testLoadDataTensor()
    path = "/opt/develop/workspace/sohu/NFM/Train_Cluster/data/train/onehot.csv"
    TRAIN_DATA = LoadData.loadData_csv(path)
    with tf.Session() as sess:
        with open('./log/graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            for i in range(len(TRAIN_DATA['X'])):
                input_train_feature = []
                for v in TRAIN_DATA['X'][i]:
                    input_train_feature.append(int(v))
                input_train_label=[]
                for v in TRAIN_DATA['Y'][i]:
                    input_train_label.append(float(v))
                input_map = {
                    'input_train_feature:0': [input_train_feature],
                    "input_dropout": [0.8, 0.5],
                    "input_phase": True,
                    "input_train_label:0": [input_train_label]
                }
                output = tf.import_graph_def(graph_def, input_map=input_map,
                                             return_elements=['output:0'])
                t = sess.run(output)
                print(sess.run(output))
