# coding=utf-8
import numpy


def getFeatureSize(path):
    file = open(path, 'r')
    tm = len(file.readlines()) + 1
    file.close()
    return tm


def parseLine(line):
    x = []
    y = []

    elem = line.strip().split("\u0001")
    for v in elem:
        tmp = v.split(":")
        if len(tmp) == 2:
            if "label".__eq__(tmp[0]):
                if "0".__eq__(tmp[1]):
                    y.append(-1)
                elif "1".__eq__(tmp[1]):
                    y.append(1)
                else:
                    continue
            elif str.isnumeric(tmp[0]) and "1".__eq__(tmp[1]):
                x.append(int(tmp[0]) + 1)

    x = numpy.array(x, dtype=numpy.uint32)
    y = numpy.array(y, dtype=numpy.int8)
    return y, x


def loadData_csv(path):
    X = []
    Y = []
    file = open(path, 'r')
    while True:
        line = file.readline()
        if not line:
            break
        tmp = line.strip().split(",")
        if len(tmp) != 54:
            print(line)
        else:
            x = []
            y = []
            if tmp[0].__eq__("1"):
                y.append(1)
            else:
                y.append(-1)

            for i in range(4, len(tmp)):
                x.append(int(tmp[i]))

            if len(x) != 50:
                print(line)

            x = numpy.array(x, dtype=numpy.uint32)
            y = numpy.array(y, dtype=numpy.int8)
            X.append(x)
            Y.append(y)
    file.close()

    data = dict()
    data['X'] = X
    data['Y'] = Y
    return data


def loadData(path, featureLength):
    X = []
    Y = []
    file = open(path, 'r')
    while True:
        line = file.readline()
        if not line:
            break
        y, x = parseLine(line)

        length = numpy.shape(x)[0]
        if length < 15 or length > featureLength - 1:
            # print(line)
            continue

        x = numpy.concatenate((x, [0] * (featureLength - length)), axis=0)
        # print("x length = ", numpy.shape(x)[0])

        X.append(x)
        Y.append(y)
    file.close()

    data = dict()
    data['X'] = X
    data['Y'] = Y
    return data


def shuffle_batch(data, batch_size):
    start_index = numpy.random.randint(0, len(data['Y']) - batch_size)
    X, Y = [], []
    i = start_index
    while len(X) < batch_size and i < len(data['X']):
        if len(data['X'][i]) == len(data['X'][start_index]):
            Y.append(data['Y'][i])
            X.append(data['X'][i])
            i += 1
        else:
            break
    i = start_index
    while len(X) < batch_size and i >= 0:
        if len(data['X'][i]) == len(data['X'][start_index]):
            Y.append([data['Y'][i]])
            X.append(data['X'][i])
            i = i - 1
        else:
            break
    return {'X': X, 'Y': Y}


def batch(data, batch_size, begin):
    start_index = begin
    X, Y = [], []
    i = start_index
    while len(X) < batch_size and i < len(data['X']):
        if len(data['X'][i]) == len(data['X'][start_index]):
            Y.append(data['Y'][i])
            X.append(data['X'][i])
            i += 1
        else:
            break
    return {'X': X, 'Y': Y}


if __name__ == '__main__':
    path = "/opt/develop/workspace/sohu/NFM/Train_Cluster/data/train/onehot.csv"
    x = loadData_csv(path=path)

