import numpy


def parseData(line):
    tmp = line.strip().split("\u0001")
    x = []
    y = None
    for v in tmp:
        elem = v.split(":")
        if len(elem) == 2:

            if "label".__eq__(elem[0]):
                y = int(elem[1])
            elif str.isnumeric(elem[0]) and "1".__eq__(elem[1]):
                x.append(int(elem[0]))
    while len(x) < 100:
        x.append(0)
    x = numpy.array(x, dtype=numpy.uint32)
    return y, x


if __name__ == '__main__':
    path = "../data/onehot"
    file = open(path, 'r')
    X = []
    Y = []
    index = 0

    while True:
        line = file.readline()
        if "".__eq__(line):
            break
        y, x = parseData(line)
        X.append(x)
        Y.append(y)
        index += 1
        if index % 10000 == 0:
            print(index)
    print(index)
