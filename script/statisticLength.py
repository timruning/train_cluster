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
    # while len(x) < 100:
    #     x.append(0)
    x = numpy.array(x, dtype=numpy.uint32)
    return y, x


if __name__ == '__main__':
    path = "/opt/develop/workspace/sohu/NFM/Train_Cluster/data/xaa"
    file = open(path, 'r')
    result = dict()

    while True:
        line = file.readline()
        if not line:
            break
        y, x = parseData(line)

        length = numpy.shape(x)[0]
        key = (int(length / 10) + 1) * 10
        if result.keys().__contains__(key):
            result[key] += 1
        else:
            result[key] = 1
    for key in result.keys():
        print(key, "\t", result[key])
