'''
    transfor onehot file to CSV file
'''

from multiprocessing import Pool

from script import Console

SEP = '\u0001'


def transToCSV(path, length):
    file = open(path, 'r')
    path_out = path + ".csv"
    file_out = open(path_out, 'w')
    while True:
        line = file.readline()
        if not line:
            break
        uid = None
        label = None
        newsid = None
        groupId = None
        feature = []
        tmp = line.strip().split(SEP)

        for v in tmp:
            elem = v.split(":")
            if len(elem) != 2:
                continue
            key = elem[0]
            value = elem[1]
            if 'label'.__eq__(key) and str.isnumeric(value):
                value_num = int(value)
                if value_num == 1:
                    label = value_num
                elif value_num == 0:
                    label = -1
                else:
                    continue
            elif 'uid'.__eq__(key):
                uid = value
            elif 'newsId'.__eq__(key):
                newsid = value
            elif 'content_group_id'.__eq__(key):
                groupId = value
            elif str.isnumeric(key) and '1'.__eq__(value):
                value_i = int(key) + 1
                feature.append(value_i)
        if uid is None or label is None or newsid is None or groupId is None or len(feature) < 10:
            continue
        feature = sorted(feature)
        if len(feature) > length - 1:
            feature = feature[:length - 1]
        feature = feature + [0] * (length - len(feature))

        tmpline = str(label) + "," + uid + "," + newsid + "," + groupId + "," + ",".join(
            [str(v) for v in feature]) + "\n"
        file_out.write(tmpline)
    file_out.close()
    file.close()


if __name__ == '__main__':
    path_dir = '../data/2018-01-18'

    files = Console.getDirFiles(path_dir)
    print(files)
    pool = Pool(6)
    for file in files:
        pool.apply_async(func=transToCSV, args=(file, 50,))
    pool.close()
    pool.join()
    print('all done')
