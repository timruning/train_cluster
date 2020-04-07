from multiprocessing import Pool

from script import Console


def splitIn15mins(path, path2):
    fileNameSet = set()
    file = open(path, 'r')
    date = "abc"
    file2 = None

    while True:
        line = file.readline().strip()
        if not line:
            break
        tmp = line.split("\u0001")
        for v in tmp:
            elems = v.split(":")
            if "date".__eq__(elems[0]):
                if not date.__eq__(elems[1]):
                    if file2 is not None:
                        file2.close()
                    date = elems[1]
                    path_w = path2 + "/" + date
                    if fileNameSet.__contains__(date):
                        print(date)
                    else:
                        fileNameSet.add(date)
                    file2 = open(path_w, 'w')
                break
        file2.write(line + "\n")

    if file2 is not None:
        file2.close()


if __name__ == '__main__':
    path = "../data/2018-01-18"
    path2 = ""
    files = Console.getDirFiles(path)
    pool = Pool(6)
    for file in files:
        pool.apply_async(func=splitIn15mins, args=(file, path2))
    pool.close()
    pool.join()
    print('all done')
