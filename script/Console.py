import numpy as np
import os
def printArray(array):
    line = ""
    for v in array:
        line = line + "\t" + str(v)
    print(line)

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def getDirFiles(path):
    result = []
    if not os.path.exists(path):
        return result
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            result.append(dirpath + '/' + file)
    result = sorted(result)
    return result

if __name__ == '__main__':
    a=np.array([1,2,3])
    b=np.array([-1,-2,-3])
    printArray(a)
    printArray(b)
    shuffle_in_unison_scary(a,b)
    printArray(a)
    printArray(b)