import pickle
from copy import deepcopy

import numpy as np

from main import Derevo, fun, if_game


def print_hi(name):
    file = None
    with open('data_test_main2.pickle', 'rb') as ff:
        file = pickle.load(ff)
    # file["data"] = file["data"]
    kor = file["data"][0]
    C = file['C']
    T = len(file['data'])
    derevo = kor
    while True:
        ma = -(10 ** 10)
        mai = None
        for i in derevo[9:]:
            # s = fun(C, T, i)
            s = file["data"][i][1] / file["data"][i][0]
            if s > ma:
                ma = s
                mai = i
        if mai is None:
            if derevo[1] > 0:
                print("YEEEEEEES")
                while derevo[8] != -1:
                    print(derevo)
                    derevo = file["data"][derevo[8]]
            else:
                print("Not found hod")
            break
        derevo = file["data"][mai]


if __name__ == '__main__':
    print_hi('PyCharm')
