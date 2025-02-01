# from numbapro import cuda, vectorize, guvectorize, check_cuda
# from numbapro import void, uint8 , uint32, uint64, int32, int64, float32, float64, f8

import itertools
import random
import sys
import time
from typing import Final
from numba import njit
# import pygame as pg
# import torch
import numpy as np
from math import *
from sortedcontainers import SortedList, SortedSet, SortedDict
# from model import *
from copy import deepcopy
from graphviz import Digraph
import pickle
from tqdm import tqdm
import json

Value = 0


def custom_comparator(a):
    # Например, сравниваем по длине строк
    return a[0]


# Создайте SortedList с использованием custom_comparator
sorted_list = SortedList(key=custom_comparator)


class Derevo:
    def __init__(self, u, v, q, flag, pred):
        if pred != -1:
            self.depth = pred.depth + 1
        else:
            self.depth = 0
        self.n = 1
        self.w = 0
        self.u = np.array(u, dtype=np.int8).tobytes()
        self.v = np.array(v, dtype=np.int8).tobytes()
        self.q = np.array(q, dtype=np.int8).tobytes()
        self.pred = pred
        self.sled = SortedList(key=custom_comparator)
        self.flag = flag
        global N
        self.setrd = set()
        global Value
        self.value = Value
        Value += 1

    def sv(self):
        dop = [self.n, self.w, self.flag, self.value] + [self.u, self.v, self.q]
        if self.pred != -1:
            dop += [self.pred.value]
        else:
            dop += [self.pred]
        dop += [i[1].value for i in self.sled]
        # print(sys.getsizeof(dop))
        return dop

    def load(self, l, data):
        self.n = l[0]
        self.w = l[1]
        self.flag = l[2]
        self.value = l[3]
        self.u = l[4]
        self.v = l[5]
        self.q = l[6]
        self.pred = data[l[7]]
        self.sled = dict()
        for i in l[8:]:
            self.sled.add(data[i])

    def pole(self):
        global POLE
        dop = deepcopy(POLE) - np.multiply.outer(np.multiply.outer(
            np.frombuffer(self.u, np.int8).ravel(), np.frombuffer(self.v, np.int8).ravel()),
            np.frombuffer(self.q, np.int8).ravel())
        it = self.pred
        while it != -1:
            dop -= np.multiply.outer(np.multiply.outer(
                np.frombuffer(it.u, np.int8).ravel(), np.frombuffer(it.v, np.int8).ravel()),
                np.frombuffer(it.q, np.int8).ravel())
            it = it.pred
        return dop


def fun(C, T, i):
    return i.w / i.n + C * sqrt(log(T) / i.n)


def fun_zero(C, T):
    return C * sqrt(log(T))


def new(derevo, dop):
    global N, SN, C, T, lim_hod
    # U = itertools.product([-1, 0, 1], repeat=N ** 2)
    # V = itertools.product([-1, 0, 1], repeat=N ** 2)
    # W = itertools.product([-1, 0, 1], repeat=N ** 2)
    flag = True
    if derevo.depth == lim_hod - 1:
        flag = False
    l3 = dop % mod_N
    l2 = dop // mod_N % mod_N
    l1 = dop // mod_N // mod_N
    for u in itertools.islice(itertools.product([-1, 0, 1], repeat=SN), l1, l1 + 1):
        for v in itertools.islice(itertools.product([-1, 0, 1], repeat=SN), l2, l2 + 1):
            for w in itertools.islice(itertools.product([-1, 0, 1], repeat=SN), l3, l3 + 1):
                derevo.setrd.add(dop)
                dopd = Derevo(u, v, w, flag, derevo)
                derevo.sled.add([fun(C, T, dopd), dopd, dop])
                return flag, dopd
    print("NO", l1, l2, l3, dop)
    return flag, None


def if_game(pole, num, Value):
    global N, SN, lim_hod
    if num < lim_hod:
        if (pole == np.zeros((SN, SN, SN), dtype=np.int8)).all():
            print("YES")
            print(Value)
            return 1
        else:
            return 0
    return 1


# def win(derevo):
#     global N
#     return derevo.sled[random.randint(0, (N ** 2) ** 3)]
# sw = [i for i in range((N ** 2) ** 3)]
# random.shuffle(sw)
# for i in sw:
#     derevo = derevo.sled[i]
#     return derevo


def mod(derevo):
    global lim_point
    p = if_game(derevo.pole(), derevo.depth, derevo.value)
    flag = True
    while p == 0 and flag:
        dop = random.randint(0, lim_point)
        if not dop in derevo.setrd:
            flag, derevo = new(derevo, dop)
        p = if_game(derevo.pole(), derevo.depth, derevo.value)
    return derevo, p


def back(derevo, hod):
    kef = 0
    if hod == 1:
        kef = 1
    while derevo.pred != -1:
        derevo.w += kef
        derevo.n += 1
        derevo = derevo.pred


def draw(derevo, C, T):
    def format(node):
        s = f"{str(fun(C, T, node))[:3]}"
        return s

    def add_edges(graph, node):
        for i in node.sled:
            graph.edge(str(node.value), str(i.value), label=f"{format(i)}")
            add_edges(graph, i)

    def visualize_tree(root):
        graph = Digraph()
        add_edges(graph, root)
        graph.render(r'tree', format='svg', cleanup=True)

    visualize_tree(derevo)


def save(sv, kor):
    sv[kor.value] = kor.sv()
    # a=sys.getsizeof(kor.u.tobytes())
    # b=sys.getsizeof([kor.u.tobytes(), kor.v.tobytes(), kor.w.tobytes()])
    for i in kor.sled:
        sv = save(sv, i[1])
    return sv


def init_pole():
    global N, SN
    pole = [[[0] * SN for i in range(SN)] for j in range(SN)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                pole[i * N + k][k * N + j][i * N + j] = 1
    for i in range(SN):
        for j in range(SN):
            print(*pole[i][j])
        print('------------------------------------------------------------------')
    return np.array(pole, dtype=np.int8)


def algorithm(kor, T):
    derevo = kor
    while len(derevo.sled) > 0:
        dop = random.randint(0, lim_point)
        if len(derevo.sled) >= dop:
            mai = derevo.sled[-1][1]
            derevo = mai
            # print("mai")
        else:
            if derevo.flag:
                dop1 = random.randint(0, lim_point)
                while dop1 in derevo.setrd:
                    dop1 = random.randint(0, lim_point)
                flag, derevo = new(derevo, dop1)
                # flag, derevo = new(derevo, random.choice(list(full_range-derevo.setrd)))
            else:
                break
    # print("2")
    derevo, p = mod(derevo)
    T += 1
    # print("3")
    back(derevo, p)

@njit
def tr(kor,T):
    for i in kor.sled:
        algorithm(i[1], T)

N: Final[int] = 2
SN: Final[int] = N ** 2
mod_N: Final[int] = 3 ** SN
lim_point: Final[int] = 3 ** (SN * 3) - 1
POLE = init_pole()
lim_hod: Final[int] = N ** 3 * 3
# full_range = set(range(lim_point+1))
C: Final[float] = sqrt(2)
T = 1


def Game():
    global N, lim_point, SN, C, T
    kor = Derevo([0] * SN, [0] * SN, [0] * SN, True, -1)
    epoch_kol = 10 ** 6
    print(epoch_kol / 3 ** (N ** 3 * SN * 3), lim_point)
    for i in tqdm(range(lim_point+1)):
        new(kor, i)
    for epoch in range(epoch_kol):
        tr(kor, T)
    global Value
    print(Value)

    # draw(kor, C, T)
    # print("draw")

    data = [0] * Value
    file = {
        "data": save(data, kor),
        "C": C,
        "epoch_kol": epoch_kol,
    }
    # with open('data.txt', 'w') as ff:
    #     for i in file["data"]:
    #         for j in i:
    #             ff.write(f"{str(j)} ")
    #         ff.write('\n')
    print(T, len(file["data"]), sys.getsizeof(file["data"]))
    with open('data.pickle', 'wb') as ff:
        pickle.dump(file, ff, pickle.HIGHEST_PROTOCOL)
        # pickle.dump(file, ff, -1)


def print_hi(name):
    tim = time.time()
    Game()
    print(time.time() - tim)


if __name__ == '__main__':
    print_hi('PyCharm')
