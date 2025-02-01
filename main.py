import itertools
import random
import sys
import time

# import pygame as pg
# import torch
import numpy as np
from math import *

# from model import *
from copy import deepcopy
from graphviz import Digraph
import pickle
from tqdm import tqdm
import json

Value = 0


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
        self.sled = list()
        self.flag = flag
        global Value
        self.value = Value
        Value += 1

    def sv(self):
        dop = [self.n, self.w, self.flag, self.value] + [self.u, self.v, self.q]
        if self.pred != -1:
            dop += [self.pred.value]
        else:
            dop += [self.pred]
        dop += [i.value for i in self.sled]
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
        self.sled = []
        for i in l[8:]:
            self.sled.append(data[i])

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


def new(derevo):
    global N
    # U = itertools.product([-1, 0, 1], repeat=N ** 2)
    # V = itertools.product([-1, 0, 1], repeat=N ** 2)
    # W = itertools.product([-1, 0, 1], repeat=N ** 2)
    flag = True
    if derevo.depth == N ** 3 - 2:
        flag = False
    for u in itertools.product([-1, 0, 1], repeat=N ** 2):
        for v in itertools.product([-1, 0, 1], repeat=N ** 2):
            for w in itertools.product([-1, 0, 1], repeat=N ** 2):
                derevo.sled.append(Derevo(u, v, w, flag, derevo))
    return flag


def if_game(pole, num, Value):
    global N
    if num < N ** 3 - 1:
        if (pole == np.zeros((N ** 2, N ** 2, N ** 2), dtype=np.int8)).all():
            print("YES")
            print(Value)
            return 1
        else:
            return 0
    return 0


def win(derevo):
    global N
    return derevo.sled[random.randint(0, (N ** 2) ** 3)]
    # sw = [i for i in range((N ** 2) ** 3)]
    # random.shuffle(sw)
    # for i in sw:
    #     derevo = derevo.sled[i]
    #     return derevo


def mod(derevo):
    p = if_game(derevo.pole(), derevo.depth, derevo.value)
    flag = True
    while p == 0 and flag:
        if len(derevo.sled) == 0:
            flag = new(derevo)
        derevo = win(derevo)
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
        sv = save(sv, i)
    return sv


def init_pole():
    global N
    pole = [[[0] * N ** 2 for i in range(N ** 2)] for j in range(N ** 2)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                pole[i * N + k][k * N + j][i * N + j] = 1
    for i in range(N ** 2):
        for j in range(N ** 2):
            print(*pole[i][j])
        print('------------------------------------------------------------------')
    return np.array(pole, dtype=np.int8)


N = 2
POLE = init_pole()


def Game():
    global N
    C = sqrt(2)
    T = 0
    kor = Derevo([0] * N ** 2, [0] * N ** 2, [0] * N ** 2, True, -1)
    epoch_kol = 10
    print(epoch_kol / 3 ** (N ** 3 * N ** 2 * 3))
    for epoch in tqdm(range(epoch_kol)):
        derevo = kor
        while len(derevo.sled) > 0:
            ma = -(10 ** 10)
            mai = 0
            for i in derevo.sled:
                if i.flag:
                    s = fun(C, T, i)
                    if s > ma:
                        ma = s
                        mai = i
            derevo = mai
        derevo, p = mod(derevo)
        T += 1
        back(derevo, p)
    global Value
    print(Value)
    # print(sys.getsizeof(kor))
    # print(sys.getsizeof(json.dumps(kor)))
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
    Game()


if __name__ == '__main__':
    print_hi('PyCharm')
