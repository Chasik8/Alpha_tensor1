import itertools
import random
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

Value = 0


class Derevo:
    def __init__(self, pole, flag, pred):
        if pred != -1:
            self.depth = pred.depth + 1
        else:
            self.depth = 0
        self.n = 1
        self.w = 0
        # self.u = [0] * N
        # self.v = [0] * N
        # self.w = [0] * N
        self.pred = pred
        self.sled = list()
        self.flag = flag
        self.pole = np.array(pole, dtype=np.int8)
        global Value
        self.value = Value
        Value += 1


def fun(C, T, i):
    return i.w / i.n + C * sqrt(log(T) / i.n)


def new(derevo):
    global N
    pole = derevo.pole
    # U = itertools.product([-1, 0, 1], repeat=N ** 2)
    # V = itertools.product([-1, 0, 1], repeat=N ** 2)
    # W = itertools.product([-1, 0, 1], repeat=N ** 2)
    flag = True
    if derevo.depth == N ** 3 - 2:
        flag = False
    for u in itertools.product([-1, 0, 1], repeat=N ** 2):
        for v in itertools.product([-1, 0, 1], repeat=N ** 2):
            for w in itertools.product([-1, 0, 1], repeat=N ** 2):
                dop = deepcopy(pole) - np.multiply.outer(np.multiply.outer(
                    np.array(u, dtype=np.int8).ravel(), np.array(v, dtype=np.int8).ravel()),
                    np.array(w, dtype=np.int8).ravel())
                derevo.sled.append(Derevo(dop, flag, derevo))
    return flag


def if_game(pole, num,Value):
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
    p = if_game(derevo.pole, derevo.depth, derevo.value)
    flag = True
    while p == 0 and flag:
        if len(derevo.sled) == 0:
            flag = new(derevo)
        derevo = win(derevo)
        p = if_game(derevo.pole, derevo.depth, derevo.value)
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
    sv.append(kor)
    for i in kor.sled:
        sv = save(sv, i)
    return sv


N = 2


def Game():
    global N
    C = sqrt(2)
    T = 0
    pole = [[[0] * N ** 2 for i in range(N ** 2)] for j in range(N ** 2)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                pole[i * N + k][k * N + j][i * N + j] = 1
    for i in range(N ** 2):
        for j in range(N ** 2):
            print(*pole[i][j])
        print('------------------------------------------------------------------')
    kor = Derevo(pole, True, -1)
    epoch_kol = 100
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
    # draw(kor, C, T)
    # print("draw")
    data = []
    file = {
        "data": save(data, kor),
        "C": C,
        "epoch_kol": epoch_kol,
    }
    print(T, len(file["data"]))
    with open('data.pickle', 'wb') as ff:
        pickle.dump(file, ff, pickle.HIGHEST_PROTOCOL)


def print_hi(name):
    Game()


if __name__ == '__main__':
    print_hi('PyCharm')
