import pygad.torchga
import torch
from model import *
import numpy as np
import random
from copy import deepcopy
import time


def Run():
    device = "cuda:0"
    # net = Net()
    # torch_ga = pygad.torchga.TorchGA(model=net.out_model(), num_solutions=10)
    kol_model = 10
    torch_ga = []
    for i in range(kol_model):
        torch_ga.append([0, Net().to(device), 0])
    # for i in range(kol_model):
    #     torch_ga[i] += torch.optim.Adam(torch_ga[i][1].parameters())
    # criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    x_train = [1, 8, 64, 512, 4096, 32768, 262144, 2097152]
    for i in range(8):
        x_train[i] /= 8 ** 7
    x_train_torch = torch.from_numpy(np.array(x_train).astype(np.float32)).to(device)
    y_train = [2101248, 16809984, 134479872, 1075838976]
    sum_y_train = sum(y_train)
    epoch = 100
    md1 = 0
    md2 = 0
    tim = time.time()
    for ep in range(epoch):
        otw = [sum_y_train] * kol_model
        for km in range(kol_model):
            # kol_iter = 0
            kol_iter_max = 100
            while torch_ga[km][0] < kol_iter_max and otw[km] != 0:
                x = torch_ga[km][1](x_train_torch)
                x = x.cpu().detach().numpy()
                a = 0
                for i in range(4):
                    if x[i] >= 0.5:
                        a += x_train[i]
                b = 0
                for i in range(4, 8):
                    if x[i] >= 0.5:
                        a += x_train[i]
                c = 0
                ab = a * b
                for i in range(8, 20, 3):
                    if x[i] >= x[i + 1] and x[i] >= x[i + 2]:
                        c += ab
                    elif x[i + 2] >= x[i + 1] and x[i + 2] >= x[i]:
                        c -= ab
                otw[km] -= c
                torch_ga[km][0] += 1
            if torch_ga[km][0] == kol_iter_max:
                torch_ga[km][2] = abs(otw[km])
        # изменнение весов
        # учшие модели-----------------------------------------------------------
        max1 = 10000000000000
        max2 = 10000000000000
        maxdop1 = 1000000000000
        maxdop2 = 1000000000000
        maxi1 = 0
        maxi2 = 0
        for i in range(kol_model):
            if torch_ga[i][0] < max1:
                max2 = max1
                maxi2 = maxi1
                max1 = torch_ga[i][0]
                maxi1 = i
            elif torch_ga[i][0] < max2:
                max2 = torch_ga[i][0]
                maxi2 = i
            elif torch_ga[i][0] == max1:
                if torch_ga[i][2] <= maxdop1:
                    maxdop2 = maxdop1
                    max2 = max1
                    maxi2 = maxi1
                    maxdop1 = torch_ga[i][2]
                    max1 = torch_ga[i][0]
                    maxi1 = i
            elif torch_ga[i][0] == max2:
                if torch_ga[i][2] <= maxdop2:
                    maxdop2 = torch_ga[i][2]
                    max2 = torch_ga[i][0]
                    maxi2 = i
        #-----------------------------------------------------------
        md1 = pygad.torchga.model_weights_as_vector(torch_ga[maxi1][1].model)
        md2 = pygad.torchga.model_weights_as_vector(torch_ga[maxi2][1].model)
        print(f"{ep} {torch_ga[maxi1][0]}")
        for km in range(kol_model):
            md = deepcopy(md1)
            for i in range(len(md1)):
                dop1 = random.uniform(0, 1)
                if 0.45 <= dop1 < 0.9:
                    md[i] = md2[i]
                else:
                    md[i] = random.uniform(-1, 1)
            torch_ga[km][0] = 0
            torch_ga[km][1].model.load_state_dict(
                pygad.torchga.model_weights_as_dict(model=torch_ga[km][1].model, weights_vector=md))
    net = Net()
    net.model.load_state_dict(
        pygad.torchga.model_weights_as_dict(model=net.model, weights_vector=md1))
    torch.save(net, "models\model1.pth")
    print(time.time() - tim)

    # torch_ga[km][1].model.load_state_dict(
    #     pygad.torchga.model_weights_as_dict(model=torch_ga[km][1].model, weights_vector=pr))
    # p = pygad.torchga.model_weights_as_vector(torch_ga[km][1].model)
    print("YES")


def print_hi(name):
    Run()


if __name__ == '__main__':
    print_hi('PyCharm')
