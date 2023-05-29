import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        self.input_size = 4 + 4
        # Количество узлов на скрытом слое
        self.num_classes = 4 + 4 + 3 * 4
        # self.num_epochs = 10**5  # Количество тренировок всего набора данных
        # self.batch_size = 100  # Размер входных данных для одной итерации
        # self.learning_rate = 0.001  # Скорость конвергенции
        # -----------------------------------------------------------
        super(Net, self).__init__()  # Наследуемый родительским классом nn.Module
        self.fc1 = nn.Linear(self.input_size,
                             100)
        self.fc2 = nn.Linear(100,
                             self.num_classes)
        # self.fc3 = nn.Linear(5000,
        #                      100)
        # self.fc4 = nn.Linear(100,
        #                      self.num_classes)
        self.relu = nn.ReLU()
        # self.fc1_dop=nn.Linear(self.input_size, self.num_classes)
        # self.model = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu, self.fc3, self.relu, self.fc4)
        self.model = nn.Sequential(self.fc1, self.relu, self.fc2)

    def out_model(self):
        return self.model

    def out(self):
        return self.num_classes

    def inp(self):
        return self.input_size

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.fc4(x)
        x = self.model(x)
        return x
