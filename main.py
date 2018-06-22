from keras.layers import Dense
from collections import deque
from random import randint
from keras import Sequential
import numpy as np


class NNModel:
    def __init__(self):
        self.nn_model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(2, input_dim=3, activation='sigmoid'))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='Adam')

        return model

def zero():
    return np.array([0, 0, 0]), 1


def one():
    return np.array([0, 0, 1]), 0


def two():
    return np.array([0, 1, 0]), 1


def three():
    return np.array([0, 1, 1]), 0


def four():
    return np.array([1, 0, 0]), 1


def five():
    return np.array([1, 0, 1]), 0


def six():
    return np.array([1, 1, 0]), 1


def seven():
    return np.array([1, 1, 1]), 0

def dataset_init():
    dataset = deque(maxlen=1000)
    for i in range(0, dataset.maxlen):
        number = randint(0, 7)
        switcher = {
            0: zero(),
            1: one(),
            2: two(),
            3: three(),
            4: four(),
            5: five(),
            6: six(),
            7: seven()
        }
        dataset.append((switcher.get(number)))

    return dataset

def train(model, dataset):

    for i in range(0, 32):
        index = randint(0, dataset.maxlen - 1)
        binar_number = np.array(dataset.__getitem__(index)[0])
        binar_number = np.expand_dims(binar_number, axis=0)
        label = np.array([dataset.__getitem__(index)[1]])
        model.nn_model.fit(binar_number,  label, batch_size=1)

if __name__ == "__main__":

    model = NNModel()

    dataset = dataset_init()

    for i in range(0, 1000):
        train(model, dataset)

    for i in range(0, dataset.maxlen):
        binar_number = binar_number = np.array(dataset.__getitem__(i)[0])
        binar_number = np.expand_dims(binar_number, axis=0)
        print('Number: {}, prediction: {}'.format(binar_number, model.nn_model.predict(binar_number)))