from tensorflow.python.keras.optimizers import Adam,Adagrad
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
from model import *
from world import World



class Picker(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.001
        # self.model = self.network()
        self.model = self.network("picker.h5")
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_state(self):

        state = [1,1,1,1,1,1,1,1,1]

        return np.array(state)

    def set_reward(self,world:World):
        self.reward = 0
        if(~world.get_friend().is_alive() and ~world.get_me().is_alive()):
            self.i_am_alive=False
            self.reward=-10
        elif(~world.get_second_enemy().is_alive() and ~world.get_first_enemy().is_alive()):
            self.fenemy_alive=False
            self.reward=10
        else:
            self.reward=2
        return self.reward

    def network(self, weights=None):

        if weights:
            model=load_model(weights)
            print(model)
        else:
            model = Sequential()
            model.add(Dense(30, activation='relu', input_dim=9))
            #model.add(Dense(output_dim=70, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(30, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(30, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(9, activation='sigmoid'))
            opt = Adam(self.learning_rate)
            model.compile(loss='mse', optimizer=opt)
        model._make_predict_function()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 100)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 9)))[0])
        target_f = action
        for i in range(len(action[0])):
            if(action[0][i]==1):
                target_f[0][i] = target
        self.model.fit(state.reshape((1, 9)), np.array(target_f[0]).reshape((1,9)), epochs=1, verbose=0)

        
