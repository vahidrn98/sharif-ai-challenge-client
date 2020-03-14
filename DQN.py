from tensorflow.python.keras.optimizers import Adam, Adagrad
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense, Dropout,Activation
import random,util
import numpy as np
import pandas as pd
from operator import add
from model import *
from world import World
import tensorflow as tf
from model import BaseUnit, Map, King, Cell, Path, Player, GameConstants, TurnUpdates, \
    CastAreaSpell, CastUnitSpell, CastSpell, Unit, Spell, Message, UnitTarget, SpellType, SpellTarget, Logs

class DQNAgent(object):

    def __init__(self):
        f = open("args.txt", "rt")
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        # self.model = self.network()
        self.model = self.network("agent.h5")
        self.epsilon = float(f.readline())
        self.actual = []
        self.memory = []
        self.alpha = 0.99
        self.qValues = util.Counter()
        self.our_health=0
        self.enemy_health=0
        self.ourtower_health=0
        self.enemytower_health=0
        self.i_am_alive=True
        self.friend_alive=True
        self.fenemy_alive=True
        self.senemy_alive=True
        f.close()

    def get_state(self,world: World):
        def sort_func(elem):
            return elem.unit_id

        # custom_map = np.zeros(shape=(world.get_map.row_num,world.get_map.col_num,5))
        # print("map dimension",world.get_map.row_num*world.get_map.col_num)
        mylast_units = np.zeros(shape=(5,),dtype=int)
        mylast_units_health = np.zeros(shape=(5,),dtype=int)
        mylast_units_cell = np.zeros(shape=(5,),dtype=int)
        mylast_units_atacking_king = np.zeros(shape=(5,),dtype=int)
        myself = world.get_me()
        myunits=[x for x in myself.units]
        myunits.sort(key=sort_func)
        s=slice(5)
        myunits=[x for x in myunits]
        myunits=myunits[s]
        for i in range(5):
            if(i<len(myunits)):
                mylast_units_health[i]=[x.hp for x in myunits][s][i]
                mylast_units_cell[i]=np.hstack([[x.cell.row,x.cell.col] for x in myunits])[s][i]
                mylast_units_atacking_king[i]=1 if myunits[i].target_if_king is not None else 0
                mylast_units[i]=myunits[i].base_unit.type_id+1
               

        
        friendlast_units = np.zeros(shape=(5,),dtype=int)
        friendlast_units_health = np.zeros(shape=(5,),dtype=int)
        friendlast_units_cell = np.zeros(shape=(5,),dtype=int)
        friendlast_units_atacking_king = np.zeros(shape=(5,),dtype=int)
        friend = world.get_friend()
        friendunits=[x for x in friend.units]
        friendunits.sort(key=sort_func)
        s=slice(5)
        friendunits=[x for x in friendunits]
        friendunits=friendunits[s]
        for i in range(5):
            if(i<len(friendunits)):
                friendlast_units_health[i]=[x.hp for x in friendunits][s][i]
                friendlast_units_cell[i]=np.hstack([[x.cell.row,x.cell.col] for x in friendunits])[s][i]
                friendlast_units_atacking_king[i]=1 if friendunits[i].target_if_king is not None else 0
                friendlast_units[i]=friendunits[i].base_unit.type_id+1

        en1last_units = np.zeros(shape=(5,),dtype=int)
        en1last_units_atacking_king = np.zeros(shape=(5,),dtype=int)
        en1last_units_health = np.zeros(shape=(5,),dtype=int)
        en1last_units_cell = np.zeros(shape=(5,),dtype=int)
        en1 = world.get_first_enemy()
        en1units=[x for x in en1.units]
        en1units.sort(key=sort_func)
        s=slice(5)
        en1units=[x for x in en1units]
        en1units=en1units[s]
        for i in range(5):
            if(i<len(en1units)):
                en1last_units_health[i]=[x.hp for x in en1units][s][i]
                en1last_units_cell[i]=np.hstack([[x.cell.row,x.cell.col] for x in en1units])[s][i]
                en1last_units_atacking_king[i]=1 if en1units[i].target_if_king is not None else 0
                en1last_units[i]=en1units[i].base_unit.type_id+1

        en2last_units = np.zeros(shape=(5,),dtype=int)
        en2last_units_health = np.zeros(shape=(5,),dtype=int)
        en2last_units_atacking_king = np.zeros(shape=(5,),dtype=int)
        en2last_units_cell = np.zeros(shape=(5,),dtype=int)
        en2 = world.get_first_enemy()
        en2units=[x for x in en2.units]
        en2units.sort(key=sort_func)
        s=slice(5)
        en2units=[x for x in en2units]
        en2units=en2units[s]
        for i in range(5):
            if(i<len(en2units)):
                en2last_units_health[i]=[x.hp for x in en2units][s][i]
                en2last_units_cell[i]=np.hstack([[x.cell.row,x.cell.col] for x in en2units])[s][i]
                en2last_units_atacking_king[i]=1 if en2units[i].target_if_king is not None else 0
                en2last_units[i]=en2units[i].base_unit.type_id+1

        spell_target=spell_type=is_area=is_damaging=None
        if world.get_remaining_turns_to_get_spell()==1 and world.get_received_spell()!=None:
            spell_target=world.get_received_spell().target
            spell_type =world.get_received_spell().type
            is_area=world.get_received_spell().is_area_spell()
            is_damaging=world.get_received_spell().is_damaging()
            
        state = [
            [world.get_me().ap],
            [1 if world.get_me().is_alive()==1 and world.get_friend().is_alive()==1 else 0],
            [1 if world.get_first_enemy().is_alive()==1 and world.get_second_enemy().is_alive()==1 else 0],
            [world.get_me().hand[0].ap],
            [world.get_me().hand[1].ap],
            [world.get_me().hand[2].ap],
            [world.get_me().hand[3].ap],
            [world.get_me().hand[4].ap],
            [world.get_me().hand[0].type_id],
            [world.get_me().hand[1].type_id],
            [world.get_me().hand[2].type_id],
            [world.get_me().hand[3].type_id],
            [world.get_me().hand[4].type_id],
            [world.get_me().get_hp()],
            [world.get_friend().get_hp()],
            [world.get_first_enemy().get_hp()],
            [world.get_second_enemy().get_hp()],
            [world.get_me().is_alive()],
            [world.get_friend().is_alive()],
            [world.get_first_enemy().is_alive()],
            [world.get_second_enemy().is_alive()],
            [world.get_range_upgrade_number()+world.get_damage_upgrade_number()],
            [1 if world.get_remaining_turns_to_get_spell() == 1 else 0],
            [1 if spell_target == SpellTarget.ENEMY else 0],
            [1 if spell_target == SpellTarget.SELF else 0],
            [1 if spell_target == SpellTarget.ALLIED else 0],
            [1 if spell_target == SpellType.TELE else 0],
            [1 if spell_target == SpellType.HASTE else 0],
            [1 if spell_target == SpellType.HP else 0],
            [1 if spell_target == SpellType.DUPLICATE else 0],
            [1 if is_area else 0],
            [1 if is_damaging else 0],
            mylast_units,
            friendlast_units,
            mylast_units_atacking_king,
            friendlast_units_atacking_king,
            en1last_units_atacking_king,
            en2last_units_atacking_king,
            en1last_units,
            en2last_units,
            mylast_units_health,
            mylast_units_cell,
            friendlast_units_health,
            friendlast_units_cell,
            en1last_units_health,
            en1last_units_cell,
            en2last_units_health,
            en2last_units_cell,
            [len(world.get_me().died_units)],
            [len(world.get_friend().died_units)],
            [len(world.get_first_enemy().died_units)],
            [len(world.get_second_enemy().died_units)],
            [len(world.get_me().played_units)],
            [len(world.get_friend().played_units)],
            [len(world.get_first_enemy().played_units)],
            [len(world.get_second_enemy().played_units)],
            [world.get_current_turn()]
        ]
        
        #hstack and turn all to int

        return np.hstack(state)

    def set_reward(self,world:World):
        self.reward = 0
        self.reward=self.reward+(len(world.get_me().died_units)+len(world.get_friend().died_units))*-15
        self.reward=self.reward+(len(world.get_first_enemy().died_units)+len(world.get_second_enemy().died_units))*10
        self.reward=self.reward+(np.sum([x.hp for x in world.get_me().units])+np.sum([x.hp for x in world.get_friend().units])-self.our_health+self.enemy_health)-np.sum([x.hp for x in world.get_first_enemy().units])-np.sum([x.hp for x in world.get_second_enemy().units])
        self.reward=self.reward-( world.get_me().get_hp()+world.get_friend().get_hp()-self.ourtower_health)+(world.get_first_enemy().get_hp()+world.get_second_enemy().get_hp()-self.enemytower_health)*1.5
        self.our_health=np.sum([x.hp for x in world.get_me().units])+np.sum([x.hp for x in world.get_friend().units])
        self.enemy_health=np.sum([x.hp for x in world.get_first_enemy().units])+np.sum([x.hp for x in world.get_second_enemy().units])
        self.ourtower_health=world.get_me().get_hp()+world.get_friend().get_hp()
        self.enemytower_health=world.get_first_enemy().get_hp()+world.get_second_enemy().get_hp()
        if(self.i_am_alive and ~world.get_me().is_alive()):
            self.i_am_alive=False
            self.reward=self.reward-2000
        if(self.friend_alive and ~world.get_friend().is_alive()):
            self.friend_alive=False
            self.reward=self.reward-1000
        if(self.fenemy_alive and ~world.get_first_enemy().is_alive()):
            self.fenemy_alive=False
            self.reward=self.reward+500
        if(self.senemy_alive and ~world.get_second_enemy().is_alive()):
            self.senemy_alive=False
            self.reward=self.reward+500
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        
        if weights:
            model=load_model(weights)
        # autoencoder layer
        else:
            model.add(Dense(120, activation='relu', input_dim=121))
            model.add(Dropout(0.15))
            model.add(Dense(120, activation='relu'))
            model.add(Dropout(0.15))
            model.add(Dense(120, activation='relu'))
            model.add(Dropout(0.15))
            model.add(Dense(120, activation='relu'))
            # model.add(Dropout(0.1))
            model.add(Dense(15,activation='sigmoid'))
            opt = Adagrad(self.learning_rate)
            model.compile(loss='mse', optimizer=opt)
        
        model._make_predict_function()
        
        
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        # print(action)
        target = reward
        if not done:
            target = reward + self.gamma * \
                np.amax(self.model.predict(next_state.reshape((1, 121)))[0])
        target_f = action
        for i in range(len(action[0])):
            if(action[0][i]==1):
                target_f[0][i] = target
        print("target",target_f)
        self.model.fit(state.reshape((1, 121)),np.array(target_f[0]).reshape((1,15)), epochs=1, verbose=0)
