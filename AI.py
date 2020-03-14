import random
from model import *
from world import World
from DQN import DQNAgent
from Picker import Picker
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from random import randint
from tensorflow.python.keras.utils import to_categorical
from model import BaseUnit, Map, King, Cell, Path, Player, GameConstants, TurnUpdates, \
    CastAreaSpell, CastUnitSpell, CastSpell, Unit, Spell, Message, UnitTarget, SpellType, SpellTarget, Logs
#it's me

# new version!
# graph = tf.get_default_graph()

class AI:
    def __init__(self):
        
        self.rows = 0
        self.cols = 0
        self.path_for_my_units = None
        self.units_points=np.zeros(shape=(9,),dtype=int)
        self.epsilon=0
        self.agent=DQNAgent()
        self.picker=Picker()
        self.state_old=np.zeros(121)
        self.new_state=[]
        self.picked=np.zeros(9,dtype=int)
        self.prediction=[[]]
        

    # this function is called in the beginning for deck picking and pre process

    def take_action(self,world:World,prediction):
        def sort_func(elem):
            return elem.unit_id


        paths=[]
        # paths.append(world.get_me().paths_from_player[0])
        paths.append(world.get_friend().paths_from_player[0])
        paths.append(world.get_first_enemy().paths_from_player[0])
        paths.append(world.get_second_enemy().paths_from_player[0])

        print("prediction",prediction)

        binary_prediction=prediction

        # norm =(np.array(prediction[0])-np.min(np.array(prediction[0]))/np.ptp(np.array(prediction[0])))
        for i in range(len(prediction[0])):
            binary_prediction[0][i]=1 if binary_prediction[0][i]>0.5 else 0
        
        # print("normalized:",norm)
        print("binary prediction:",binary_prediction[0])
        i=0
        # if(binary_prediction[0][15]==1):
        for j in range(5):
            for k in range(3):
                if(binary_prediction[0][i]==1):
                    world.put_unit(base_unit=world.get_me().hand[j],path=paths[k])
                i=i+1
        en1last_units = np.zeros(shape=(5,),dtype=int)
        en1last_units_health = np.zeros(shape=(5,),dtype=int)
        en1last_units_cell = np.zeros(shape=(5,),dtype=int)
        en1 = world.get_first_enemy()
        en1units=[x for x in en1.units]
        en1units.sort(key=sort_func)
        s=slice(5)
        en1units=[x for x in en1units]
        en1units=en1units[s]
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
        # if(binary_prediction[0][16]==1):

        for unit in world.get_me().units:
            if(unit.target_if_king is not None):
                world.upgrade_unit_damage(unit=unit)
                world.upgrade_unit_range(unit=unit)

        if(len(world.get_me().units)+len(world.get_friend().units)<len(world.get_first_enemy().units)+len(world.get_first_enemy().units)/2):
            for unit in myunits:
                world.upgrade_unit_damage(unit=unit)
                world.upgrade_unit_range(unit=unit)

        for unit in world.get_first_enemy().units:
            if(unit.target_if_king is not None):
                for myunit in world.get_me().units:
                    world.upgrade_unit_damage(unit=myunit)
                    world.upgrade_unit_range(unit=myunit)

        for unit in world.get_second_enemy().units:
            if(unit.target_if_king is not None):
                for myunit in world.get_me().units:
                    world.upgrade_unit_damage(unit=myunit)
                    world.upgrade_unit_range(unit=myunit)

    
        # if(binary_prediction[0][17]==1):
        for spell in myself.spells:
            if(spell.type_id==0):
                world.cast_area_spell(center=myunits[0].cell, spell=spell)
            if(spell.type_id==1):
                world.cast_area_spell(center=en1units[0].cell, spell=spell)
            if(spell.type_id==2):
                for unit in world.get_me().units:
                    if(unit.target_if_king is not None):
                            world.cast_area_spell(center=unit.cell, spell=spell)
                            break
                if(len(world.get_me().units)+len(world.get_friend().units)<len(world.get_first_enemy().units)+len(world.get_first_enemy().units)/2):
                        world.cast_area_spell(center= mylast_units[0].cell, spell=spell)
                        break
            if(spell.type_id==3):
                my_units = myself.units
                if len(my_units) > 0:
                    unit = my_units[0]
                    my_paths = myself.paths_from_player
                    path = my_paths[random.randint(0, len(my_paths) - 1)]
                    size = len(path.cells)
                    cell = path.cells[int((size - 1) / 2)]
                    world.cast_unit_spell(unit=unit, path=path, cell=cell, spell=spell)
            if(spell.type_id==4):
                for unit in world.get_me().units:
                    if(unit.target_if_king is not None):
                            world.cast_area_spell(center=unit.cell, spell=spell)
                            break
                mymax=0
                selected_cell=None
                if(len(world.get_me().units)+len(world.get_friend().units)<len(world.get_first_enemy().units)+len(world.get_first_enemy().units)/2):
                        world.cast_area_spell(center= mylast_units[0].cell, spell=spell)
                        break
                for cell in world.get_map().cells:
                    if(len([x for x in world.get_cell_units(cell=cell) if x in world.get_me().units])>mymax):
                        mymax=len([x for x in world.get_cell_units(cell=cell) if x in world.get_me().units])
                        selected_cell=cell
                if(selected_cell is not None):
                    world.cast_area_spell(center=selected_cell, spell=spell)
            if(spell.type_id==5):
                    world.cast_area_spell(center=en1units[0].cell, spell=spell)
                    
                
            

            
        


    def pick(self, world: World):
        
        # print(world.get_first_enemy().paths_from_player)
        # self.state_old=self.agent.get_state(world=world)
        print("pick started!")
<<<<<<< HEAD
        # print("first state",self.state_old)
=======
>>>>>>> 8de9c05846341467b1bf6330daeadade8ac26f1b
        # pre process
        map = world.get_map()
        self.rows = map.row_num
        self.cols = map.col_num

        # choosing all flying units
        my_hand=[]
        all_base_units = world.get_all_base_units()
        # global graph
	    
        pick = self.picker.model.predict(self.picker.get_state().reshape((1,9)))[0]
        # print(pick)
        pick_sorted=np.flip(np.sort(self.picker.model.predict(self.picker.get_state().reshape((1,9)))[0]))[0:5]
        for i in range(len(all_base_units)):
            if(pick[i] in pick_sorted):
                self.picked[i]=1
                my_hand.append(all_base_units[i])

        # picking the chosen hand - rest of the hand will automatically be filled with random base_units
        # self.old_state=self.agent.get_state(world=world)
        # print("first state",self.old_state)
        world.choose_hand(base_units=my_hand)
        # print(my_hand)
        # other pre process
        self.path_for_my_units = world.get_friend().paths_from_player[0]
        

    # it is called every turn for doing process during the game
    
    def turn(self, world: World):
        
        reward = self.agent.set_reward(world=world)
        self.new_state = self.agent.get_state(world=world)
        # print("state:",self.new_state)
        print("turn started:", world.get_current_turn())
<<<<<<< HEAD
        # print(world.get_map().paths)
=======

>>>>>>> 8de9c05846341467b1bf6330daeadade8ac26f1b
        myself = world.get_me()
        max_ap = world.get_game_constants().max_ap

        print("reward:",reward)
        
        # print(predicton[0])
        # print(to_categorical(np.argmax(self.picker.model.predict(self.picker.get_state().reshape((1,9)))[0]),num_classes=9))
        # play all of hand once your ap reaches maximum. if ap runs out, putUnit doesn't do anything



        epsilon = (1000 - self.agent.epsilon)/10
        self.agent.epsilon=self.agent.epsilon+0.01

        print("epsilon:",epsilon)

        if(epsilon<=1):
            epsilon=1
        
        prediction=self.agent.model.predict(self.agent.get_state(world=world).reshape((1,121)))
        prediction[0][np.argmax(prediction[0][0:3])]=1
        prediction[0][3+np.argmax(prediction[0][3:6])]=1
        prediction[0][6+np.argmax(prediction[0][6:9])]=1
        prediction[0][9+np.argmax(prediction[0][9:12])]=1
        prediction[0][12+np.argmax(prediction[0][12:15])]=1

        for i in range(len(prediction[0])):
            if(prediction[0][i]<1):
                prediction[0][i]=0

        
        print("model prediction",prediction)

        

        

        if randint(0, 100) < epsilon:
            random_action=np.zeros(15,dtype=int)
            
            for spell in myself.spells:
                # if(randint(0,1)==1):
                    # random_action[17]=1
                    if spell.is_area_spell():
                        if spell.target == SpellTarget.ENEMY:
                            enemy_units = world.get_first_enemy().units
                            if len(enemy_units) > 0:
                                world.cast_area_spell(center=enemy_units[randint(0,len(enemy_units)-1)].cell, spell=spell)
                        elif spell.target == SpellTarget.ALLIED:
                            friend_units = world.get_friend().units
                            if len(friend_units) > 0:
                                world.cast_area_spell(center=friend_units[randint(0,len(friend_units)-1)].cell, spell=spell)
                        elif spell.target == SpellTarget.SELF:
                            my_units = myself.units
                            if len(my_units) > 0:
                                world.cast_area_spell(center=my_units[randint(0,len(my_units)-1)].cell, spell=spell)
                    else:
                        my_units = myself.units
                        if len(my_units) > 0:
                            unit = my_units[0]
                            my_paths = myself.paths_from_player
                            path = my_paths[random.randint(0, len(my_paths) - 1)]
                            size = len(path.cells)
                            cell = path.cells[int((size - 1) / 2)]
                            world.cast_unit_spell(unit=unit, path=path, cell=cell, spell=spell)
    
            if (len(myself.units) > 0):
                # random_action[16]=1
                unit = myself.units[0]
                world.upgrade_unit_damage(unit=unit)
                world.upgrade_unit_range(unit=unit)

            paths=[]
            # paths.append(world.get_me().paths_from_player[0])
            paths.append(world.get_friend().paths_from_player[0])
            paths.append(world.get_first_enemy().paths_from_player[0])
            paths.append(world.get_second_enemy().paths_from_player[0])
            if(randint(0,1)==1):
                # random_action[15]=1
                for base_unit in myself.hand:
                    if((world.get_me().ap>=base_unit.ap)):
                        selected=randint(0,len(paths)-1)
                        random_action[myself.hand.index(base_unit)*3+selected]=1
                        world.put_unit(base_unit=base_unit, path=paths[selected])
            prediction=[random_action]
            print("random action",prediction)
            
        else:

            self.take_action(world=world,prediction=prediction)
            
        # print("old state:",self.state_old)
        # print("new state:",self.new_state)

        binary_prediction=prediction

        self.prediction=binary_prediction
        # normalized=(np.array(binary_prediction[0])-np.min(np.array(prediction[0]))/np.ptp(np.array(prediction[0])))
        for i in range(len(prediction[0])):
                # a=binary_prediction[0][i]/10
                # print("a",a)
                binary_prediction[0][i]=1 if binary_prediction[0][i]>0.5 else 0
        self.agent.train_short_memory(np.array(self.state_old), binary_prediction, reward,np.array(self.new_state),((~world.get_friend().is_alive() and ~world.get_me().is_alive())or(~world.get_second_enemy().is_alive() and ~world.get_first_enemy().is_alive())))
        
        self.agent.remember(np.array(self.state_old), binary_prediction[0], reward, np.array(self.new_state),((~world.get_friend().is_alive() and ~world.get_me().is_alive())or(~world.get_second_enemy().is_alive() and ~world.get_first_enemy().is_alive())))

        self.state_old = self.new_state
        
        

    # it is called after the game ended and it does not affect the game.
    # using this function you can access the result of the game.
    # scores is a map from int to int which the key is player_id and value is player_score
    def end(self, world: World, scores):
        reward = self.agent.set_reward(world=world)
        f=open("results.txt", "a")
        if(scores[world.get_me().player_id]<5):
            f.write(str(0))
            f.close()
            reward=reward-1000
        elif(scores[world.get_me().player_id]>5):
            f.write(str(1))
            f.close()
            reward=reward+1000
        else:
            f.write(str(2))
            f.close()
        self.agent.train_short_memory(np.array(self.state_old), self.prediction, reward,np.array(self.new_state),((~world.get_friend().is_alive() and ~world.get_me().is_alive())or(~world.get_second_enemy().is_alive() and ~world.get_first_enemy().is_alive())))
        self.agent.remember(np.array(self.state_old), self.prediction[0], reward, np.array(self.new_state),((~world.get_friend().is_alive() and ~world.get_me().is_alive())or(~world.get_second_enemy().is_alive() and ~world.get_first_enemy().is_alive())))
        self.agent.replay_new(self.agent.memory)
        if(self.agent.epsilon<999):
            f=open("args.txt", "w")
            f.write(str(self.agent.epsilon))
            f.close()
        reward = self.picker.set_reward(world=world)
        print("end started!")
        print("My score:", scores[world.get_me().player_id])
        # print("////////////////////units score///////////////////")
        # for i in range(9):
        #     print(self.units_points[i])
        self.picker.train_short_memory(np.array([1,1,1,1,1,1,1,1,1]), [self.picked],self.picker.set_reward(world=world),np.array(self.picked),self.agent.epsilon==999)
        self.agent.model.save('agent.h5')
        self.picker.model.save('picker.h5')
        print("picker model saved!")
