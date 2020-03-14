import random
from random import randint
import numpy as np
from model import *
from world import World
from model import BaseUnit, Map, King, Cell, Path, Player, GameConstants, TurnUpdates, \
    CastAreaSpell, CastUnitSpell, CastSpell, Unit, Spell, Message, UnitTarget, SpellType, SpellTarget, Logs

# new version!


class AI:
    def __init__(self):
        self.rows = 0
        self.cols = 0
        self.path_for_my_units = None
        self.sneaky_enemy = False
        self.enable_sneaky=True

    # this function is called in the beginning for deck picking and pre process
    def pick(self, world: World):
        print("pick started!")

        # pre process
        map = world.get_map()
        self.rows = map.row_num
        self.cols = map.col_num

        # choosing all flying units
        all_base_units = world.get_all_base_units()
        my_hand = [
            base_unit for base_unit in all_base_units if (base_unit.type_id == 0 or base_unit.type_id == 1 or base_unit.type_id == 2 or base_unit.type_id == 5 or base_unit.type_id == 6)]

        # picking the chosen hand - rest of the hand will automatically be filled with random base_units
        world.choose_hand(base_units=my_hand)
        # other pre process
        self.path_for_my_units = world.get_first_enemy().paths_from_player[0]

    # it is called every turn for doing process during the game
    def turn(self, world: World):
        def sort_func(elem):
            return elem.unit_id
        print("turn started:", world.get_current_turn())

        myself = world.get_me()
        max_ap = world.get_game_constants().max_ap

        mylast_units = np.zeros(shape=(5,), dtype=int)
        mylast_units_health = np.zeros(shape=(5,), dtype=int)
        mylast_units_cell = np.zeros(shape=(5,), dtype=int)
        mylast_units_atacking_king = np.zeros(shape=(5,), dtype=int)
        myself = world.get_me()
        myunits = [x for x in myself.units]
        myunits.sort(key=sort_func)
        s = slice(5)
        myunits = [x for x in myunits]
        myunits = myunits[s]
        for i in range(5):
            if(i < len(myunits)):
                mylast_units_health[i] = [x.hp for x in myunits][s][i]
                mylast_units_cell[i] = np.hstack(
                    [[x.cell.row, x.cell.col] for x in myunits])[s][i]
                mylast_units_atacking_king[i] = 1 if myunits[i].target_if_king is not None else 0
                mylast_units[i] = myunits[i].base_unit.type_id+1

        friendlast_units = np.zeros(shape=(5,), dtype=int)
        friendlast_units_health = np.zeros(shape=(5,), dtype=int)
        friendlast_units_cell = np.zeros(shape=(5,), dtype=int)
        friendlast_units_atacking_king = np.zeros(shape=(5,), dtype=int)
        friend = world.get_friend()
        friendunits = [x for x in friend.units]
        friendunits.sort(key=sort_func)
        s = slice(5)
        friendunits = [x for x in friendunits]
        friendunits = friendunits[s]
        for i in range(5):
            if(i < len(friendunits)):
                friendlast_units_health[i] = [x.hp for x in friendunits][s][i]
                friendlast_units_cell[i] = np.hstack(
                    [[x.cell.row, x.cell.col] for x in friendunits])[s][i]
                friendlast_units_atacking_king[i] = 1 if friendunits[i].target_if_king is not None else 0
                friendlast_units[i] = friendunits[i].base_unit.type_id

        en1last_units = np.zeros(shape=(5,), dtype=int)
        en1last_units_atacking_king = np.zeros(shape=(5,), dtype=int)
        en1last_units_health = np.zeros(shape=(5,), dtype=int)
        en1last_units_cell = np.zeros(shape=(5,), dtype=int)
        en1 = world.get_first_enemy()
        en1units = [x for x in en1.units]
        en1units.sort(key=sort_func)
        s = slice(5)
        en1units = [x for x in en1units]
        en1units = en1units[s]
        for i in range(5):
            if(i < len(en1units)):
                en1last_units_health[i] = [x.hp for x in en1units][s][i]
                en1last_units_cell[i] = np.hstack(
                    [[x.cell.row, x.cell.col] for x in en1units])[s][i]
                en1last_units_atacking_king[i] = 1 if en1units[i].target_if_king is not None else 0
                en1last_units[i] = en1units[i].base_unit.type_id

        en2last_units = np.zeros(shape=(5,), dtype=int)
        en2last_units_health = np.zeros(shape=(5,), dtype=int)
        en2last_units_atacking_king = np.zeros(shape=(5,), dtype=int)
        en2last_units_cell = np.zeros(shape=(5,), dtype=int)
        en2 = world.get_first_enemy()
        en2units = [x for x in en2.units]
        en2units.sort(key=sort_func)
        s = slice(5)
        en2units = [x for x in en2units]
        en2units = en2units[s]
        for i in range(5):
            if(i < len(en2units)):
                en2last_units_health[i] = [x.hp for x in en2units][s][i]
                en2last_units_cell[i] = np.hstack(
                    [[x.cell.row, x.cell.col] for x in en2units])[s][i]
                en2last_units_atacking_king[i] = 1 if en2units[i].target_if_king is not None else 0
                en2last_units[i] = en2units[i].base_unit.type_id
        for i in range(5):
            if(i < len(myunits)):
                mylast_units_health[i] = [x.hp for x in myunits][s][i]
                mylast_units_cell[i] = np.hstack(
                    [[x.cell.row, x.cell.col] for x in myunits])[s][i]
                mylast_units_atacking_king[i] = 1 if myunits[i].target_if_king is not None else 0
                mylast_units[i] = myunits[i].base_unit.type_id

        # play all of hand once your ap reaches maximum. if ap runs out, putUnit doesn't do anything

        for unit in myself.units:
            if((unit.base_unit.type_id == 6 or unit.base_unit.type_id == 1) and (unit.hp > (unit.base_unit.max_hp/2))):
                world.upgrade_unit_damage(unit=unit)
                world.upgrade_unit_range(unit=unit)

        f = len(world.get_shortest_path_to_cell(from_player=world.get_me(
        ), cell=world.get_king_by_id(world.get_first_enemy().player_id).center).cells)
        s = len(world.get_shortest_path_to_cell(from_player=world.get_me(
        ), cell=world.get_king_by_id(world.get_second_enemy().player_id).center).cells)

        ################################# spells ###################################
        if(s < f):
            for spell in myself.spells:
                if spell.is_area_spell():
                    if spell.target == SpellTarget.ENEMY:
                        if(len(en2units) > 0):
                            world.cast_area_spell(center=en2units[randint(
                                0, len(en2units)-1)].cell, spell=spell)
                    elif (spell.target == SpellTarget.ALLIED and spell.type_id != 4):
                        if(len(myunits) > 0):
                            world.cast_area_spell(
                                center=myunits[randint(0, len(myunits)-1)].cell, spell=spell)
                    elif (spell.target == SpellTarget.ALLIED and spell.type_id == 4):
                        for unit in myunits:
                            if((unit.base_unit.type_id == 1 or unit.base_unit.type_id == 6) and unit.hp > 5):
                                world.cast_area_spell(
                                    center=unit.cell, spell=spell)
                elif spell.is_unit_spell():
                    my_units = myself.units
                    for unit in my_units:
                        if(unit.base_unit.type_id == 6 and unit.hp > 3):
                            path = world.get_shortest_path_to_cell(from_player=world.get_me(
                            ), cell=world.get_king_by_id(world.get_first_enemy().player_id).center)
                            size = len(path.cells)
                            cell = path.cells[int((size - 1) / 2)]
                            world.cast_unit_spell(
                                unit=unit, path=path, cell=cell, spell=spell)
        else:
            for spell in myself.spells:
                if spell.is_area_spell():
                    if spell.target == SpellTarget.ENEMY:
                        if(len(en1units) > 0):
                            world.cast_area_spell(center=en1units[randint(
                                0, len(en1units)-1)].cell, spell=spell)
                    elif (spell.target == SpellTarget.ALLIED and spell.type_id != 4):
                        if(len(myunits) > 0):
                            world.cast_area_spell(
                                center=myunits[randint(0, len(myunits)-1)].cell, spell=spell)
                    elif (spell.target == SpellTarget.ALLIED and spell.type_id == 4):
                        for unit in myunits:
                            if(unit.base_unit.type_id == 1 or unit.base_unit.type_id == 6):
                                world.cast_area_spell(
                                    center=unit.cell, spell=spell)
                elif spell.is_unit_spell():
                    my_units = myself.units
                    for unit in my_units:
                        if(unit.base_unit.type_id == 6 and unit.hp > 3):
                            path = world.get_shortest_path_to_cell(from_player=world.get_me(
                            ), cell=world.get_king_by_id(world.get_second_enemy().player_id).center)
                            size = len(path.cells)
                            cell = path.cells[int((size - 1) / 2)]
                            world.cast_unit_spell(
                                unit=unit, path=path, cell=cell, spell=spell)
        ################################# spells end ###################################
        sneaky_count = 0
        sneaky = True
        if(s < f):
            for i in range(int(2*len(world.get_shortest_path_to_cell(from_player=world.get_me(), cell=world.get_king_by_id(world.get_second_enemy().player_id).center).cells)/3)):
                for unit in (world.get_shortest_path_to_cell(from_player=world.get_me(), cell=world.get_king_by_id(world.get_second_enemy().player_id).center).cells[i].units):
                    if(unit in world.get_first_enemy().units or unit in world.get_second_enemy().units):
                        sneaky = False
                        self.enable_sneaky=True
            

            for i in range(10):
                for unit in (world.get_shortest_path_to_cell(from_player=world.get_me(), cell=world.get_king_by_id(world.get_first_enemy().player_id).center).cells[i].units):
                    if(unit in world.get_first_enemy().units or unit in world.get_second_enemy().units):
                        self.sneaky_enemy=True

            

            print(sneaky_count)

            for base_unit in myself.hand:
                if(((world.get_current_turn() % 20 == 0 and (base_unit.type_id == 6 or base_unit.type_id == 1 or base_unit.type_id == 2)) or (sneaky and self.enable_sneaky and (base_unit.type_id == 6 or base_unit.type_id == 1 or base_unit.type_id == 2)) or (self.sneaky_enemy and (base_unit.type_id == 6 or base_unit.type_id == 1 or base_unit.type_id == 2)))and sneaky_count < 3):
                    sneaky_count = sneaky_count+1
                    path = world.get_shortest_path_to_cell(from_player=world.get_me(
                    ), cell=world.get_king_by_id(world.get_first_enemy().player_id).center)
                    world.put_unit(base_unit=base_unit, path=path)
                    self.sneaky_enemy = False
                    self.enable_sneaky=False
                if(base_unit.type_id == 0 or base_unit.type_id == 1 or base_unit.type_id == 2 or base_unit.type_id == 5 or base_unit.type_id == 6):
                    path = world.get_shortest_path_to_cell(from_player=world.get_me(
                    ), cell=world.get_king_by_id(world.get_second_enemy().player_id).center)
                    world.put_unit(base_unit=base_unit, path=path)

        else:
            for i in range(int(2*len(world.get_shortest_path_to_cell(from_player=world.get_me(), cell=world.get_king_by_id(world.get_first_enemy().player_id).center).cells)/3)):
                for unit in (world.get_shortest_path_to_cell(from_player=world.get_me(), cell=world.get_king_by_id(world.get_first_enemy().player_id).center).cells[i].units):
                    if(unit in world.get_first_enemy().units or unit in world.get_second_enemy().units):
                        sneaky = False
                        self.enable_sneaky=True
            for i in range(10):
                for unit in (world.get_shortest_path_to_cell(from_player=world.get_me(), cell=world.get_king_by_id(world.get_second_enemy().player_id).center).cells[i].units):
                    if(unit in world.get_first_enemy().units or unit in world.get_second_enemy().units):
                        self.sneaky_enemy=True
        
            print(sneaky_count)
            for base_unit in myself.hand:
                if(((world.get_current_turn() % 20 == 0 and (base_unit.type_id == 6 or base_unit.type_id == 1 or base_unit.type_id == 2)) or (sneaky and self.enable_sneaky and (base_unit.type_id == 6 or base_unit.type_id == 1 or base_unit.type_id == 2)) or (self.sneaky_enemy and (base_unit.type_id == 6 or base_unit.type_id == 1 or base_unit.type_id == 2))) and sneaky_count < 3):
                    sneaky_count = sneaky_count+1
                    path = world.get_shortest_path_to_cell(from_player=world.get_me(
                    ), cell=world.get_king_by_id(world.get_second_enemy().player_id).center)
                    world.put_unit(base_unit=base_unit, path=path)
                    self.sneaky_enemy = False
                    self.enable_sneaky=False
                if(base_unit.type_id == 0 or base_unit.type_id == 1 or base_unit.type_id == 2 or base_unit.type_id == 5 or base_unit.type_id == 6):
                    path = world.get_shortest_path_to_cell(from_player=world.get_me(
                    ), cell=world.get_king_by_id(world.get_first_enemy().player_id).center)
                    world.put_unit(base_unit=base_unit, path=path)
                # it is called after the game ended and it does not affect the game.
                # using this function you can access the result of the game.
                # scores is a map from int to int which the key is player_id and value is player_score

    def end(self, world: World, scores):
        print("end started!")
        print("My score:", scores[world.get_me().player_id])
