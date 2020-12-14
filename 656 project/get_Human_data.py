#import pygame
import gym
import keyboard
import math
import pickle
import numpy as np
from MC_support import *
import json
from gym import spaces
from gym.utils import seeding
import pyglet
from gym.envs.classic_control import rendering
import itertools
import operator
from collections import Counter
from ExtendedMountainCarEnv import *

def main():
    #This function handles the main interation between human and
    # the enviromwnt.
    # pygame.init()
    # gameDisplay = pygame.display.set_mode((800, 600))
    # pygame.display.set_caption('A bit Racey')
    # max_steps = 100
    # while not crashed:
    #
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             crashed = True
    #
    #         print(event)
    #
    #     pygame.display.update()
    #     clock.tick(60)
    iht_size = 4096*80
    num_tilings = 32
    num_tiles = 64
    tile_coder = ExtendedMountainCarTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles) # We records which action that the agent takes in each active tile.
    action_list = {}
    env = ExtendedMountainCarEnv()
    for i_episode in range(100):
        observation = env.reset()
        done = False
        t = 0
        while not done:
            env.render()
            if keyboard.is_pressed('a'):
                action = 0
                active_tiles = tile_coder.get_tiles(observation[0], observation[1])
                string_active_tiles = np.array2string(active_tiles)
                if string_active_tiles in action_list:
                    action_list[string_active_tiles].append(int(action))
                else:
                    action_list[string_active_tiles] = [int(action)]
            elif keyboard.is_pressed('d'):
                action = 2
                active_tiles = tile_coder.get_tiles(observation[0], observation[1])
                string_active_tiles = np.array2string(active_tiles)
                if string_active_tiles in action_list:
                    action_list[string_active_tiles].append(int(action))
                else:
                    action_list[string_active_tiles] = [int(action)]
            else:
                action =1
            observation, reward, done, info = env.step(action)
            t+=1
            if done:
                print("Episode ",i_episode)
                print(len(action_list))
                break
    env.close()
    for key, value in action_list.items():
        action_list[key]= most_common(value)
    save_obj(action_list, "action_list")

def save_obj(obj, name):
    result = json.dumps(obj)
    with open('human_demonstration_data/'+ name + '.json', 'w+') as f:
        f.write(result)



def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

def load_obj(name):
    with open('human_demonstration_data/' + name + '.json', 'rb') as f:
        return json.load(f)

main()


