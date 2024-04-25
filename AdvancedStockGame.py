import threading
import time

import gymnasium as gym
from gymnasium import spaces
from gym.wrappers import FlattenObservation
from gym.spaces.utils import unflatten
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise
import torch.nn as nn
import torch
from stable_baselines3 import A2C, DQN
import datetime
import pandas as pd
import warnings
import argparse
import os
import pathlib

class AdvancedStockGame(gym.Env):
    def __init__(self, data_file_path, window_size=1024):
        super(AdvancedStockGame, self).__init__()
        self.window_size = window_size
        self.starting_money = 1000
        self.trade_cost = 0.35
        self.force_start_index = None
        self.trade_cooldown_in_hours = 24 #hours
        self.episode_length = 25 * 6.5 * 60 #25 days

        self.data_block, self.stock_names, self.stock_timestamps = self.import_data(data_file_path)
        self.current_index = 0
        self.start_index = 0
        self.features_size = len(self.data_block[0][0])
        self.stocks_number = len(self.data_block)
        self.total_length = len(self.data_block[0])

        self.real_portfolio = np.zeros(shape=(self.stocks_number + 1,), dtype=np.float32)
        self.cool_down = np.zeros(shape=(self.stocks_number,), dtype=np.int16)
        
        
        self.action_space = gym.spaces.Discrete(self.stocks_number + 1, start=0)
        self.observation_space = spaces.Dict({
            "stock_data": spaces.Box(low=-1, high=1, shape=(self.stocks_number, self.window_size, self.features_size), dtype=np.float32),
            "portfolio": spaces.Box(low=-1, high=1, shape=(self.stocks_number,), dtype=np.int16),
            "portfolio_value": spaces.Discrete(10)
        })  

  

        self.current_observation = {
            "stock_data": np.zeros(shape=(self.stocks_number, self.window_size, self.features_size),  dtype=np.float32), 
            "portfolio": np.zeros(shape=(self.stocks_number,), dtype=np.int16),
            "portfolio_value": 0
        } 
        warnings.filterwarnings("ignore", message="It seems that your observation  is an image but its `dtype` is")
        warnings.filterwarnings("ignore", message="It seems that your observation space  is an image but the upper and lower bounds are not in")
        warnings.filterwarnings("ignore", message="The minimal resolution for an image is 36x36 for the default `CnnPolicy`")
        warnings.filterwarnings("ignore", message="We recommend you to use a symmetric and normalized Box action space")                 

    def step(self, action):
        
        reward = 0
        current_data = self.data_block[:, self.start_index + self.current_index, 0]
        portfolio_value = self.get_portfolio_value()
        if(action != self.stocks_number):
            if(self.current_observation['portfolio'][action] < 0):
                reward -= 1
                
            elif(self.current_observation['portfolio'][action] > 0):
                self.sell(action)
            else:
                self.buy(action)
        else:
            made_action = False
            for i in range(self.stocks_number):
                if(self.current_observation['portfolio'][i] != 0):
                    made_action = True
            if not made_action:
                reward -= 1

                
            
        
        done = self.roll_into_observation()

        self.roll_portfolio_forward()

        new_portfolio_value = self.get_portfolio_value()

        if new_portfolio_value <= 0:
            done = True


        reward += (new_portfolio_value - portfolio_value) / portfolio_value

        
        return self.current_observation, reward, done, False, {'data': current_data, 'time': self.stock_timestamps[self.start_index + self.current_index - 1]}

    def reset(self, seed=0):
        self.current_index = 0
        if not self.force_start_index:
            self.start_index = np.random.randint(0, self.total_length - self.episode_length)
        else:
            self.start_index = self.force_start_index
            self.force_start_index = None
        self.real_portfolio = np.zeros(shape=(self.stocks_number + 1,), dtype=np.float32)
        self.cool_down = np.zeros(shape=(self.stocks_number,), dtype=np.int16)
        self.current_observation = {
            "stock_data": np.zeros(shape=(self.stocks_number, self.window_size, self.features_size),  dtype=np.float32), 
            "portfolio": np.zeros(shape=(self.stocks_number,), dtype=np.int16),
            "portfolio_value": 0
        } 
        self.real_portfolio[-1] = self.starting_money
        self.current_observation['portfolio_value'] = self.portfolio_value_to_obs()
        for i in range(self.window_size):
            self.roll_into_observation()
        
        return self.current_observation, {'start_time' : self.stock_timestamps[self.start_index]}
    
    def buy(self, stock_index):
        trade_amount = self.real_portfolio[-1] / 3
        self.real_portfolio[-1] -= trade_amount
        self.real_portfolio[-1] -= self.trade_cost
        self.real_portfolio[stock_index] += trade_amount
        

    def sell(self, stock_index):
        self.real_portfolio[-1] += self.real_portfolio[stock_index]
        self.real_portfolio[-1] -= self.trade_cost
        self.real_portfolio[stock_index] = 0
        self.cool_down[stock_index] = self.trade_cooldown_in_hours * 60


    def roll_into_observation(self):
        for i in range(len(self.current_observation['stock_data'])):
            self.current_observation['stock_data'][i] = np.roll(self.current_observation['stock_data'][i], shift=-1, axis=0)
            self.current_observation['stock_data'][i][-1] = self.data_block[i][self.start_index + self.current_index]
        self.current_index += 1
        return self.start_index + self.current_index >= len(self.data_block[0]) or self.current_index >= self.episode_length + self.window_size
            

    def get_stock(self, stock_index):
        if(stock_index == self.stocks_number):
            return "None"
        else:
            return self.stock_names[stock_index]
        
    def portfolio_value_to_obs(self):
        value = self.get_portfolio_value()
        if value <= 1:
            value = 1
        ret_val = int(np.ceil(np.log10(value)))
        return ret_val
    
        
    def roll_portfolio_forward(self):
        for i in range(self.stocks_number):
            self.real_portfolio[i] *= (1 + self.current_observation['stock_data'][i][-1][0])
            if(self.cool_down[i] > 0):
                self.cool_down[i] -= 1
                self.current_observation['portfolio'][i] = -1
            
            else:
                self.current_observation['portfolio'][i] = np.sign(self.real_portfolio[i])
        self.current_observation['portfolio_value'] = self.portfolio_value_to_obs()
        
            

    def get_portfolio_value(self):
        value = 0
        for i in range(self.stocks_number + 1):
            value += self.real_portfolio[i]
        return value

    def import_data(self, data_file_path):
        data = pd.read_csv(data_file_path)
        date_times = np.array(data.iloc[:,0])
        data.drop(columns=['Time'], inplace=True)
        stock_names = []
        numpy_data = data.to_numpy()
        return_data = []
        stock_data = []
        numpy_data = numpy_data.transpose(1,0)
        for i, header in enumerate(data.columns):
            if(i % 2 == 0):
                stock_name = header.split("_")[1]
                stock_names.append(stock_name)
                stock_data.append(numpy_data[i][:])
            else:
                stock_data.append(numpy_data[i][:])
                return_data.append(stock_data)
                stock_data = []
        
        
        return_data = np.array(return_data)
        return_data = return_data.transpose(0,2,1)
        return return_data, stock_names, date_times
        


    def render(self, mode='human'):
        pass  # You can add code to visualize the game if desired
