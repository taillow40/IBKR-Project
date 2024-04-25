from ibapi.client import EClient
from ibapi.wrapper import EWrapper  
from ibapi.contract import Contract


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

from AdvancedStockGame import AdvancedStockGame



    


class CustomLSTMNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomLSTMNetwork, self).__init__(observation_space, features_dim)


        stock_space = observation_space.spaces['stock_data']
        
        self.num_stocks = stock_space.shape[0]
        self.seq_length = stock_space.shape[1]
        self.n_features = stock_space.shape[2]

        portfolio_space = observation_space.spaces['portfolio']
        self.portfolio_dim = portfolio_space.shape[0]
        
        stock_layer_size = 1024

        portfolio_layer_size = 512 
        # LSTM layer
        self.lstm = nn.LSTM(dropout=0.4, input_size=self.n_features * self.num_stocks, num_layers=4, hidden_size=stock_layer_size, batch_first=True)

        self.portfolio_layer = nn.Linear(self.portfolio_dim, portfolio_layer_size)  

        self.final_layer = nn.Linear(stock_layer_size + portfolio_layer_size, features_dim)  
        
    def forward(self, observations):
        main_input = observations['stock_data'].view(-1, self.seq_length, self.num_stocks * self.n_features)
        lstm_out, _ = self.lstm(main_input)
        lstm_out = lstm_out[:, -1, :] 
        
        portfolio_input = observations['portfolio']
        portfolio_out = self.portfolio_layer(portfolio_input)  
        
        combined = torch.cat((lstm_out, portfolio_out), dim=1)  
        return self.final_layer(combined)
        

def new_model(data_file_path):
    policy_kwargs = dict (
        features_extractor_class=CustomLSTMNetwork,
        features_extractor_kwargs=dict(features_dim=500)
    )
    model = A2C(
        policy='MultiInputPolicy', 
        env=AdvancedStockGame(data_file_path),  
        n_steps=5,
        gamma=1,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=2,
        use_rms_prop=True,
        use_sde=False,
        sde_sample_freq=-1,
        normalize_advantage=False,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=40,
        device="cuda",
        _init_setup_model=True
    )
    return model

env = AdvancedStockGame("full_data.csv")
env.reset()
pass