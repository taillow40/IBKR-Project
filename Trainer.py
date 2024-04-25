import numpy as np
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch
import argparse
import os
import signal
import sys
import contextlib
import threading
import csv

from AdvancedStockGame import AdvancedStockGame
from AdvancedStockGameTD3 import AdvancedStockGameTD3

currently_running = True

def make_stock_env(data_file_path):
    def _init():
        env = AdvancedStockGame(data_file_path, window_size=4096)
        return env
    return _init

def listen_for_exit_command():
    global currently_running
    while True:
        try:
            exit_command = input("Type 'exit' to stop\n")
            if exit_command.lower() == "exit":
                print("Stopping...")
                currently_running = False
                model.save("backup")
                print("Backed up Save")
                break
        except EOFError:
            print("Canceling Exit Thread")
            break


def save_model():
    model.save(modelFileName)
    model.save(f'models/{args.name}/{args.name}_{model.num_timesteps}.zip')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model Builder")
    parser.add_argument('-n', '--name', type=str, help='Model Name')


    args = parser.parse_args()
    if(args.name is None):
        print('Please provide a name for the model')
        exit()
    modelFileName = f'models/{args.name}/{args.name}.zip'
    if not os.path.exists(modelFileName):
        print('Model Does not Exist')
        exit()
    
    listener_thread = threading.Thread(target=listen_for_exit_command)
    listener_thread.start()
    
    num_envs = 8  
    env_fns = [make_stock_env('full_data.csv') for _ in range(num_envs)]
    
    vec_env = SubprocVecEnv(env_fns)
    model = PPO.load(modelFileName, env=vec_env, device='cuda:0')
    
   # model = DQN.load(modelFileName, env=StockGame(historicalList), device=f'cuda:{args.gpu}')
    
    saved_on_exit = False
    train_batch = 1_000_000
    max_timesteps = 100_000_000_000
    total_trainsteps = 0
    current_session = 0
    try:
        print("Started Training,", modelFileName)
        print("\n\n")
        while(model.num_timesteps < max_timesteps):
            model.learn(total_timesteps=train_batch, reset_num_timesteps=False)
            
            
            save_model()
            print("Saved... Total Timesteps:", model.num_timesteps)


            if not currently_running:
                vec_env.close()
                break

    except KeyboardInterrupt:
        print("\n")
        if(not saved_on_exit):
            save_model()
            print("Saved model after interrupt")
            saved_on_exit = True
        vec_env.close()
        exit()
    except Exception as e:  
        print("\n")
        print(e)
        if(not saved_on_exit):
            save_model()
            print("Saved model after error")
            saved_on_exit = True
        exit()
    finally:
        print("\n")
        if(not saved_on_exit):
            save_model()
            print("Saved in finally")
            saved_on_exit = True
        listener_thread.join()
        vec_env.close()
        print("Exiting")