from AdvancedStockGame import AdvancedStockGame
import numpy as np
from stable_baselines3 import DQN
import csv
from datetime import datetime
import argparse
import os

def add_to_csv(data):
    modelFileName = f'models/{args.name}_history.csv'
    if(not os.path.exists(modelFileName)):
        with open(modelFileName, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'Total Timesteps', 'Current Session', 'Accuracy', 'Negative %', 'Reward', 'Profit', 'Weighted Proft', 'Next', 'Next Weight'])
    with open(modelFileName, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)



def evaluate(model, env):
    accuracy = 0
    obs, info = env.reset()
    #test2023List[stock][time][feature]
    #test2023List[feature][time][stock]

    i = 0
    while True:  
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, term, info = env.step(action)
        total_reward += reward
        negative_choices += 1 if env.get_stock_change(info['data'], action) < 0 else 0

        profit += np.log10(1 + env.get_stock_change(info['data'], action))

        accuracy += 1 if int(action) == np.argmax(info['data']) else 0
        i += 1
        if done:
            break


    action, _states = model.predict(obs,deterministic=True)

    accuracy =  accuracy / (i) * 100
    negative_choices = negative_choices / (i) * 100
    total_reward = total_reward
    profit = profit
    weighted_profit = profit
    next_stock = env.get_stock(action)
    next_weight = 1
    del model
    return accuracy, negative_choices, total_reward, profit, weighted_profit, next_stock, next_weight

def pct(x):
    return str(np.round(x, decimals=2)) + '%'
def rnd(x):
    return str(np.round(x, decimals=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Builder")
    parser.add_argument('-n', '--name', type=str, help='Model Name')
    parser.add_argument('-d', '--data', type=str, help='Data File')

    args = parser.parse_args()
    if(args.name is None):
        print('Please provide a name for the model')
        exit()
    modelFileName = f'models/{args.name}.zip'
    if not os.path.exists(modelFileName):
        print('Model Does not Exist')
        exit()
    dataFile = 'full_data.csv'
    env = AdvancedStockGame(dataFile)
    model = DQN.load(modelFileName, env=env, device='cuda:0')
    accuracy, negative_choices, total_reward, profit, weighted_profit, next_stock, next_weight = evaluate(model, env)
    now = datetime.now().strftime("%H:%M")
    with open(modelFileName + '_timesteps.txt', 'r') as file:
        data = file.read() 
        total_trainsteps = int(data)
    current_session = 0
    print(f"{'Total Timesteps':<20}{'Accuracy':<20}{'Negative %':<20}{'Reward':<20}{'Profit':<20}{'Weighted Proft':<20}{'Next':<20}{'Next Weight':<20}")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(f'''{total_trainsteps:<20}{pct(accuracy):<20}{pct(negative_choices):<20}{rnd(total_reward):<20}{"e"+rnd(profit):<20}{"e"+rnd(weighted_profit):<20}{next_stock:<20}{next_weight:<20}''', end="\n")
    
    data = [now, total_trainsteps, current_session, accuracy, negative_choices, total_reward, profit, weighted_profit, next_stock, next_weight]
