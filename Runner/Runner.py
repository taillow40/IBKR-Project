from ibapi.client import EClient
from ibapi.common import TickerId
from ibapi.wrapper import EWrapper  
from ibapi.contract import Contract
from ibapi.order import Order

import time

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas_market_calendars as mcal
from stable_baselines3 import PPO
from stable_baselines3 import A2C, DQN
import datetime
import pandas as pd
import warnings


import threading
import time

import numpy as np


import logging
import datetime
import pandas
import warnings
import os
import pathlib

stock500 = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BK', 'BBWI', 'BAX', 'BDX', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BX', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BLDR', 'BG', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'COR', 'CNC', 'CNP', 'CF', 'CHRW', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 
'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DAY', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'LLY', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'EL', 'ETSY', 'EG', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'FLT', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GEHC', 'GEN', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VTRS', 'V', 'VMC', 'WRB', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WRK', 'WY', 'WHR', 'WMB', 'WTW', 'GWW', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']

stock3 = ['NVDA', 'TSLA', 'AAPL']

stockList = stock3

trades_today = 0
	
class IBapi(EWrapper, EClient):
	def __init__(self, model_name):
		EClient.__init__(self, self)
		self.data = [] #Initialize variable to store candle
		self.last_close_prices = {}
		self.positions = {}
		self.download_index = 0
		self.data_info = ''
		self.environment = AdvancedStockGame(stockList)
		self.df = pd.DataFrame()
		self.cash = 1000
		self.portfolio_value_log = 0
		self.df_non_percent = pd.DataFrame()
		self.simple_positions = []
		self.model = PPO.load('../models/' + model_name, env=self.environment, device='cuda:1')
		self.next_placement = None 
		self.decision = None

	def begin_action(self, decision=None):
		self.df = pd.DataFrame()
		self.download_index = 0
		self.decision = decision
		self.last_close_prices = {}
		self.next_placement = None
		self.reqGlobalCancel()
		self.download_account_summary()

	def download_account_summary(self):
		self.reqAccountSummary(101,'All', 'NetLiquidation, TotalCashValue')	
	
	def accountSummary(self, reqId, account, tag, value, currency):
		if(tag == 'NetLiquidation'):
			self.portfolio_value_log = int(np.ceil(np.log10(float(value))))
		elif(tag == 'TotalCashValue'):
			self.cash = float(value)
			#self.cash = 1000
			
	
	def accountSummaryEnd(self, reqId: int):
		#print('Cash:', self.cash)
		#print('Portfolio Value:', self.portfolio_value_log)
		self.download_portfolio()


	def download_portfolio(self):
		self.positions = {}
		self.reqPositions()
		
		
	
		#print(bar)
	
	def position(self, account, contract, pos, avgCost):
		self.positions[contract.symbol] = pos
		super(IBapi, self).position(account, contract, pos, avgCost)

	def positionEnd(self):
		self.simple_positions = [0] * len(stockList)
		recent_sell = self.import_last_sell()
		for i in range(len(self.simple_positions)):
			if(stockList[i] in self.positions and self.positions[stockList[i]] > 0):
				self.simple_positions[i] = 1
			if(recent_sell[i] != 0):
				self.simple_positions[i] = -1
			
		#print("Simple Positions:", self.simple_positions)
		
		self.req_and_save_data(stockList[self.download_index])
		self.cancelPositions()

	def req_and_save_data(self, symbol):
		contract = Contract()
		contract.symbol = symbol
		contract.secType = 'STK'
		contract.exchange = 'SMART'
		contract.currency = 'USD'
		duration = int(self.environment.window_size / 6.5 / 60 + 2)
		self.reqHistoricalData(1, contract, '', f'{duration} D', '1 min', 'TRADES', 1, 1, False, [])

	def historicalData(self, reqId, bar):
		self.data.append([bar.date, bar.close, bar.volume])
	
	def historicalDataEnd(self, reqId: int, start, end):
	 	#super().historicalDataEnd(reqId, start, end)
		if(self.data):
			for i in range(len(stockList)):
				self.last_close_prices[stockList[i]] = self.data[-1][1]
			symbol = stockList[self.download_index]
			new_df = pandas.DataFrame(self.data, columns=['Time', 'Close_' + symbol, 'Volume_'+symbol])
			self.df = self.combine_dfs(self.df, new_df)
			self.data = []
			if(self.download_index < len(stockList) - 1):
				self.download_index += 1
				self.req_and_save_data(stockList[self.download_index])
			else:
				self.finish_data()
	

	

	def finish_data(self):
		self.df.ffill(inplace=True)
		self.df.sort_index(inplace=True)
		#print("Last Date:", self.df['Time'].iloc[-1])
		percent_change_df = self.df.iloc[:, 1:].pct_change(fill_method=None)
		percent_change_df.clip(-1, 1, inplace=True)
		result_df = pd.concat([self.df.iloc[:, :1], percent_change_df], axis=1)
		percent_change_df = result_df.iloc[1:]  
		final_df = percent_change_df.fillna(0)
		self.df_non_percent = self.df
		self.df = final_df
		#print(self.df.shape)
		self.make_decision()
										
	
	
	def make_decision(self):
		self.environment.set_data(self.df)
		self.environment.set_portfolio(self.simple_positions)
		self.environment.set_portfolio_value(self.portfolio_value_log)
		obs, _ = self.environment.reset()
		action, _ = self.model.predict(obs, deterministic=True)
		if(self.decision != None):
			action = stockList.index(self.decision)
		#print("Decision:", stockList[action])
		self.make_action(action, obs['portfolio'])

	def make_action(self, action, simple_portfolio):
		if(action == len(stockList)):
			print("-" * datetime.datetime.now().minute, end="\r")
			return
		if(trades_today >= 3):
			print("Flagged, Too many trades today")
			return
		stock = stockList[action]
		if(simple_portfolio[action] == 0):
			trades_today += 1
			print("buying", stock)
			self.buy(stock)
		elif(simple_portfolio[action] == 1):
			print("selling", stock)
			trades_today += 1
			self.sell(stock)
		else:
			print("Invalid Action - Attempted to buy a recently sold stock")
	

	def buy(self, stock):
		#buy stock
		price = self.last_close_prices[stock]
		shares = int(np.floor(self.cash / len(stockList) / price))
		print("\nBuying " + str(shares) + " " + stock + " at " + str(price) + " at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		order = Order()
		order.action = "BUY"
		order.orderType = "LMT"
		order.totalQuantity = shares
		order.PostToAts = 0
		order.optOutSmartRouting = True
		#order.cashQuantity = self.cash / len(stockList)
		order.lmtPrice = price + 0.05

		contract = Contract()
		contract.symbol = stock
		contract.secType = 'STK'
		contract.exchange = 'NYSE'
		contract.Exchange = 'NYSE'
		contract.primaryExchange = 'NYSE'
		contract.currency = 'USD'
		self.next_placement = {
			'order': order,
			'contract': contract
		}
		self.reqIds(-1)
		pass

	def sell(self, stock):
		#sell stock
		price = self.last_close_prices[stock]
		shares = self.positions[stock]
		print("\nSelling " + str(shares) + " " + stock + " at " + str(price) + " at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		order = Order()
		order.action = "SELL"
		order.orderType = "LMT"
		order.PostToAts = 0
		order.optOutSmartRouting = True
		order.totalQuantity = abs(self.positions[stock])
		order.lmtPrice = self.last_close_prices[stock] - 0.05

		contract = Contract()
		contract.symbol = stock
		contract.secType = 'STK'
		contract.exchange = 'NYSE'
		contract.Exchange = 'NYSE'
		contract.primaryExchange = 'NYSE'
		contract.currency = 'USD'
		self.next_placement = {
			'order': order,
			'contract': contract
		}
		self.reqIds(-1)
		pass

	def nextValidId(self, orderId: int):
		if self.next_placement:
			self.placeOrder(orderId, self.next_placement['contract'], self.next_placement['order'])
			self.next_placement = None


	

	def combine_dfs(self, df1, df2):
		matching_cols = set(df1.columns) & set(df2.columns)
		if not matching_cols:
			combined_df = pd.concat([df1, df2], axis=1, join='outer')
		else:
			combined_df = pd.concat([df1, df2], axis=0)
		combined_df = combined_df.loc[~combined_df.index.duplicated(keep='first')]
		return combined_df
	
	def import_last_sell(self):
		df = pd.read_csv('last_sell.csv')
		recent_sell = [0] * len(stockList)
		for i in range(len(stockList)):
			if(self.recent_date(df[stockList[i]][0])):
				recent_sell[i] = 1
		
		return recent_sell
	
	def recent_date(self, date):
		nyse = mcal.get_calendar('NYSE')
		now = datetime.datetime.now()
		date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
		schedule = nyse.schedule(end_date=datetime.datetime.strftime(now, '%Y-%m-%d'), start_date=datetime.datetime.strftime(date, '%Y-%m-%d'))
		return len(schedule) < 4
		




class AdvancedStockGame(gym.Env):
	def __init__(self, stocksList):
		super(AdvancedStockGame, self).__init__()
		self.window_size = 4096
		self.starting_money = 1000
		self.trade_cost = 0.35
		self.trade_cooldown_in_hours = 24 #hours
		self.episode_length = 25 * 6.5 * 60 #25 days
		self.data_block = []
		self.portfolio_value = 0
		self.portfolio = []


		self.features_size = 2
		self.stocks_number = len(stocksList)


		
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
		print("ERROR: DO NOT STEP THIS ENVIRONMENT")
		return self.current_observation, 0, True, False, {}

	def reset(self, seed=0):
		self.current_observation = {
			"stock_data": np.zeros(shape=(self.stocks_number, self.window_size, self.features_size),  dtype=np.float32), 
			"portfolio": np.zeros(shape=(self.stocks_number,), dtype=np.int16),
			"portfolio_value": 0
		} 
		self.current_observation['portfolio_value'] = self.portfolio_value
		self.current_observation['stock_data'] = self.data_block
		self.current_observation['portfolio'] = self.portfolio
		return self.current_observation, {}
	


	def set_data(self, df):
		self.data_block = self.import_data(df)
		
	def set_portfolio(self, portfolio):
		self.portfolio = portfolio
	
	def set_portfolio_value(self, value):
		self.portfolio_value = value
		

	def import_data(self, df):		
		date_times = np.array(df.iloc[:,0])
		df.drop(columns=['Time'], inplace=True)
		stock_names = []
		numpy_data = df.to_numpy()
		return_data = []
		stock_data = []
		numpy_data = numpy_data.transpose(1,0)
		for i, header in enumerate(df.columns):
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
		return_data = return_data[:,:self.window_size,:]
		return_data = return_data.astype(np.float32)
		return return_data
		


	def render(self, mode='human'):
		pass  # You can add code to visualize the game if desired

def is_market_open():
    nyse = mcal.get_calendar('NYSE')
    today = pd.Timestamp.now().normalize()
    day = nyse.valid_days(start_date=today, end_date=today).size > 0
    return datetime.datetime.now().time() < datetime.datetime.strptime("16:00", "%H:%M").time() and datetime.datetime.now().time() > datetime.datetime.strptime("09:30", "%H:%M").time() and day

def listen_for_exit_command():
    global currently_running
    while True:
        try:
            exit_command = input("Type 'exit' to stop\n")
            if exit_command.lower() == "exit":
                print("Sent Stop Command")
                currently_running = False
                break
        except EOFError:
            print("Canceling Exit Thread")
            break



def run_loop():
	try:
		app.run()
	except Exception as e:
		print(e)
		app.disconnect()

logging.getLogger().setLevel(logging.INFO)

#model_name = 'PPOBig_23101440'
model_name = 'BACSloew4096_50211600'
app = IBapi(f'{model_name.split("_")[0]}/{model_name}')
app.connect('127.0.0.1', 7496, 123) #Live
#app.connect('127.0.0.1', 7497, 123)	#Paper

#Start the socket in a thread

api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()
time.sleep(1)

listener_thread = threading.Thread(target=listen_for_exit_command)
listener_thread.start()

print("Launching")

currently_running = True
while True:
	try:
		if(not currently_running):
			print('Stopping...')
			app.disconnect()
			break
		if(is_market_open()):
			app.begin_action()
			time.sleep(60)
		else:
			print("Market Closed..." + datetime.datetime.strftime(datetime.datetime.now(), "%H:%M"), end="\r")
			trades_today = 0
			time.sleep(60)
		
	except EOFError:
		print("Canceling Exit Thread")
		break


