# encoding: utf-8

__author__ = 'Diogo Souto'
__email__  = 'diogodusouto.at.gmail.coms'


##################################################################################
##..............................................................................##
##..............................................................................##
##............................. IMPORTING MODULES...............................##
##..............................................................................##
##..............................................................................##
##################################################################################

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from scipy import stats


import heapq
import pandas as pd
import glob as glob
import os

from datetime import datetime, timedelta
import mplfinance as mpf
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

##################################################################################
##..............................................................................##
##..............................................................................##
##............................. LOADING FILES	................................##
##..............................................................................##
##..............................................................................##
##################################################################################


# Loading the list of stocks to be studied
list_of_stoks = np.loadtxt('/home/diogo/Dropbox/finances/data_input/IBRA_list.txt',dtype=str).T[:]
#list_of_stoks = np.loadtxt('/home/diogo/Dropbox/finances/data_input/lista_ativos_joebiden.txt',dtype=str).T[:50]

# Loading tables as a pandas DataFrame
adj_prices = pd.read_csv('data_output/adj_prices_IBRA.csv', sep="\t"); adj_prices = adj_prices.set_index(pd.DatetimeIndex(adj_prices['Date'])); adj_prices = adj_prices.drop(['Date'], axis=1)
Open = pd.read_csv('data_output/open_IBRA.csv', sep="\t"); Open = Open.set_index(pd.DatetimeIndex(Open['Date'])); Open = Open.drop(['Date'], axis=1)
Low = pd.read_csv('data_output/low_IBRA.csv', sep="\t"); Low = Low.set_index(pd.DatetimeIndex(Low['Date'])); Low = Low.drop(['Date'], axis=1)
High = pd.read_csv('data_output/high_IBRA.csv', sep="\t"); High = High.set_index(pd.DatetimeIndex(High['Date'])); High = High.drop(['Date'], axis=1)
Close = pd.read_csv('data_output/close_IBRA.csv', sep="\t"); Close = Close.set_index(pd.DatetimeIndex(Close['Date'])); Close = Close.drop(['Date'], axis=1)
Volume = pd.read_csv('data_output/volumes_IBRA.csv', sep="\t"); Volume = Volume.set_index(pd.DatetimeIndex(Volume['Date'])); Volume = Volume.drop(['Date'], axis=1)

#adj_prices_aux = pd.read_csv('adj_prices_sobesobe_S&P.csv', sep="\t"); adj_prices_aux = adj_prices_aux.set_index(pd.DatetimeIndex(adj_prices_aux['Date'])); adj_prices_aux = adj_prices_aux.drop(['Date'], axis=1)
#High = pd.read_csv('high_sobesobe_S&P.csv', sep="\t"); High = High.set_index(pd.DatetimeIndex(High['Date'])); High = High.drop(['Date'], axis=1)
#month_AdjClose = pd.read_csv('month_AdjClose_sobesobe_S&P.csv', sep="\t"); month_AdjClose = month_AdjClose.set_index(pd.DatetimeIndex(month_AdjClose['Unnamed: 0'])); month_AdjClose = month_AdjClose.drop(['Unnamed: 0'], axis=1)
#month_High = pd.read_csv('month_High_sobesobe_S&P.csv', sep="\t"); month_High = month_High.set_index(pd.DatetimeIndex(month_High['Unnamed: 0'])); month_High = month_High.drop(['Unnamed: 0'], axis=1)
#month_Volume = pd.read_csv('month_Volume_sobesobe_S&P.csv', sep="\t"); month_Volume = month_Volume.set_index(pd.DatetimeIndex(month_Volume['Unnamed: 0'])); month_Volume = month_Volume.drop(['Unnamed: 0'], axis=1)


#adj_prices_aux = adj_prices_aux.drop(['TSLA','MRNA'], axis=1)
#High = High.drop(['TSLA','MRNA'], axis=1)
#month_AdjClose = month_AdjClose.drop(['TSLA','MRNA'], axis=1)
#month_High = month_High.drop(['TSLA','MRNA'], axis=1)
#month_Volume = month_Volume.drop(['TSLA','MRNA'], axis=1)

##################################################################################
##..............................................................................##
##..............................................................................##
##...........................	DEFINING CONSTANTS	............................##
##..............................................................................##
##..............................................................................##
##################################################################################

# Defining the time in years to be analized untill today.
total_years = 5
shif = 0
#start_date = '2021-12-27' #  datetime.now() - timedelta(days=365*total_years)- timedelta(days=shif);
#end_date   = '2022-01-31' #  datetime.now() - timedelta(days=shif);
period = '180d'
interval = '1d'

##################################################################################
##..............................................................................##
##..............................................................................##
##..............................	PROGRAM CORE 	............................##
##..............................................................................##
##..............................................................................##
##################################################################################


'''
	This is the main function to control the code


	Inputed Parameters
	----------

	Returns
	-------

	Notes
	-------

	'''
res = []
def run():

	# Calling gambit:
	MC_aux = []

	for k in range(1,10):
	    print (k)	
	    c,var_minima,gain,loss,max_trades,capital = 2,2.0/100.,0,-5,2,10000
	    #adj_prices,Open,Low,High,Close,Volume = get_data(list_of_stoks,0,period,interval,True)
	    gambit, MC = trade_Gambit(adj_prices,Open,Low,High,Close,Volume,c,var_minima,gain,loss,max_trades,capital)
	    data_resume = plot_rentabilidade('Gambit_IBRA_500_',max_trades,gambit['Rent'],gambit['Rent_reinvest'],\
	0,0,0,gambit['Rent'].index,list_of_stoks,gambit['Pct_change'],capital,\
	'Gambit com maximo de '+str(max_trades)+" trades por dia")
	    MC_aux.append(MC)
	#print (np.mean(np.array(res)))
	df_MC = pd.DataFrame(np.array(MC_aux).T, index = gambit['Rent'].index)
	#print (df_MC)
	df_MC.to_csv("data_output/Gambit_200_MC_1000_10.csv",sep='\t',float_format='%.2f')
	monte_carlo(df_MC)
	return df_MC

##################################################################################
##..............................................................................##
##..............................................................................##
##..............................	FUNCTIONS	................................##
##..............................................................................##
##..............................................................................##
##################################################################################

	'''

	'''
	
def monte_carlo(df):

	ax1 = plt.figure(1)
	fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(8,6))
	plt.subplots_adjust(left  = 0.1,right = 0.94,bottom = 0.09,top = 0.95,wspace = 0.2,hspace = 0.2)

	df_min = np.in1d((df.values[-1]),min(df.values[-1]))
	df_min = np.where(df_min)[0][0]
	df_max = np.in1d((df.values[-1]),max(df.values[-1]))
	df_max = np.where(df_max)[0][0]
	df_median = np.in1d((df.values[-1]),np.median(df.values[-1]))
	df_median = np.where(df_median)[0][0]
	
	ax1 = plt.subplot2grid((1,1), (0, 0))

	plt.plot(df.index,df.T.min().values,'-',color='r',lw=3,alpha=1,zorder=5)	
	plt.plot(df.index,df.T.max().values,'-',color='g',lw=3,alpha=1,zorder=5)	
	plt.plot(df.index,df.T.median().values,'-',color='b',lw=3,alpha=1,zorder=5)

	plt.legend(('Lowest profit : '+str(np.around(100*(df.values.T[df_min][-1]/10000 -1),2))+ ' %',
				'Highest profit :'+str(np.around(100*(df.values.T[df_max][-1]/10000 -1),2))+ ' %',
				'Median profit : '+str(np.around(100*(df.values.T[df_median][-1]/10000 -1),2))+ ' $\pm$ '+str(np.around(100*(df.values[-1].std()/10000),2))+' %'),
				fontsize=8, loc='upper left', scatterpoints=1, markerscale=1,frameon=False)
	
	
#	plt.plot(df.index,df.values.T[df_median],'-',color='k',lw=4,alpha=0.8)	
#	plt.plot(df.index,df.values.T[df_median],'-',color='b',lw=3,alpha=1)

	plt.plot(df.index,df.values.T[df_min],'-',color='lightsalmon',lw=3,alpha=1)
	plt.plot(df.index,df.values.T[df_max],'-',color='limegreen',lw=3,alpha=1)
	
	plt.plot(df.index,df.values.T[df_min],'-',color='k',lw=4,alpha=0.5,zorder=-5)
	plt.plot(df.index,df.T.min().values,'-',color='k',lw=4,alpha=0.5,zorder=-5)
	plt.plot(df.index,df.values.T[df_max],'-',color='k',lw=4,alpha=0.5,zorder=-5)	
	plt.plot(df.index,df.T.max().values,'-',color='k',lw=4,alpha=0.5,zorder=-5)		
	plt.plot(df.index,df.T.median().values,'-',color='k',lw=4,alpha=0.5,zorder=-5)
				
	plt.plot(df.index,df.values,'-',color='k',lw=1,alpha=0.2,zorder=-10)

	# Setting box
	#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	plt.tick_params(axis='x',which='minor',top=True,direction='in')
	plt.tick_params(axis='y',which='minor',right=True,direction='in')
	plt.tick_params(axis='x',which='major',top=True,direction='in')
	plt.tick_params(axis='y',which='major',right=True,direction='in')
	plt.minorticks_on()
	plt.tick_params(axis='both', left=True, top=True, right=True, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
	#ax1.set_yticks([ax1.xaxis.set_major_formatter], minor=True)
	#ax1.xaxis.grid(True, which='minor')
	plt.grid(":",linewidth=1,color='gray',alpha=0.3)

	plt.xlabel("Months",fontsize=12)
	plt.ylabel("Total profit",fontsize=12)

	plt.ylim(000,30000)
#	plt.ylim(min(prices) - .2*min(prices),max(prices) + .2*max(prices))
	plt.xticks(fontsize=10)
	plt.yticks(fontsize=10)

	#plt.title(str(title))

	# Saving the figure.
	plt.savefig('figures/MC_semanal_2.jpg',dpi=500)
	plt.savefig('figures/MC_semanal_2.pdf',dpi=1000)	
	plt.clf()	 
	return 	
	
def trade_Gambit(adj_prices,O,L,H,C,vol,c,var_minima,gain,loss,max_trades,capital):

	# Defining auxiliary lists to append the data
	aux = []
	aux_alldata = []
	stock_ind = []
	log = []
	# Looping on each day
	for i in range(c,len(C)):

		# Looping on each stock from the list
		for j in range(len(adj_prices.keys())):

			# Define the arguments or operators to be analized in the trading
			arg_a = L.values[i].T[j]# Current lowest value for certain day for a certain stock
			arg_b = min(L.values[i-c:i].T[j]) # The minimum value from c candles behind for certain day for certain stock
			arg_c = C.values[i].T[j]  # Current close value for certain day for a certain stock

			# Main trading condition
			if arg_a < (1-var_minima)*arg_b:

				# Saving the long or short position if the condition is satisfied
				buy = 100*((arg_c/((1-var_minima)*arg_b)) -1)

				# Appending the list of stock indexes satisfying the condition
				stock_ind.append(j)

		# If there is a tradeble condition for at least one stock on this day:
		if len(stock_ind)>0:

			# Defining the lowest profit for the trading day
			#buy = [100*((C.values[i].T[g]/((1-var_minima)*min(L.values[i-c:i].T[g]))) -1) for g in stock_ind]
			#buy = max(buy)
						
			# Obtaining a x random trade choice for n trades avaiable on this day
			ind = np.random.choice(stock_ind,max_trades)
	
			# Removing duplicities on indexes
			ind = np.unique(ind)

			# Defining the random traded position from the n trades avaiable on this day
			buy = [100*((C.values[i].T[g]/((1-var_minima)*min(L.values[i-c:i].T[g]))) -1) for g in ind]
			#print (buy)

			
			# Computing the mean value of the closed positions
			buy = np.mean(buy)

			# Condition to simulate loss and appending the daily return
			if buy > loss:
				#print ("Stocks traded on :"+str(C.index[i])+" : ",list(C.keys()[ind]), "Daily return = " +str(np.around(buy,2)))
				aux.append(buy)
				log.append(["Stocks traded on :"+str(C.index[i])+" : ",list(C.keys()[ind]), "Daily return = " +str(np.around(buy,2))])
			else:
				#print ("Stocks traded on :"+str(C.index[i])+" : ",list(C.keys()[ind]), "Daily return (LOSS) = "+str(loss))
				aux.append(loss)
				log.append(["Stocks traded on :"+str(C.index[i])+" : ",list(C.keys()[ind]), "Daily return (LOSS) = "+str(loss)])

		# If there is not a tradeble condition for at least one stock on this day:
		else: aux.append(np.nan)

		# Cleaning the index stock_ind for a new day simulation
		stock_ind = []

	# Saving the log file
	#np.savetxt('data_output/Logfile_trade_Gambit.txt',log,fmt='%s')

	# Converting the percentual return into a pandas DataFrame
	df = pd.DataFrame(np.array(aux),columns=['Pct_change'],index=C.index[c:])

	# Removing the NaNs from the DataFrame
	df = df[df['Pct_change']>-100]

	# Computing the total returns over the inputed data
	returns = return_capital(capital, df.values)

	final_soma = (100*(returns[0][-1]/capital -1))
	final_acc  = (100*(returns[1][-1]/capital -1))

	# Adding returns into a DataFrame
	df["Rent"] = returns[0]; df["Rent_reinvest"] = returns[1]; #df["IBOV"] = return_capital(capital, adj_prices['^BVSP'].values[2:])

	return df, returns[0]


##################################################################################
##################################################################################


	'''
	This function computes the sum and the accumulative sum for a certain list

	Inputed Parameters
	----------
	capital: float
		Input the total amount of capital

	data: list
		A list or series number

	Returns
	-------
	soma: list
	The resulting list of the sum of the data

	accumalative: np.array()
		The resulting array of the cumulative sum of the data

	Notes
	-------


	'''
def return_capital(capital, data):

	accumalative = []
	soma = capital + capital*np.cumsum(data)*0.01

	for i in range(len(data)):

		acc_capital = capital + capital*data[i]*0.01
		capital = acc_capital
		accumalative.append(acc_capital)

	return soma,np.array(accumalative)

##################################################################################
##################################################################################


def Percent(Shift_data,Price):

	'''
	This function computes the percentage change of a stock in a certain period


	Inputed Parameters
	----------
	Shift_data: Integer
		Include a shift in the data to be computed the percentage. Example: if you choose 2,
		the percentage will be computed based on the last and the second last numbers.
		If you choose 3, the change percentage will be computed based on the last 3 data.

	Price: pandas.core.frame.DataFrame
		The dataframe with the stock price data. Example: data['Adjusted Close']


	Returns
	-------
	df_per_1month: pandas.core.frame.DataFrame
		The dataframe with the percentage change compared to the previous data.

	df_per_xmonth: pandas.core.frame.DataFrame
		The dataframe with the percentage change compared to the integer defined
		in the Shift_data inputed parameter.


	Notes
	-------


	'''
	per = 100*(np.array(Price)[1:]/np.array(Price)[:-1] - 1)
	df_per_1month = pd.DataFrame(np.array(per),columns=Price.keys(),index=Price.index[1:])

	per = 100*(np.array(Price)[Shift_data:]/np.array(Price)[:-Shift_data] - 1)
	df_per_xmonth = pd.DataFrame(np.array(per),columns=Price.keys(),index=Price.index[Shift_data:])

	return df_per_1month,df_per_xmonth

##################################################################################
##################################################################################

	'''
	This function plot the cumulative sum for certain list of data


	Inputed Parameters
	----------

	prices, prices2, prices3, prices4, prices5: list
		A list or series number


	Returns
	-------
	accumalative: np.array()
		The resulting array of the cumulative sum of the data

	Notes
	-------


	'''

def plot_rentabilidade(strategy,j,prices,prices2,prices3,prices4,prices5,index,stocks,step_percent,capital,title):

	etfs = yf.download(["^BVSP","SMAL11.SA","^GSPC","SLY"],period=str(period), interval =str(interval),progress=False)
#	etfs = yf.download(["^BVSP","SMAL11.SA","^GSPC","SLY"], start = start_date, end = end_date,progress=False)
	#print (etfs)

	# define the Date (x.index) as a datetime pandas array
	index.index = pd.DatetimeIndex(index)
	aux_date2num = index.index.map(mdates.date2num)

	#print (aux_date2num)

	# Define the figure size.
	ax1 = plt.figure(1)
	fig, ax1 = plt.subplots(nrows=2, ncols=1,figsize=(8,6))
	plt.subplots_adjust(left  = 0.1,right = 0.94,bottom = 0.09,top = 0.95,wspace = 0.2,hspace = 0.2)

	ax1 = plt.subplot2grid((1,1), (0, 0))

	plt.plot(index,prices,'.-',color='k',lw=2)
	plt.plot(index,prices2,'.-',color='r',lw=2)
#	plt.plot(index,prices3,'.-',color='r',lw=2)
#	plt.plot(index,prices4,'.-',color='g',lw=2)
	plt.plot(etfs.index,(etfs["Adj Close"].values/etfs["Adj Close"].values[1])*capital,'.-',lw=1,alpha = 0.6)
#	plt.plot(index,np.array(prices5))

	plt.legend((r"Rentabilidade sem reinvestimento: "+str(np.around(prices[-1]/(capital/100) -100,2))+\
	" %; Média por periodo: "+str(np.around(np.mean(step_percent),2))+ ' $\pm$ '+str(np.around(np.std(step_percent),2))+' %; '\
	'Taxa de acerto = '+str(np.around(len(step_percent[step_percent>0])/len(step_percent)*100,2))+' % \n Dias com pelo menos 1 trade = '+str(len(step_percent)),\
	"Rentabilidade com reinvestimento: "+str(np.around(prices2[-1]/(capital/100) -100,2)) + "%",\
	"S&P 600","SMAL11","S&P 500","IBOV"),fontsize=8, loc='upper left', \
	scatterpoints=1, markerscale=1,frameon=False)
	#print (j, np.around(prices[-1]/(capital/100) -100,2), np.around(np.mean(step_percent),2), np.around(len(step_percent[step_percent>0])/len(step_percent)*100,2), len(step_percent))
	res.append([np.around(prices[-1]/(capital/100) -100,2), np.around(np.mean(step_percent),2), np.around(len(step_percent[step_percent>0])/len(step_percent)*100,2)])

	# Setting box
	#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	plt.tick_params(axis='x',which='minor',top=True,direction='in')
	plt.tick_params(axis='y',which='minor',right=True,direction='in')
	plt.tick_params(axis='x',which='major',top=True,direction='in')
	plt.tick_params(axis='y',which='major',right=True,direction='in')
	plt.minorticks_on()
	plt.tick_params(axis='both', left=True, top=True, right=True, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
	#ax1.set_yticks([ax1.xaxis.set_major_formatter], minor=True)
	#ax1.xaxis.grid(True, which='minor')
	plt.grid(":",linewidth=1,color='gray',alpha=0.3)

	plt.xlabel("Months",fontsize=12)
	plt.ylabel("Total profit",fontsize=12)

	plt.ylim(5000,50000)
#	plt.ylim(min(prices) - .2*min(prices),max(prices) + .2*max(prices))
	plt.xticks(fontsize=10)
	plt.yticks(fontsize=10)

	plt.title(str(title))

	# Saving the figure.
#	plt.savefig('figures/'+str(strategy)+str(j)+'.jpg',dpi=300)
	plt.clf()


	return np.around(np.sum(step_percent),2),np.around(prices[-1]/100,2),np.around(np.std(step_percent),2),np.around(np.mean(step_percent),2)


##################################################################################
##################################################################################

def Figure(i,Stock,Adj_close,High,Volume):
	# Define the figure size.
	ax1 = plt.figure(1)
	fig, ax1 = plt.subplots(nrows=2, ncols=1,figsize=(12,7))
	plt.subplots_adjust(left  = 0.07,right = 0.94,bottom = 0.09,top = 0.95,wspace = 0.2,hspace = 0.2)

	ax1 = plt.subplot2grid((1,1), (0, 0))

	# define the Date (x.index) as a datetime pandas array
	Adj_close.index = pd.DatetimeIndex(Adj_close.index)
	aux_date2num = Adj_close.index.map(mdates.date2num)


	# compute exponencial moving avarages.
	x_MA72 = Adj_close.ewm(span=72).mean()
	x_MA17 = Adj_close.ewm(span=17).mean()
	x_MA4 = Adj_close.ewm(span=4).mean()

#	# plot the exponencial moving avarages.
#	plt.plot(aux_date2num,x_MA4,'.-',color='steelblue',linewidth=1,alpha=0.7)
#	plt.plot(aux_date2num,x_MA17,'--',color='steelblue',linewidth=1,alpha=0.7)
#	plt.plot(aux_date2num,x_MA72,'-',color='steelblue',linewidth=1,alpha=0.7)
#	# Legend
#	plt.legend(("Moving Average (4)","Moving Average (17)","Moving Average (72)",""),fontsize=8, loc='best', scatterpoints=1, markerscale=1,frameon=False)

	plt.plot(Adj_close,'.-',lw=2)

#	plt.plot(High,'.-',lw=1,color='k',alpha=0.8)

	#for k in range(len(Stock)):
#	plt.legend((str(Stock[0]),str(Stock[1]),str(Stock[2])),fontsize=8, loc='best', scatterpoints=1, markerscale=1,frameon=False)
#	plt.legend((str(Stock[0]),str(Stock[1]),str(Stock[2]),str(Stock[3]),str(Stock[4]),str(Stock[5]),str(Stock[6]),str(Stock[7]),str(Stock[8]),str(Stock[9])),fontsize=8, loc='best', scatterpoints=1, markerscale=1,frameon=False)
#	plt.legend((str(Stock[0]),str(Stock[1]),str(Stock[2]),str(Stock[3]),str(Stock[4]),str(Stock[5]),str(Stock[6]),str(Stock[7]),str(Stock[8])),fontsize=8, loc='best', scatterpoints=1, markerscale=1,frameon=False)


	# Setting box
	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	plt.tick_params(axis='x',which='minor',top=True,direction='in')
	plt.tick_params(axis='y',which='minor',right=True,direction='in')
	plt.tick_params(axis='x',which='major',top=True,direction='in')
	plt.tick_params(axis='y',which='major',right=True,direction='in')
	plt.minorticks_on()
	plt.tick_params(axis='both', left=True, top=True, right=True, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
	#ax1.set_yticks([ax1.xaxis.set_major_formatter], minor=True)
	#ax1.xaxis.grid(True, which='minor')
	plt.grid(":",linewidth=1,color='gray',alpha=0.3)

	plt.xlabel("Days",fontsize=10)
	plt.ylabel("Price in BLR of "+str(Stock),fontsize=10)
	plt.title(str(Stock),fontsize=10)

	plt.xlim(min(aux_date2num)-50,max(aux_date2num)+50);
#	plt.ylim(min(Adj_close.values)-min(Adj_close.values)*0.05,max(Adj_close.values)+max(Adj_close.values)*0.05)
	plt.xticks(fontsize=7)
	plt.yticks(fontsize=7)

	# Saving the figure.
	plt.savefig('figures/'+str(i)+'_sobesobe.jpg',dpi=200)
	plt.clf()


	return

##################################################################################
##################################################################################

def get_data(list_of_stoks,volume,period,interval,convert_month):

	'''
	This function loads a list of stocks codes and search in the yahoo finances database for the adjusted prices in certain dates.


	Inputed Parameters
	----------
	list_of_stoks: numpy.ndarray
		Loads a file with the stocks codes to be seached in the yahoo database.

	volume: numpy.float64
		The minimum mean volume of the stock in the period to be considered.

	period: str
		The period to obtain the data.

	interval: str
		The interval of the data.


	convert_month: Boolean
		If True, zip the data in days to month. If False, do nothing.


	Returns
	-------
	adj_prices: pandas.core.frame.DataFrame
		The dataframe with the adjusted price for all stocks loaded in list_of_stoks with volume higher than the adopted volume.


	Notes
	-------


	'''

	# Getting stocks data from yahoo finance
	aux_data = yf.download(list(list_of_stoks),period=str(period), interval =str(interval),progress=False)
#	aux_data = yf.download(list(list_of_stoks), start = start_date, end = end_date,progress=False)

	aux_indexes = aux_data.index

	# Creating a DataFrame for each OHLC data for each stock analyzed.
	adj_prices = aux_data['Adj Close']
	Open 	   = aux_data['Open']
	Low  	   = aux_data['Low']
	High 	   = aux_data['High']
	Close  	   = aux_data['Close']
	Volume     = aux_data['Volume']

	# Saving the data
	adj_prices.to_csv("data_output/adj_prices_IBRA.csv",sep='\t',float_format='%.2f')
	Volume.to_csv("data_output/volumes_IBRA.csv",sep='\t',float_format='%.2f')
	High.to_csv("data_output/high_IBRA.csv",sep='\t',float_format='%.2f')
	Low.to_csv("data_output/low_IBRA.csv",sep='\t',float_format='%.2f')
	Open.to_csv("data_output/open_IBRA.csv",sep='\t',float_format='%.2f')
	Close.to_csv("data_output/close_IBRA.csv",sep='\t',float_format='%.2f')


	# Finishing the routine and returning the data
	return adj_prices,Open,Low,High,Close,Volume

##################################################################################
##################################################################################


def convert_to_month(shift_days,aux_data):

	'''
	This function converts the stock data from day to month


	Inputed Parameters
	----------
	shift_days: numpy.integer
		Define the shift in days from today to the past.

	aux_data: pandas.core.frame.DataFrame
		The pandas Dataframe of a stock using one of the OHLC data. The data need to be in a daily interval.


	Returns
	-------
	mon_data: pandas.core.frame.DataFrame
		The montly stock dataframe.


	Notes
	-------


	'''


	shift_days = shift_days
	mon_data = pd.DataFrame(aux_data.resample('BM').apply(lambda x: x[-1-shift_days]))
	end_of_months = mon_data.index.tolist()
	end_of_months[-1] = aux_data.index[-1]
	mon_data.index = end_of_months
	mon_data.index = mon_data.index - timedelta(days=shift_days)

	return (mon_data)

##################################################################################
##################################################################################





if __name__=='__main__':
	run()
