# encoding: utf-8
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import glob as glob
import os
from datetime import datetime, timedelta
from scipy import stats
#from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY
import urllib2,cookielib
import pandas_datareader.data as web
#import statsmodels.api as sm
import seaborn

def run():

	# loading the list of stock to be analyzed.
	list_of_stoks = np.loadtxt('main_stocks_list_ibov.txt',dtype=str).T[:]

	# Define the range of dates to be searched.
	start_date = datetime.now() - timedelta(days=200); 	
	end_date   = datetime.now();

	# 
	#while end_date.hour < 17:
	end_date   = datetime.now();
	print "The current date and time is ",end_date
	#print end_date.hour
	#print e
	adj_prices,aux_indexes = get_data(list_of_stoks,100000,start_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d"))

	adj_prices = pd.read_csv('adj_prices.csv', sep="\t")
	data_corr = stocks_correlation(adj_prices,0.95,1,False,True,False)[1]
	scatter_close(data_corr,adj_prices,2.0,0.1,True,end_date)


	return data_corr,adj_prices


##################################################################################################################
##														##
##														##
##					scatter_close function							##
##														##
##														##
##################################################################################################################


def scatter_close(data_corr,adj_prices,percentage_val,adv,plot_figure,end_date):

	'''
	This function "scatter_close" analysis the various possibiities of long/shorting stocks in the B3.
	It uses as input the correlation matrix of the stocks and their dayly adjusted prices to search for possibiities of trading.
	A figure comparing the stocks prices is also plotted.


	Inputed Parameters
	----------
	data_corr: pandas.core.frame.DataFrame
		The list of the correlated indexes using the adopted seletion from min e max similarity.
		As a note, this is the output from the stocks_correlation funtion.

	adj_prices: pandas.core.frame.DataFrame
		The dataframe with the adjusted price for all stocks loaded with volume higher than the adopted volume.
		As a note, this is the output from the get_data funtion.

	percentage_val: np.float64
		The minimum or maximum difference in the normalized value of both stocks that a pair tradding might start.
	
	adv: np.float64
		The minimum or maximum values in the normalized stocks that a pair tradding might ends.

	plot_figure: bool
		Boolean for whether plot or not the correlation matrix. 
		If plot_figure = True, do the plot. If plot_figure = False, pass.

	end_date: np.float64
		The date in which the position was closed.

	Returns
	-------



	Notes
	-------

	'''

	# define useful empty lists
	buying = []
	selling = []
	buying_hold = []
	selling_hold = []
	# define a possible useful indexes
	k=0
	# looping condition to getting into the data_corr values
	for i in xrange(len(data_corr.index)):
		# organizing the data
		k=k+1
		# statement to pass duplicated stocks (it must to be improved, still computing duplicated stocks.)
		if data_corr['stocks1'][i]!=data_corr['stocks2'][i]:

			# simplifying the calling of the respectives stocks to be analyzed
			x = adj_prices[data_corr['stocks1'][i]]; x_stock = data_corr['stocks1'][i]
			y = adj_prices[data_corr['stocks2'][i]]; y_stock = data_corr['stocks2'][i]

			# defining the percentage of changing compared to the last day
			x_pct_day = 100*(x.pct_change(1))
			y_pct_day = 100*(y.pct_change(1))

			# Calling the linear regression from scype stats.
			# the beta_v and alpha_v indexes also indicates how similar the stocks are.
			beta_v, alpha_v, r_value, p_value, std_err = stats.linregress(x_pct_day.values[1:],y_pct_day.values[1:])
			# statement to remove stocks that are not well correlated. Refining the sample.
			if beta_v > 0.40:#1000:#0.40:

				# define an spread funcion that is the difference of the adjusted prices from the stock1 - stock2.
				spread = x.values - y.values
				# calling z_score function in the spread data. It normalizes the spread difference to 0.
				z_score = zscore(spread)

				#
				print "Comparing the stoks "+str(data_corr['stocks1'][i])+" and "+str(data_corr['stocks2'][i])

				# define the Date (x.index) as a datetime pandas array
				x.index = pd.DatetimeIndex(x.index)

				# defining auxiliary variables and lists to save prices for backtest
				b1 = 0
				b1_aux = []
				b2 = 0
				b2_aux = []
				b3 = 0
				b3_aux = []
				b4 = 0
				b4_aux = []

				# looping to analyze the adjusted prices for each day on both stock1 and stock2.
				# the trigger to open or close a pair tradding is obtained here. 
				for m in xrange(len(x)):
					# An inputed np.float64 that defines the normalized value from spread (percentage_val) where a pair tradding might start.  
					percentage_val = percentage_val
					
					#print b1,b2,b3
					b3=0
					b4=0
					# statement to advise that there is the oportunity to start a pair tradding.
					if b1 == 0 and b2 == 0 and z_score[m] > percentage_val:
						#print "In the current date "+str(x.index[m].strftime("%Y-%m-%d"))+" there is the option to go short_sell on "+str(x_stock)+" ("+str(x.values[m])+") and long_buy on "+str(y_stock)+ " ("+str(y.values[m])+") with Z_score = "+str(np.around(z_score[m],2))
						#print
		
						# Start a pair tradding if we are no longer in a position (b1=0). Save adjusted prices and zsore.
						b1 = x.values[m],y.values[m],z_score[m],m
						#print str(b1)+" this is b1"
						b1_aux.append(b1)
					if b1 != 0 and b2 == 0 and -adv < z_score[m] < adv:
						b1 = 0
						b2 = 0
						# Finishing a pair tradding if we are positioned.
						b3 = x.values[m],y.values[m],z_score[m],m
						#print 
						#print str(b3)+" this is b3"
						b3_aux.append(b3)

					# statement to advise that there is the oportunity to start a pair tradding.
					if b1 == 0 and b2 == 0 and z_score[m] < -percentage_val:
						#print "In the current date "+str(x.index[m].strftime("%Y-%m-%d"))+" there is the option to go long_buy on "+str(x_stock)+" ("+str(x.values[m])+") and short_sell on "+str(y_losing my mindstock)+ " ("+str(y.values[m])+") with Z_score = "+str(np.around(z_score[m],2))
						#print 
						# Start a pair tradding if we are no longer in a position (b1=0). Save adjusted prices and zsore.
						b2 = x.values[m],y.values[m],z_score[m],m
						#print str(b2)+" this is b2"
						b2_aux.append(b2)
					# statement to advise that there is the oportunity to finish the pair tradding.
	
					if b1 == 0 and b2 != 0 and -adv < z_score[m] < adv:
						#print "In the current date "+str(x.index[m].strftime("%Y-%m-%d"))+" there is the oportunity to finish the position with Z_score = "+str(np.around(z_score[m],2))+". The current price for "+str(x_stock)+" is "+str(x.values[m])+"  and for "+str(y_stock)+" is "+str(y.values[m])

						# If we are finishing a position, we set b1 and b2 back to null and start looking for new entries in the statements above.
						b1 = 0
						b2 = 0
						# Finishing a pair tradding if we are positioned.
						b4 = x.values[m],y.values[m],z_score[m],m
						#print 
						#print str(b3)+" this is b3"
						b4_aux.append(b4)


				#print b1_aux,b3_aux
				#print b2_aux,b4_aux


				# Calling long_short_bt funtion to backtest our data. 
				if b1_aux != []: # and b3_aux != []:
					for l in xrange(len(b1_aux)):
						#print 'cima'
						if l+1 > len(b3_aux):
							print "Currently holding a position here shorting "+str(x_stock)+" and longing "+str(y_stock)

						#elif b3_aux[l][3]-b1_aux[l][3] < 0:
						#	print "Currently holding a position here shorting "+str(x_stock)+" and longing "+str(y_stock)

						else:
							#print long_short_bt(b1_aux[0][1],b1_aux[0][0],b3_aux[0][1],b3_aux[0][0],100,100)
							q1, q2 = quant_stocks(b1_aux[l][1],b1_aux[l][0],False)
							buying.append(long_short_bt(b1_aux[l][1],b1_aux[l][0],b3_aux[l][1],b3_aux[l][0],b3_aux[l][3]-b1_aux[l][3],q1*100,q2*100,True)[4])

				if b2_aux != []: # and b4_aux != []:
					for n in xrange(len(b2_aux)):
						#print 'baixo'
						if n+1 > len(b4_aux):
							print "Currently holding a position here shorting "+str(y_stock)+" and longing "+str(x_stock)

						#elif b3_aux[n][3]-b2_aux[n][3] < 0:
						#	print "Currently holding a position here shorting "+str(y_stock)+" and longing "+str(x_stock)

						else:
							#print long_short_bt(b1_aux[0][1],b1_aux[0][0],b3_aux[0][1],b3_aux[0][0],100,100)
							q1, q2 = quant_stocks(b2_aux[n][0],b2_aux[n][1],False)
							selling.append(long_short_bt(b2_aux[n][0],b2_aux[n][1],b4_aux[n][0],b4_aux[n][1],b4_aux[n][3]-b2_aux[n][3],q1*100,q2*100,True)[4])

				print 


#####################################################################################################################################################


				# building the figure
				# plot_figure is one of the input variables. If it is true, do the figure.
				plot_figure = plot_figure
				if plot_figure:	
					
					# Define the figure size.
					plt.figure(1)
					fig, ax = plt.subplots(nrows=4, ncols=2,figsize=(10,10))
					plt.subplots_adjust(left  = 0.07,right = 0.94,bottom = 0.06,top = 0.95,wspace = 0.2,hspace = 0.2)

					plt.subplot2grid((4,2), (0, 0))
					
					# plotting the adjusted prices for stocks1 and stocks2 in the x and y axis, respectively.
					plt.scatter(x.values[0:-30],y.values[0:-30],color='k',alpha=0.50,marker='x',s=20,linewidth=2)
					plt.scatter(x.values[-30:-15],y.values[-30:-15],color='crimson',alpha=0.50,marker='o',s=40,linewidth=1)
					plt.scatter(x.values[-15:-5],y.values[-15:-5],color='darkviolet',alpha=0.50,marker='o',s=40,linewidth=1)
					plt.scatter(x.values[-5],y.values[-5],s=40,color='blue',edgecolor='k',alpha=0.80,marker='o',linewidth=1)
					plt.scatter(x.values[-4],y.values[-4],s=60,color='cyan',edgecolor='k',alpha=0.80,marker='o',linewidth=1)
					plt.scatter(x.values[-3],y.values[-3],s=80,color='green',edgecolor='k',alpha=0.80,marker='o',linewidth=1)
					plt.scatter(x.values[-2],y.values[-2],s=100,color='darkorange',edgecolor='k',alpha=0.80,marker='o',linewidth=1)
					plt.scatter(x.values[-1],y.values[-1],s=120,color='red',edgecolor='k',alpha=0.80,marker='o',linewidth=1)

					# plot legend
					plt.legend(("200 days before","30 days before","15 days before","5 days before","4 days before","3 days before","2 days before","1 day before"),fontsize=5, loc='best', scatterpoints=1, markerscale=1,frameon=False)

					# Connect the last 20 days
					plt.plot(x.values[-20:],y.values[-20:],'-k',linewidth=0.8)

					# Setting box
					plt.tick_params(axis='x',which='minor',top='on',direction='in')
					plt.tick_params(axis='y',which='minor',right='on',direction='in')
					plt.tick_params(axis='x',which='major',top='on',direction='in')
					plt.tick_params(axis='y',which='major',right='on',direction='in')
					plt.minorticks_on()
					plt.tick_params(axis='both', left='on', top='on', right='on', bottom='on', labelleft='on', labeltop='off', labelright='off', labelbottom='on')
					plt.xlabel("Price in BLR "+str(x_stock),fontsize=8)
					plt.ylabel("Price in BLR "+str(y_stock),fontsize=8)
					plt.title("The current date is: "+str(end_date),fontsize=8)
					plt.xticks(fontsize=7)
					plt.yticks(fontsize=7)
					#plt.xlim(min(x)-2,max(x)+2)
					#plt.ylim(min(y)-2,max(y)+2)
				#####################################################################################################################################################

					plt.subplot2grid((4,2), (0, 1))

					# define an auxiliary vector X in order to plot the beta_v and alpha_v coefficients.
					X = np.linspace(-10.,10,60)
					plt.plot(X,X*beta_v+alpha_v,'-',color='darkolivegreen',linewidth=1,alpha=0.9,zorder=-10)

					# plotting the percent variation in the adjusted prices based on the last day for stocks1 and stocks2 in the x and y axis, respectively.
					plt.scatter(x_pct_day[0:-30],y_pct_day[0:-30],color='k',alpha=0.50,marker='x',s=20,linewidth=2)
					plt.scatter(x_pct_day[-30:-15],y_pct_day[-30:-15],color='crimson',alpha=0.50,marker='o',s=40,linewidth=1)
					plt.scatter(x_pct_day[-15:-5],y_pct_day[-15:-5],color='darkviolet',alpha=0.50,marker='o',s=40,linewidth=1)
					plt.scatter(x_pct_day[-5],y_pct_day[-5],s=40,color='blue',edgecolor='k',alpha=0.80,marker='o',linewidth=1)
					plt.scatter(x_pct_day[-4],y_pct_day[-4],s=60,color='cyan',edgecolor='k',alpha=0.80,marker='o',linewidth=1)
					plt.scatter(x_pct_day[-3],y_pct_day[-3],s=80,color='green',edgecolor='k',alpha=0.80,marker='o',linewidth=1)
					plt.scatter(x_pct_day[-2],y_pct_day[-2],s=100,color='darkorange',edgecolor='k',alpha=0.80,marker='o',linewidth=1)
					plt.scatter(x_pct_day[-1],y_pct_day[-1],s=120,color='red',edgecolor='k',alpha=0.80,marker='o',linewidth=1)
					# Connect the last 5 days
					plt.plot(x_pct_day[-6:],y_pct_day[-6:],'-k',linewidth=0.8)
					# draw lines in the x and y label centered in 0.
					plt.plot([-20,20],[-0,0],'--k',alpha=0.8)
					plt.plot([-0,0],[-20,20],'--k',alpha=0.8)

					# Printing in the figure the mean values of the difference between stock1 and stock2
					plt.text(-9.8,8,"<X/Y> 200d: "+str(np.around(np.mean(x_pct_day[:]/y_pct_day[:]),3))+" +- "+str(np.around(np.std(x_pct_day[:]/y_pct_day[:]),3)),fontsize=6)
					plt.text(-9.8,7,"<X/Y> 20d: "+str(np.around(np.mean(x_pct_day[-21:-1]/y_pct_day[-21:-1]),3))+" +- "+str(np.around(np.std(x_pct_day[-21:-1]/y_pct_day[-21:-1]),3)),fontsize=6)
					plt.text(-9.8,6,"<X/Y> 5d: "+str(np.around(np.mean(x_pct_day[-6:-1]/y_pct_day[-6:-1]),3))+" +- "+str(np.around(np.std(x_pct_day[-6:-1]/y_pct_day[-6:-1]),3)),fontsize=6)
					plt.text(-9.8,5,"<X/Y> 1d: "+str(np.around(np.mean(x_pct_day[-1]/y_pct_day[-1]),3)),fontsize=6)

					# Printing in the figure the beta_v and alpha_v values.
					plt.text(-9.5,-9.,"beta value = "+str(np.around(beta_v,2))+"\nalpha value = "+str(np.around(alpha_v,2)),fontsize=6)

					# Setting box
					plt.tick_params(axis='x',which='minor',top='on',direction='in')
					plt.tick_params(axis='y',which='minor',right='on',direction='in')
					plt.tick_params(axis='x',which='major',top='on',direction='in')
					plt.tick_params(axis='y',which='major',right='on',direction='in')
					plt.minorticks_on()
					plt.tick_params(axis='both', left='on', top='on', right='on', bottom='on', labelleft='on', labeltop='off', labelright='off', labelbottom='on')
					if data_corr['corr_coef'][i] > 0.95:
						plt.title("The correlation index for the stocks is: "+str(np.around(data_corr['corr_coef'][i],3)),fontsize=12,fontweight="bold",color='green')
					elif 0.92 < data_corr['corr_coef'][i] <= 0.95:
						plt.title("The correlation index for the stocks is: "+str(np.around(data_corr['corr_coef'][i],3)),fontsize=12,fontweight="bold",color='blue')
					elif data_corr['corr_coef'][i] <= 0.92:
						plt.title("The correlation index for the stocks is: "+str(np.around(data_corr['corr_coef'][i],3)),fontsize=12,fontweight="bold",color='red')
					plt.xlabel("Percentage of change to last day "+str(x_stock),fontsize=8)
					plt.ylabel("Percentage of change to last day "+str(y_stock),fontsize=8)
					plt.xlim(-10,10); plt.ylim(-10,10)
					plt.xticks(fontsize=7)
					plt.yticks(fontsize=7)
				#####################################################################################################################################################

					ax = plt.subplot2grid((4,2), (1, 0),colspan=2)

					# define the Date (x.index) as a datetime pandas array
					x.index = pd.DatetimeIndex(x.index)
					aux_date2num = x.index.map(mdates.date2num)

					# compute exponencial moving avarages.
					x_MA72 = x.ewm(span=72).mean()
					x_MA17 = x.ewm(span=17).mean()
					x_MA4 = x.ewm(span=4).mean()

					# plot the exponencial moving avarages.
					plt.plot(aux_date2num,x_MA4,'.-',color='steelblue',linewidth=1,alpha=0.8)
					plt.plot(aux_date2num,x_MA17,'--',color='steelblue',linewidth=1,alpha=0.8)
					plt.plot(aux_date2num,x_MA72,'-',color='steelblue',linewidth=1,alpha=0.8)
					# Legend
					plt.legend(("Moving Average (4)","Moving Average (17)","Moving Average (72)",""),fontsize=8, loc='best', scatterpoints=1, markerscale=1,frameon=False)
					# plot the adjusted prices data for the stock1
					plt.plot(aux_date2num,x.values,'-',color='k',linewidth=1,alpha=0.8)
					plt.scatter(aux_date2num,x.values,color='k',linewidth=1,alpha=0.8,facecolor='None',s=20)

					# Setting box
					ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
					plt.tick_params(axis='x',which='minor',top='on',direction='in')
					plt.tick_params(axis='y',which='minor',right='on',direction='in')
					plt.tick_params(axis='x',which='major',top='on',direction='in')
					plt.tick_params(axis='y',which='major',right='on',direction='in')
					plt.minorticks_on()
					plt.tick_params(axis='both', left='on', top='on', right='on', bottom='on', labelleft='on', labeltop='off', labelright='off', labelbottom='on')
					ax.set_yticks([ax.xaxis.set_major_formatter], minor=True)
					ax.xaxis.grid(True, which='minor')
					plt.grid(":",linewidth=1,color='gray',alpha=0.3)

					plt.xlabel("Days",fontsize=8)
					plt.ylabel("Price in BLR of "+str(x_stock),fontsize=8)
					plt.xlim(min(aux_date2num)-4,max(aux_date2num)+4); 
					plt.ylim(min(x.values)-min(x.values)*0.05,max(x.values)+max(x.values)*0.05)
					plt.xticks(fontsize=7)
					plt.yticks(fontsize=7)
				######################################################################################################################################################


					ax = plt.subplot2grid((4,2), (2, 0),colspan=2)

					# define the Date (y.index) as a datetime pandas array.
					y.index = pd.DatetimeIndex(y.index)
					aux_date2num = y.index.map(mdates.date2num)

					# compute exponencial moving avarages.
					y_MA72 = y.ewm(span=72).mean()
					y_MA17 = y.ewm(span=17).mean()
					y_MA4 = y.ewm(span=4).mean()

					# plot exponencial moving avarages.
					plt.plot(aux_date2num,y_MA4,'.-',color='steelblue',linewidth=1,alpha=0.8)
					plt.plot(aux_date2num,y_MA17,'--',color='steelblue',linewidth=1,alpha=0.8)
					plt.plot(aux_date2num,y_MA72,'-',color='steelblue',linewidth=1,alpha=0.8)

					# plot the adjusted prices data for the stock2.
					plt.plot(aux_date2num,y.values,'-',color='k',linewidth=1,alpha=0.8)
					plt.scatter(aux_date2num,y.values,color='k',linewidth=1,alpha=0.8,facecolor='None',s=20)

					# Setting box
					ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
					plt.tick_params(axis='x',which='minor',top='on',direction='in')
					plt.tick_params(axis='y',which='minor',right='on',direction='in')
					plt.tick_params(axis='x',which='major',top='on',direction='in')
					plt.tick_params(axis='y',which='major',right='on',direction='in')
					plt.minorticks_on()
					plt.tick_params(axis='both', left='on', top='on', right='on', bottom='on', labelleft='on', labeltop='off', labelright='off', labelbottom='on')
					ax.set_yticks([ax.xaxis.set_major_formatter], minor=True)
					ax.xaxis.grid(True, which='minor')
					plt.grid(":",linewidth=1,color='gray',alpha=0.3)

					plt.xlabel("Days",fontsize=8)
					plt.ylabel("Price in BLR of "+str(y_stock),fontsize=8)
					plt.xlim(min(aux_date2num)-4,max(aux_date2num)+4); 
					plt.ylim(min(y.values)-min(y.values)*0.05,max(y.values)+max(y.values)*0.05)
					plt.xticks(fontsize=7)
					plt.yticks(fontsize=7)


########################################################################################################################################################################

					ax = plt.subplot2grid((4,2), (3, 0),colspan=2)

					#1 day moving average of the price spread.
					spread_mavg1 = pd.DataFrame(spread).rolling(1).mean()
					
					# 30 day moving average of the price spread.
					spread_mavg30 = pd.DataFrame(spread).rolling(30).mean()
					
					# Take a rolling 30 day standard deviation.
					std_30 = pd.DataFrame(spread).rolling(30).std()
					# Compute the z score for each day.
					zscore_30_1 = (spread_mavg1 - spread_mavg30)/std_30		

					# plotting the zscore index from the z_score funtion as well as its mean values.
					plt.plot(aux_date2num,z_score,'-',color='k',linewidth=1,alpha=0.8)
					plt.scatter(aux_date2num,z_score,color='k',linewidth=1,alpha=0.8,facecolor='None',s=20)
					plt.plot(aux_date2num,zscore_30_1,'-',color='b',linewidth=1,alpha=0.8)


					# drawing the lines in the figure that are indicative of open or finish the trading positions.				
					plt.plot([-5+min(aux_date2num),max(aux_date2num)+5],[0,0],'-r',linewidth=8,alpha=0.4)
					plt.plot([-5+min(aux_date2num),max(aux_date2num)+5],[-2.0,-2.0],'-g',linewidth=1,alpha=0.8)
					plt.plot([-5+min(aux_date2num),max(aux_date2num)+5],[-1.5,-1.5],'-.g',linewidth=1,alpha=0.8)
					plt.plot([-5+min(aux_date2num),max(aux_date2num)+5],[-1.,-1.],'--g',linewidth=1,alpha=0.8)

					plt.plot([-5+min(aux_date2num),max(aux_date2num)+5],[2,2],'-g',linewidth=1,alpha=0.8)
					plt.plot([-5+min(aux_date2num),max(aux_date2num)+5],[1.5,1.5],'-.g',linewidth=1,alpha=0.8)
					plt.plot([-5+min(aux_date2num),max(aux_date2num)+5],[1.0,1.0],'--g',linewidth=1,alpha=0.8)

					# drawing regions in the figure that are indicative of open or finish the trading positions.	
					xs = np.arange(-5+min(aux_date2num),max(aux_date2num)+5,0.1)
					plt.fill_between(xs,1,percentage_val,color='green', alpha=0.2)
					plt.fill_between(xs,percentage_val,10,color='green', alpha=0.5)
					plt.fill_between(xs,-1,-percentage_val,color='green', alpha=0.2)
					plt.fill_between(xs,-percentage_val,-10,color='green', alpha=0.5)


					# Setting box
					ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
					plt.tick_params(axis='x',which='minor',top='on',direction='in')
					plt.tick_params(axis='y',which='minor',right='on',direction='in')
					plt.tick_params(axis='x',which='major',top='on',direction='in')
					plt.tick_params(axis='y',which='major',right='on',direction='in')
					plt.minorticks_on()
					plt.tick_params(axis='both', left='on', top='on', right='on', bottom='on', labelleft='on', labeltop='off', labelright='off', labelbottom='on')
					ax.set_yticks([ax.xaxis.set_major_formatter], minor=True)
					ax.xaxis.grid(True, which='minor')
					plt.grid(":",linewidth=1,color='gray',alpha=0.3)

					plt.xlabel("Days",fontsize=8)
					plt.ylabel("Normalized diference (or Zscore) of\n"+str(x_stock)+" and "+str(y_stock),fontsize=8)
					plt.xlim(min(aux_date2num)-4,max(aux_date2num)+4); 
					plt.ylim(min(z_score)-1,max(z_score)+1)					
					plt.xticks(fontsize=7)
					plt.yticks(np.linspace(-2,2,9),fontsize=7)
					
					# Saving the figure.
					plt.savefig('/home/user/finances/main_'+str(x_stock)+'_'+str(y_stock)+'.jpg',dpi=200)
					plt.clf()
	#print buying
	#print selling
	print len(np.hstack([buying,selling])), np.sum((np.hstack([buying,selling]))),np.mean((np.hstack([buying,selling])))

	return	


##################################################################################################################
##														##
##														##
##					get_data function							##
##														##
##														##
##################################################################################################################


def get_data(list_of_stoks,volume,start_date,end_date):

	'''
	This function loads a list of stocks codes and search in the yahoo finances database for the adjusted prices in certain dates.


	Inputed Parameters
	----------
	list_of_stoks: numpy.ndarray
		Loads a file with the stocks codes to be seached in the yahoo database.

	volume: numpy.float64
		The minimum mean volume of the stock in the period to be considered

	start_date: str
		The starting date to be used in the search of stocks data

	end_date: numpy.float64
		The ending date to be used in the search of stocks data


	Returns
	-------
	adj_prices: pandas.core.frame.DataFrame
		The dataframe with the adjusted price for all stocks loaded in list_of_stoks with volume higher than the adopted volume.


	Notes
	-------

	'''

	# using PETR4 as date and size as reference.
	aux_indexes = web.DataReader("PETR4.SA", "yahoo",start=start_date, end=end_date)
	aux_indexes = aux_indexes[aux_indexes['Volume'] > 0]

	# define auxiliary lists
	aux = []
	aux_stocks = []
	# looping in each stock
	for i in list_of_stoks:
		# getting the data using pandas web.DataReader and the yahoo plataform
		aux_data = web.DataReader(i+".SA", "yahoo",start=start_date, end=end_date)
		#print aux_data.tail()
		# selecting only the adjusted close price for each stock. The [aux_data['Volume'] > 0] is aimed to remove holydays and weekends
		aux_close = aux_data['Adj Close'][aux_data['Volume'] > 0]
		
		# statement to get the most negocieted stocks. The volume here is also one of the function inputs.
		# the statement for len(aux_close.values) == len(aux_indexes.index) is to force the same about data for all stocks. Note that it might remove recent IPOs.
		if len(aux_close.values) == len(aux_indexes.index) and np.mean(aux_data['Volume']) > volume:
			# append stock adjusted price data		
			aux.append(aux_close.values)
			# append stock name
			aux_stocks.append(i)
			print "getting data for "+str(i)
			

	adj_prices = pd.DataFrame(np.array(aux).T,columns=aux_stocks,index=aux_indexes.index)
	adj_prices.to_csv("adj_prices.csv",sep='\t',float_format='%.2f')
	#print adj_prices

	return adj_prices,aux_indexes

##################################################################################################################
##														##
##														##
##					stocks_correlation function						##
##														##
##														##
##################################################################################################################

def stocks_correlation(df,min_corr,max_corr,plot_figure,dropDuplicates,annot):
	'''
	This function runs a correlation matrix using the pandas library for several stocks selected.

	Inputed Parameters
	----------
	df: pandas.core.frame.DataFrame
		Dataframe with the adjusted price for n stocks

	min_corr: np.float64
		the lowest correlation degree to be analysed (min = -1)
		
	max_corr: np.float64
		the higher correlation degree to be analysed (max = +1)

	plot_figure: bool
		Boolean for whether plot or not the correlation matrix. 
		If plot_figure = True, do the plot. If plot_figure = False, pass 

	dropDuplicates: bool
		Boolean for whether plot or not the duplicated correlated stocks. 
		If dropDuplicates = True, mask the duplicated. If dropDuplicates = False, pass 

	annot: bool
		Boolean for whether write or not the value for the correlated matrix. 
		If annot = True, print the numbers on each box. If annot = False, pass.


	Returns
	-------
	data_sim_corr_matrix: pandas.core.frame.DataFrame
		The matrix of the correlated indexes using the adopted seletion from min e max similarity

	data_sim_corr_list: pandas.core.frame.DataFrame
		The list of the correlated indexes using the adopted seletion from min e max similarity



	Notes
	-------
	figure were based on (https://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor) discussion.

	'''
	# changing index
	df.index = df['Date']
	df = df.drop('Date',1)

	# Selecting the last x days to compute the correlation matrix. Default is 60 days, roughly 3 months.
	# It uses the data as dataframe as input.
	df = pd.DataFrame(df[len(df)-60:])
	# Define the list of stocks analyzed.
	stocks = df.keys()

	# compute the correlation.
	data = df.corr()

	# Selecting stocks with certain correlation degree, from the function input min_corr and max_corr.
	aux=[] 
	aux_sim = []  
	# sim_min (minimum of similarity)
	sim_min = min_corr
	# sim_max (maximum of similarity)
	sim_max = max_corr

	# loop statement to get into the correlation matrix to select the data desired.
	for i in xrange(len(data.values)):
		for j in xrange(len(data.values)):
			# conditional statement to select the higher values from the correlation matrix.
			if str(stocks[i])!=str(stocks[j]) and sim_min < data.values[i][j] < sim_max:
			    	#print data.values[i][j], stocks[i],stocks[j]
				# append the data				
			   	aux.append(data.values[i][j])
			   	aux_sim.append([stocks[i],stocks[j],data.values[i][j]])

			# conditional statement to select the lowest values from the correlation matrix.
			elif str(stocks[i])!=str(stocks[j]) and -sim_max < data.values[i][j] < -sim_min:    
				#print data.values[i][j], stocks[i],stocks[j]
				# append the data
				aux.append(data.values[i][j])
			   	aux_sim.append([stocks[i],stocks[j],data.values[i][j]])
			else: 
				# append the data
		   		aux.append(np.nan) 
#	   
	# organizing the data from the correlation matrix selected from the correlation degree
	data_sim_corr_matrix = np.reshape(aux,(len(data.values.T),len(data.values.T))) 
	data_sim_corr_matrix = pd.DataFrame(data_sim_corr_matrix,index=stocks,columns=stocks)
	data_sim_corr_list = pd.DataFrame(aux_sim,columns=['stocks1','stocks2','corr_coef'])
	# Removing duplicated stocks based on their correlation coefficient.
	data_sim_corr_list = data_sim_corr_list.drop_duplicates('corr_coef')
	data_sim_corr_list.index = np.linspace(0,len(data_sim_corr_list)-1,len(data_sim_corr_list))
	
	#saving selected correlation matrix data
	data_sim_corr_matrix.to_csv("correlation_matrix.csv",sep='\t',float_format='%.3f')
	data_sim_corr_list.to_csv('correlation_list.csv',sep='\t')


###################################################
	
	# Making Figure

	# For plot_figure = True, build the figure. 
	plot_figure = plot_figure
	if plot_figure:	
		import seaborn as sns
		# Figure format
		fig = plt.figure(figsize=(8,8))                                                               
		ax = fig.add_subplot(1,1,1)                                                                     
		plt.subplots_adjust(left  = 0.1,right = 0.99,bottom = 0.05,top = 0.95,wspace = 0.1,hspace = 0.1)

		# if statement for mask the duplicated values in the correlation matrix.
		dropDuplicates = dropDuplicates
		if dropDuplicates:    
			mask = np.zeros_like(data_sim_corr_matrix, dtype=np.bool)
			mask[np.triu_indices_from(mask)] = True

		# Color options
		#cmap = sns.diverging_palette(0,255,sep=77, as_cmap=True)
		#cmap = 'RdBu'
		cmap = 'PuOr'
		#ax = sns.heatmap(data_sim_corr_matrix, cmap=cmap, linewidths=.1)#annot=True

		# Drawing the figure from sns.heatmap	
		if dropDuplicates:
	       		ax = sns.heatmap(data_sim_corr_matrix, mask=mask, cmap=cmap, square=True, linewidth=.5, cbar_kws={"shrink": .5}, annot=annot, fmt='.2f')
		else:
			ax = sns.heatmap(data_sim_corr_matrix, cmap=cmap, square=True, linewidth=.1, cbar_kws={"shrink": .5}, annot=annot, fmt='.2f')


		# relabel columns
		labels = stocks
		# set appropriate font and dpi
		sns.set(font_scale=1)
	
		# set the x-axis labels on the top
		ax.xaxis.tick_top()
		# rotate the x-axis labels
		plt.xticks(rotation=90)
		plt.yticks(rotation=0)
		# set the x,y ticks size
		plt.xticks(fontsize=7)
		plt.yticks(fontsize=7)

	
		# Saving figure
		plt.show()
		#plt.savefig("Correlation_vol100000_stocks.pdf",dpi=1000)
		plt.clf()	

	return data_sim_corr_matrix,data_sim_corr_list

##################################################################################################################
##														##
##														##
##					zscore function								##
##														##
##														##
##################################################################################################################

def zscore(stocks_spread):

	'''
	The zscore function compute the normalized value of the difference between two stocks.

	Inputed Parameters
	----------
	stocks_spread: pandas.core.frame.DataFrame
		The difference in prices between two stocks.

	Returns
	-------
	zscore: pandas.core.frame.DataFrame
		The normalized value of the difference between two stocks.

	Notes
	-------

	'''
	return (stocks_spread - stocks_spread.mean()) / np.std(stocks_spread)

##################################################################################################################
##														##
##														##
##					quant_stocks function							##
##														##
##														##
##################################################################################################################


def quant_stocks(price_stock1, price_stock2, run_stat):

	'''
	The quant_stocks function compute the value to have both stocks prices with the same wheigh.

	Inputed Parameters
	----------
	price_stock1: np.float64
		Current price for the stock1.

	price_stock2: np.float64
		Current price for the stock2.

	run_stat: bool
		Boolean for whether obtain or not the multiplicative factor. 
		If run_stat = True, run. If plot_figure = False, pass.


	Returns
	-------
	q1: np.float64
		The factor to be multiplied in the stocks1 price to have the same wheigh on both stoks.

	q2: np.float64
		The factor to be multiplied in the stocks2 price to have the same wheigh on both stoks.

	Notes
	-------

	'''
	# define the ratio between the adjusted prices of the analysed stocks.
	ratio_price = price_stock1/price_stock2

	# define q1 and q2. If there is no multiplicative factor, the coefficients does not change.
	q1 = 1
	q2 = 1

	run_stat = run_stat
	if run_stat:	
		# statement to include the multiplicative factor on either stock1 or stock2.
		if price_stock1 > price_stock2:
			multiply_factor = ratio_price
			q2 = q2*multiply_factor
		elif price_stock2 > price_stock1:
			multiply_factor = 1/ratio_price
			q1 = q1*multiply_factor

		#print 
		#print price_stock1,price_stock2, ratio_price
		#print q1,q2
		#print 
	
	return q1,q2

##################################################################################################################
##														##
##														##
##					long_short_bt function							##
##														##
##														##
##################################################################################################################

def long_short_bt(long_in,short_in,long_fi,short_fi,days_positioned,n_st_long,n_st_short,comments):

	'''
	This function "long_short_bt" do the backtest of the long short positions in the previous data desired.
	

	Inputed Parameters
	----------
	long_in: np.float64
		The price of the opening position for the long part (buying stocks).

	short_in: np.float64
		The price of the opening position for the short part (selling stocks).

	long_fi: np.float64
		The price of the closing position for the long part (selling the stocks previously purchased).

	short_fi: np.float64
		The price of the closing position for the short part (buying the stocks previously sold).

	days_positioned: np.float64
		The total of days having the stocks in the portifolio.

	n_st_long: np.float64
		The number of stocks to be longed.

	n_st_short: np.float64
		The number of stocks to be shorted.

	comments: bool
		Boolean for whether comment or not the operation summary. 
		If comments = True, run. If comments = False, pass.


	Returns
	-------
	longs: np.float64
		Final profit or wast in the longs. Prices in BLR.

	short: np.float64
		Final profit or wast in the shorts. Prices in BLR.

	total_invested: np.float64
		The total money invested.

	total_returned: np.float64
		The total money returned.

	total_returned_perc: np.float64
		The percetage of profit or wast in the operation.

	days_positioned: np.float64
		The total of days having the stocks in the portifolio.


	Notes
	-------

	'''
	# the total of days positioned.
	days_positioned = days_positioned
	# Prices for the opening positions.
 	long_in = long_in*n_st_long; short_in = short_in*n_st_short
	# Prices for the closing positions.
 	long_fi = long_fi*n_st_long; short_fi = short_fi*n_st_short;\
	# delta difference
 	longs = long_fi-long_in;\
 	short = short_in-short_fi;\
	# percentage of gains
 	percent_longs = 100*((long_fi-long_in)/long_in)
 	percent_short = 100*((short_fi-short_in)/short_in)
	# gains or losses
 	total_invested = long_in+short_in
 	total_returned = longs+short
 	total_returned_perc = 100*(((total_invested+total_returned)/total_invested)-1)

	comments = comments
	if comments:

		print "A pair tradding was perfomed, where:"
		print "The total of money invested is: "+str(np.around(total_invested,2))
		print "Returns from the longs: "+str(np.around(longs,2))
		print "Returns from the shorts: "+str(np.around(short,2))
	 	print "The final profit in BLR: "+str(np.around(total_returned,2))
	 	print "The final profit in % "+str(np.around(total_returned_perc,2))
		print "The total of days positioned was: "+str(days_positioned)

 	return longs,short,total_invested,total_returned,total_returned_perc,days_positioned


##################################################################################################################
##														##
##														##
##														##
##														##
##														##
##################################################################################################################


if __name__=='__main__':
	run()




