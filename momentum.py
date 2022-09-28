# encoding: utf-8
'''sobesobe.py:  The goal of this program is to find the stocks that raised the most
in a certain period. Backtests are also included.
'''

__author__ = 'Diogo Souto'
__email__ = 'diogodusouto.at.gmail.com'

import numpy as np
import matplotlib.pyplot as plt
import heapq
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
yf.pdr_override()
plt.rcParams.update({'figure.max_open_warning': 0})

adj_prices_aux = pd.read_csv('data_output/adj_prices_sobesobe.csv', sep="\t")
adj_prices_aux = adj_prices_aux.set_index(pd.DatetimeIndex(adj_prices_aux['Date']))
adj_prices_aux = adj_prices_aux.drop(['Date'], axis=1)
High = pd.read_csv('data_output/high_sobesobe.csv', sep="\t")
High = High.set_index(pd.DatetimeIndex(High['Date']))
High = High.drop(['Date'], axis=1)
month_AdjClose = pd.read_csv('data_output/month_AdjClose_sobesobe.csv', sep="\t")
month_AdjClose = \
    month_AdjClose.set_index(pd.DatetimeIndex(month_AdjClose['Unnamed: 0']))
month_AdjClose = month_AdjClose.drop(['Unnamed: 0'], axis=1)
month_High = pd.read_csv('data_output/month_High_sobesobe.csv', sep="\t")
month_High = month_High.set_index(pd.DatetimeIndex(month_High['Unnamed: 0']))
month_High = month_High.drop(['Unnamed: 0'], axis=1)
month_Volume = pd.read_csv('data_output/month_Volume_sobesobe.csv', sep="\t")
month_Volume = \
    month_Volume.set_index(pd.DatetimeIndex(month_Volume['Unnamed: 0']))
month_Volume = month_Volume.drop(['Unnamed: 0'], axis=1)

# adj_prices_aux = adj_prices_aux.drop(['TSLA'], axis=1)
# High = High.drop(['TSLA'], axis=1)
# month_AdjClose = month_AdjClose.drop(['TSLA'], axis=1)
# month_High = month_High.drop(['TSLA'], axis=1)
# month_Volume = month_Volume.drop(['TSLA'], axis=1)

# Defining the time in years to be analized untill today.
total_years = 5
shif = 0
start_date = datetime.now() - timedelta(days=365*total_years) - \
    timedelta(days=shif)
end_date = datetime.now() - timedelta(days=shif)
period = '3y'
interval = '1d'


'''
This is the main function to control the code

Inputed Parameters
----------

Returns
-------

Notes
-------

'''


def run():

    # Declare and choose *one* of the list of stocks to be analyzed.
    list_of_stoks = 'data_input/IBRA_list.txt',\
                    'data_input/S&P500_list.txt', \
                    'data_input/SMALL_S&P600_list.txt', \
                    'data_input/cripto_list.txt', \
                    'data_input/NIKKEI_list.txt'
    list_of_stoks = list_of_stoks[0]
    print(list_of_stoks)
    #print(sd)
    # Creating auxiliary list
    data_resume_aux = []

    # Running get_data function to obtain the OLHC data from the list_of_stoks
    adj_prices_aux, High, aux_indexes, month_AdjClose, month_High,\
         month_Volume = get_data(np.loadtxt(str(list_of_stoks),
                                 dtype=str).T[:], 10, period, interval, True)
    print (month_Volume)
    # Defining the steps in days to compute the percentage change
    days = 1*1  # 5 dias == uma semana

    # Splitting the whole sample into packages of data with the size of dias
    s = [int(i) for i in np.linspace(0, len(adj_prices_aux)-1,
                                     int(len(adj_prices_aux)/days))]

    # Definig a dataframe with the adjusted prices data based on shifts of dias
    df_dias = pd.DataFrame(np.array(adj_prices_aux.values[s]),
                           columns=adj_prices_aux.keys(),
                           index=adj_prices_aux.index[s])
    # Removing bugs for empty data
    # df_dias = df_dias[df_dias.index != '2019-03-06']
    print(df_dias)
    # Defining the number of months or steps of days to compute the \
    # percentage change value
    m = 6*1
    n = 3*1
    p = 1*1

    # Running Percent function.
    # Note that month_AdjClose[:]>0] remove undefined divisions.
    # If we need to use the df_dias data, include it into the Percent function.
    df_p, df_n, df_m, df_mean = \
        Percent(m, n, p, month_AdjClose[:][month_AdjClose[:] > 0])
    # Percent(m, n, p, adj_prices_aux[adj_prices_aux > 0])
    # Removing infinity values from the DataFrame.
    # df_p = df_p[df_p != np.inf]
    # df_n = df_n[df_n != np.inf]
    # df_m = df_m[df_m != np.inf]
    # df_mean = df_mean[df_mean != np.inf]

    print(df_p, df_n, df_m, df_mean)

    # Define what will be the adopted shift in the data as main data analyzed.
    df = df_m
    # Define the volume
    df_vol = month_Volume[m:]

    print(df, df_vol)

    # Loop to simulate buying X stocks for each period m, n or p.
    for j in range(1, 21):
        # print(j)

        # Define the initial capital in the analysis
        capital = 10000
        capital_rand = 10000
        aportes = 0

        # Defining lists to append the data.
        aux_price = []
        cap = []
        stocks_list = []
        step_percent = []

        aux_price_rand = []
        cap_rand = []
        step_percent_rand = []
        stocks_list_rand = []

        # Looping into each price data or percentage change for a certain time
        # (date)
        for i in range(len(df)):

            # Removing nan values from the data. Stocks not listed at this day
            val_remove_nan = [r for r in df.values[i] if r > -9999]

            # Using the function heapq.nlargest to obtan the j highers values
            # from the each row (list).
            maxs = heapq.nlargest(j, val_remove_nan)

            # Return the dataframe indexes
            x = np.in1d(df.values[i], maxs)

            # Creating a random list to be used as a stock random indicator
            R = np.random.randint(low=0, high=len(df.T), size=(len(df.T),))

            # Appending the stocks names for the highest values from the list
            stocks_list.append(np.array(df.keys()[x]))

            # Appending the stocks names for random list
            stocks_list_rand.append(np.array(df.keys()[R[x]]))

            # Using an if statement to use the percentage result as a backtest
            # reporting the profit for the following period
            if (i < len(df) - 1):

                # Printing some usefull information as sanity checks.
                print(maxs)
                print(df.keys()[x])
                print(df.index[i])
                print(df[df.keys()[x]].values[i])
                print(df_p[df.keys()[x]].values[i+1])
                print(np.mean(df_p[df.keys()[x]].values[i+1]))
                print()
                print()

                # defining the list from the highest percentage change values
                ptg_change_aux = df_p[df.keys()[x]].values[i+1]
                ptg_change = [h for h in ptg_change_aux if h > -100]

                # Using the percentage changes to compute the cumulative
                # capital and appending the data for the highest
                # percentage change.
                # If statement to include government taxes.
                if (np.mean(ptg_change) > 0):  # if profit was obtained:
                    acc_capital = capital + capital*np.mean(ptg_change)*0.01 \
                        - (capital*np.mean(ptg_change)*0.01)*0.05 + aportes

                else:
                    acc_capital = capital + capital*np.mean(ptg_change)*0.01 \
                                + aportes

                # Appending the data
                capital = acc_capital
                cap.append(capital)
                step_percent.append(np.mean(ptg_change))
                # aux_mean.append(np.mean(ptg_change))
                aux_price.append(ptg_change)

                # CleanedList is defined to remove the nan created due to non
                # existence of a certain stock in the past
                cleanedList = [x for x in df_p[df.keys()[R[x]]].values[i+1]
                               if str(x) != 'nan']

                # Computing the profit from random trades.
                acc_capital_rand = capital_rand + capital_rand*np.mean(
                                   cleanedList)*0.01 + aportes

                # Appending random data
                capital_rand = acc_capital_rand
                cap_rand.append(capital_rand)
                step_percent_rand.append(np.mean(cleanedList))
                # aux_mean_rand.append(np.mean(cleanedList))
                aux_price_rand.append(cleanedList)

                # Calling Figure function to plot the stocks data
                # Figure(i,df_p.keys()[x],df_p[df_p.keys()[x]][:i+1],
                #        df_p[df_p.keys()[x]][:i+1],df_p[df_p.keys()[x]][:i+1])

        # Calling plot_rentabilidade function to plot profit of the data trades
        capital = 10000
        data_resume = plot_rentabilidade('Sobesobe_' + str(days) + '_' + str(m),  # Name of the Figure
                                         j,
                                         np.array(cap),
                                         return_capital(capital, step_percent)
                                         [0],
                                         np.array(cap_rand),
                                         return_capital(capital, step_percent)
                                         [1],
                                         0,
                                         df[:-1].index,
                                         stocks_list,
                                         np.around(step_percent, 2),
                                         capital,
                                         r'Sobesobe assumindo uma carteira com '
                                         + str(j) + ' ações \nCompras no último'
                                         + ' dia do mês')

        # Appending the resume of the trades over the period
        data_resume_aux.append(np.array(data_resume))

    # Saving the resume of the trades over the period into an ascii file.
    np.savetxt('figures/data_resume_'+str(days)+'_'+str(m)+'.txt',
               np.array(data_resume_aux).T, fmt='%.2f', delimiter='\t')
    # print(np.array(np.around(step_percent, 2)))
    # print(np.array(np.around(step_percent, 2)))
    # Ending the main routine.
    return


def return_capital(capital, data):
    '''

    This function computes the accumulative sum for a certain data (list, array
    ).

    Inputed Parameters
    ----------

    capital: float
        The total capital allocated in the trading.

    data: list, array
        A list or series number of certain data


    Returns
    -------

    soma: np.array()
        The resulting array of the cumulative sum of the data

    accumalative: np.array()
        The resulting array of the cumulative sum of the data considering the
        reinvestiment of the profit.

    Notes
    -------


    '''

    accumalative = []
    soma = capital + capital*np.cumsum(data)*0.01

    for i in range(len(data)):

        acc_capital = capital + capital*data[i]*0.01
        capital = acc_capital
        accumalative.append(acc_capital)

    return np.array(soma), np.array(accumalative)


def Percent(Shift_data_1, Shift_data_2, Shift_data_3, Price):
    '''

    This function computes the percentage change of a list (stock) for a
    certain period, a, b, and c.

    Inputed Parameters
    ----------
    Shift_data_1: Integer
        Include a shift in the data to computed the percentage change.
        Example: if you choose 2, the percentage will be computed
        based on the two last numbers. If you choose 3, the change percentage
        will be computed based on the last 3 data.

    Shift_data_2: Integer
        Include a shift in the data to computed the percentage change. Same
        idea as Shift_data_1

    Shift_data_3: Integer
        Include a shift in the data to computed the percentage change. Same
        idea as Shift_data_1

    Price: pandas.core.frame.DataFrame
        The dataframe with the list/stock price data.
        Example: data['Adjusted Close']

    Returns
    -------
    df_per_1month: pandas.core.frame.DataFrame
        The dataframe with the percentage change compared to the previous data.

    df_per_xmonth: pandas.core.frame.DataFrame
        The dataframe with the percentage change compared to the integer
        defined in the Shift_data inputed parameter.


    Notes
    -------


    '''
    # Defining empty lists to append the data.
    aux_mean = []
    aux_mean2 = []

    # Computing the percentage change based on Shift_data_1.
    per = 100*(np.array(Price)[Shift_data_1:]/np.array(Price)[:-Shift_data_1]
               - 1)
    df_a = pd.DataFrame(np.array(per), columns=Price.keys(),
                        index=Price.index[Shift_data_1:])

    # Computing the percentage change based on Shift_data_2.
    per = 100*(np.array(Price)[Shift_data_2:]/np.array(Price)[:-Shift_data_2]
               - 1)
    df_b = pd.DataFrame(np.array(per), columns=Price.keys(),
                        index=Price.index[Shift_data_2:])

    # Computing the percentage change based on Shift_data_3.
    per = 100*(np.array(Price)[Shift_data_3:]/np.array(Price)[:-Shift_data_3]
               - 1)
    df_c = pd.DataFrame(np.array(per), columns=Price.keys(),
                        index=Price.index[Shift_data_3:])

    # Converting all percentage change into the same length.
    df_a, df_b, df_c = df_a, df_b[Shift_data_1-Shift_data_2:], \
        df_c[Shift_data_1-Shift_data_3:]

    # Printing the data if needed.
    print(df_a, df_b, df_c)

    # Looping into each index a_ij of each dataframe to compute the mean
    # percentage change from the values.
    for row in range(len(df_a)):
        for l in range(len(df_a.values[row])):
            mean = np.average([df_a.values[row][l], df_b.values[row][l],
                              df_c.values[row][l]], weights=[1, 1, 1])

            # printing data if needed.
            # print(df_a.values[row][l], df_a.keys()[l], df_b.values[row][l],
            #      df_b.keys()[l], df_c.values[row][l], df_c.keys()[l])

            # Appending the mean value
            aux_mean.append(mean)
        # Appending the appended aux_mean into a major list
        aux_mean2.append(aux_mean)
        # Cleaning aux_mean
        aux_mean = []

    # Creating a DataFrame for the mean values
    df_mean = pd.DataFrame(np.array(aux_mean2), columns=df_a.keys(),
                           index=df_a.index)
    # print(df_mean)

    # Saving the data into a .csv file.
    df_c.to_csv("data_output/sobesobe_pctchange_1mes.csv",
                sep='\t', float_format='%.2f')
    df_b.to_csv("data_output/sobesobe_pctchange_3mes.csv",
                sep='\t', float_format='%.2f')
    df_a.to_csv("data_output/sobesobe_pctchange_6mes.csv",
                sep='\t', float_format='%.2f')
    df_mean.to_csv("data_output/sobesobe_pctchange_media.csv",
                   sep='\t', float_format='%.2f')

    return df_c, df_b, df_a, df_mean


def plot_rentabilidade(strategy, j, prices, prices2, prices3, prices4, prices5,
                       index, stocks, step_percent, capital, title):
    '''

    This function performs a visualization plot for up to five curves over its
    dates. It is also shown ETF's indexes as a comparison reference.

    Inputed Parameters
    ----------
    strategy: String
        Define the name of the current strategy. Ex: Gambit, max_min, ATR ...

    j: Integer
        An integer value to account for loops and save the figure name with a
        different name.

    prices, prices2, prices3, prices4, prices5: list or np.array
        Up to five main data to be shown in y-axis of the Figure.

    index: list or MDATE
        The index used as a x-axis reference. Index must have the same size as
        prices.

    stocks: list
        List of stocks names.

    step_percent: list or array or dataframe.
        The percentage change of each item of the data. This is the minimum
        percentage change value.

    capital: float.
        The total capital to be allocated in the trading.

    title: string
        The figures' title.


    Returns
    -------
    Accumulative sum of the data: float
        Return the mean value of the accumulative sum of the data

    Accumulative sum of the data reinvested: float
        Return the mean value of the accumulative sum of the data considering
        the reinvestiment of each trade.

    data_std: float
        Return the standard deviation of the mean of the data.

    data_mean: float
        Return the mean of the data.


    Notes
    -------


    '''

    # Downloading ETFs data to be included as a reference in the figure.
    etfs = yf.download(["^BVSP", "SMAL11.SA", "^GSPC", "SLY"],
                       period=str(period),
                       interval=str(interval))

    # define the Date (x.index) as a datetime pandas array
    index.index = pd.DatetimeIndex(index)

    # Define the figure size, shape, and subplots.
    ax1 = plt.figure(1)
    fig, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    plt.subplots_adjust(left=0.08, right=0.94, bottom=0.09, top=0.89,
                        wspace=0.2, hspace=0.2)

    # Defining subplot 1:
    plt.subplot2grid((1, 1), (0, 0))

    # Plotting the data (prices)
    plt.plot(index, prices,  '.-', color='k', lw=2)
    plt.plot(index, prices4, '.-', color='g', lw=1, alpha=0.8)
    plt.plot(index, prices2, '.-', color='b', lw=2)
    plt.plot(index, prices3, '.-', color='purple', lw=2)

    # Plotting the data (ETFs)
    plt.plot(etfs.index,
             (etfs["Adj Close"].values/etfs["Adj Close"].values[1])*capital,
             '-', lw=1, alpha=0.8, zorder=-10)

    # Making legend. Various data are shown in the legend.
    # Defining legends data.
    aux_str1 = ("Rentabilidade com reinvestimento: "
                + str(np.around(prices[-1]/(capital/100) - 100, 2))
                + " %; Média por periodo: "
                + str(np.around(np.mean(step_percent), 2))
                + ' %; Taxa de acerto = '
                + str(np.around(len(step_percent[step_percent > 0])/len(
                    step_percent)*100, 2)) + ' %')
    aux_str2 = ("Rentabilidade com reinvestimento (excluindo IR): "
                + str(np.around(prices4[-1]/(capital/100) - 100, 2))
                + " %")
    aux_str3 = ("Rentabilidade sem reinvestimento: "
                + str(np.around(prices2[-1]/(capital/100) - 100, 2))
                + " %")

    plt.legend((aux_str1, aux_str2, aux_str3,
                "Monkey random picking",
                "IBOV", "SMAL11", "S&P 500", "S&P 600"),
               fontsize=8, loc='upper left', scatterpoints=1, markerscale=1,
               frameon=False)

    # Writing the pergentage changes in the figure.
    # for s in range(len(prices2)):
    # plt.text(index[s],prices[s],'    '+str(step_percent[s])+' % :
    # '+str(stocks[s]),rotation=90,fontsize=6)
    # plt.text(index[s], prices[s], '    ' + str(step_percent[s]) + ' %',
    #         rotation=90, fontsize=6)

    # Setting box
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.tick_params(axis='x', which='minor', top=True, direction='in')
    plt.tick_params(axis='y', which='minor', right=True, direction='in')
    plt.tick_params(axis='x', which='major', top=True, direction='in')
    plt.tick_params(axis='y', which='major', right=True, direction='in')
    plt.minorticks_on()
    plt.tick_params(axis='both', left=True, top=True, right=True, bottom=True,
                    labelleft=True, labeltop=False, labelright=False,
                    labelbottom=True)
    plt.grid(":", linewidth=1, color='gray', alpha=0.3)

    # Setting labels
    plt.xlabel("Months", fontsize=12)
    plt.ylabel("Total profit", fontsize=12)

    # Setting x_y limits and ticks
    plt.ylim(0, 350000)
    # plt.ylim(float(capital)/2, 100000)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Figure title.
    plt.title(str(title))

    # Saving the figure.
    # plt.savefig('celso/Figures/'+str(strategy)+str(j)+'.jpg', dpi=500)
    plt.savefig('figures/'+str(strategy)+'_'+str(j)+'.jpg', dpi=500)
    plt.clf()

    return np.around(np.sum(step_percent), 2), \
        np.around(prices[-1]/100, 2), \
        np.around(np.std(step_percent), 2), \
        np.around(np.mean(step_percent), 2) \



def get_data(list_of_stoks, volume, period, interval, convert_month):
    '''
    This function loads a list of stocks and search in the yahoo
    finances database for the adjusted prices in a certain period.


    Inputed Parameters
    ----------
    list_of_stoks: numpy.ndarray
        Loads a file with the stocks codes to be seached in the database.

    volume: numpy.float64
        The minimum mean volume of the stock in the period to be considered.

    period: str
        The period to obtain the data.

    interval: str
        The interval of the data.

    convert_month: Boolean
        If True, zip the data of days into month. If False, do nothing.


    Returns
    -------
    adj_prices, High: pandas.core.frame.DataFrame
        The dataframe with the adjusted price for all stocks loaded in
        list_of_stoks with volume higher than the adopted volume.

    aux_indexes: pandas DatetimeIndex
        The pandas Date for the defined period.

    month_AdjClose, month_High, month_Volume: pandas.core.frame.DataFrame
        The dataframe with the monthly adjusted price, high, and volume for
        all stocks loaded in list_of_stoks with volume higher than the
        adopted volume.

    Notes
    -------
    Other data can be included in the return function, as low, open...
    Just include it.

    '''

    # Getting stocks data from yahoo finance
    aux_data = yf.download(list(list_of_stoks),
                           period=str(period), interval=str(interval))
    aux_data = aux_data[:-4]                           
    aux_indexes = aux_data.index
    print(aux_indexes)
    # Creating a DataFrame for each OHLC data for each stock analyzed.
    adj_prices = aux_data['Adj Close']
    High = aux_data['High']
    Low = aux_data['Low']
    Open = aux_data['Open']
    Close = aux_data['Close']
    volume = aux_data['Volume']

    # Creating a DataFrame converted from day to month for OHLC data for each
    # stock analyzed.
    print(adj_prices,volume)
    month_AdjClose = convert_to_month(0, adj_prices)
    month_Open = convert_to_month(0, Open)
    month_Low = convert_to_month(0, Low)
    month_Close = convert_to_month(0, Close)
    month_High = convert_to_month(0, High)
    month_Volume = convert_to_month(0, volume)

    # Saving the data
    adj_prices.to_csv("data_output/adj_prices_sobesobe.csv",
                      sep='\t', float_format='%.2f')
    volume.to_csv("data_output/volumes_sobesobe.csv",
                  sep='\t', float_format='%.2f')
    Open.to_csv("data_output/open_sobesobe.csv",
                sep='\t', float_format='%.2f')
    Low.to_csv("data_output/low_sobesobe.csv",
               sep='\t', float_format='%.2f')
    High.to_csv("data_output/high_sobesobe.csv",
                sep='\t', float_format='%.2f')
    Close.to_csv("data_output/close_sobesobe.csv",
                 sep='\t', float_format='%.2f')

    month_AdjClose.to_csv(
        "data_output/month_AdjClose_sobesobe.csv", sep='\t',
        float_format='%.2f')
    month_Open.to_csv("data_output/month_Open_sobesobe.csv",
                      sep='\t', float_format='%.2f')
    month_Low.to_csv("data_output/month_Low_sobesobe.csv",
                     sep='\t', float_format='%.2f')
    month_High.to_csv("data_output/month_High_sobesobe.csv",
                      sep='\t', float_format='%.2f')
    month_Close.to_csv("data_output/month_Close_sobesobe.csv",
                       sep='\t', float_format='%.2f')
    month_Volume.to_csv("data_output/month_Volume_sobesobe.csv",
                        sep='\t', float_format='%.2f')

    # Finishing the routine and returning the data
    return adj_prices, High, aux_indexes, month_AdjClose, month_High, \
        month_Volume


def convert_to_month(shift_days, aux_data):
    '''
    This function converts the stock data from day to month


    Inputed Parameters
    ----------
    shift_days: numpy.integer
        Define the shift in days from today to the past.

    aux_data: pandas.core.frame.DataFrame
        The pandas Dataframe of a stock using one of the OHLC data.
        The data need to be in a daily interval.


    Returns
    -------
    mon_data: pandas.core.frame.DataFrame
        The montly stock dataframe.


    Notes
    -------


    '''

    shift_days = shift_days
    mon_data = pd.DataFrame(aux_data.resample('BM').apply(
                            lambda x: x[-1-shift_days]))
    end_of_months = mon_data.index.tolist()
    end_of_months[-1] = aux_data.index[-1]
    mon_data.index = end_of_months
    mon_data.index = mon_data.index - timedelta(days=shift_days)

    return (mon_data)


if __name__ == '__main__':
    run()
