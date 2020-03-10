# Flight Price Predictor

# Here, we use attempt to use the ARIMA Model but take the average of only one flight
# everyday to avoid all the problems we encountered the last time

# Importing the libraries

import re
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

def AULVJ(flightValue):
    # Importing the dataset
    dataset = pd.read_csv('fliable_app/predictions/Deals5.csv')
    
    # Dropping rows that have literally no content
    rows, columns = dataset.shape
    half_count = columns/2
    dataset = dataset.dropna(thresh=half_count, axis=0)      # axis is optional
    dataset.to_csv("Deals_Mod1.csv")
    
    # Replacing NaN with 0
    dataset = dataset.fillna(0)
    
    # Saving progress
    dataset.to_csv("Deals_Mod2.csv")
    dataset = pd.read_csv('Deals_Mod2.csv')
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    # Merging record date, month and year to one column
    dateOfRec = list()
    for i in range(len(dataset)):
        dd = int(dataset['recording_date'][i])
        mm = int(dataset['recording_month'][i])
        yyyy = int(dataset['recording_year'][i])
        tempDate = date(yyyy, mm, dd)
        dateOfRec.append(tempDate)
    dataset.insert(0, 'record_date', dateOfRec)
    dataset.drop(['recording_date', 'recording_month', 'recording_year'], axis=1, inplace=True)
    dataset.to_csv('Deals_Mod4.csv')
    dataset = pd.read_csv('Deals_Mod4.csv')
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    # Converting recording date to number of days to departure
    days_to_depart = list()
    for i in range(len(dataset)):
        dateOfRec = datetime.strptime(dataset['record_date'][i], '%Y-%m-%d').date()
        dateOfDept = date(2020, 3, 31)
        daysToDept = dateOfDept - dateOfRec
        days_to_depart.append(daysToDept.days)
    dataset.insert(1, "days_to_depart", days_to_depart)
    dataset.to_csv('Deals_Mod3.csv')
    dataset = pd.read_csv('Deals_Mod3.csv')
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    # Finding dates on which scraping was not done
    days_included = list()
    all_days = list(range(23,107))
    all_days.sort(reverse = True)
    for i in dataset['days_to_depart']:
        if i not in days_included:
            days_included.append(i)
    missing_days = set(all_days) - set(days_included)
    print(sorted(list(missing_days)))
    
    # Creating content for missing days
    new_dataset = pd.DataFrame()
    temp = pd.DataFrame()
    for day in all_days:
        temp = dataset[dataset.days_to_depart == day]
        temp = temp.reset_index()
        if day == 89 or day == 37 or day == 31:
            currentDateString = temp.at[0, 'record_date']
            currentDateDate = datetime.strptime(currentDateString, '%Y-%m-%d').date()
            newDate = currentDateDate - timedelta(1)
            new_dataset = new_dataset.append(temp.replace({day : day+1, currentDateString : newDate}))
        new_dataset = new_dataset.append(temp)
        if day-1 in missing_days and temp.empty == False:
            currentDateString = temp.at[0, 'record_date']
            currentDateDate = datetime.strptime(currentDateString, '%Y-%m-%d').date()
            newDate = currentDateDate + timedelta(1)
            temp.replace({day : day-1, currentDateString : newDate}, inplace = True)
            new_dataset = new_dataset.append(temp)
    print(new_dataset)
    
    # Save progress
    new_dataset.to_csv('Deals_Mod4.csv')
    dataset = pd.read_csv('Deals_Mod4.csv')
    dataset.drop(['Unnamed: 0', 'index'], axis=1, inplace=True)
    
    # Merging normal price and discount price to get one target variable
    dataset['final_price'] = 0
    dataset['final_price'] = dataset['normal_price'] + dataset['discounted_price']
    
    # Removing unneeded columns and saving progress
    dataset = dataset.drop(dataset.columns[[2, 3, 6, 7, 8, 12, 13]], axis=1)
    dataset.to_csv('Deals_Mod5.csv')
    dataset = pd.read_csv('Deals_Mod5.csv')
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    # Selecting matrix of features
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    
    # Selecting only a particular flight's details
    X_axis = list()
    Y_axis = list()
    counter = 0
    for row in X[:, [0,3]]:             # change first parameter between 1 and 0 to take date vs. days_to_depart
        if row[1] == flightValue:
            X_axis.append(row[0])
            Y_axis.append(Y[counter])
        counter += 1
    
    # Taking avaerage for days where it was recorded more than once
    X_plot = list()
    Y_plot = list()
    i = 0
    while i < len(X_axis):
        temp_sum = [Y_axis[i]]
        while i < len(X_axis)-1 and X_axis[i] == X_axis[i+1]:
            temp_sum.append(Y_axis[i+1])
            i += 1
        X_plot.append(X_axis[i])
        Y_plot.append(statistics.mean(temp_sum))
        i += 1
    
    # Creating indexed dataset to check rolling statistics
    # First we check all prices every day and show the mean of everyday's prices
    indexed_Dataset = pd.DataFrame(data = X_plot, columns = ['date'])
    indexed_Dataset.insert(1,'price',Y_plot)
    indexed_Dataset = indexed_Dataset.set_index('date')
    
    if len(indexed_Dataset) < 73:
        return 0                    # Insufficient datapoints to make a prediction
    
    # Estimating the trend i.e. taking log to lower rate at which rolling mean increases
    indexed_Dataset_logScale = np.log(indexed_Dataset)
    
    # Now try another method - shift the values in time series to use in forecasting
    # to see if a better solution exists
    datasetLogDiffShifting = indexed_Dataset_logScale - indexed_Dataset_logScale.shift()
    datasetLogDiffShifting.dropna(inplace = True)
    # Again, may or may not perform better than subtracting the rolling mean
    
    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(indexed_Dataset_logScale, order = (2, 1, 2))
    results_ARIMA = model.fit(disp = -1)
    plt.plot(datasetLogDiffShifting)
    plt.plot(results_ARIMA.fittedvalues, color = 'red')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['price'])**2))
    
    # Fitting them in a combined model
    # Convering fitted values into a Series format
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
    print(predictions_ARIMA_diff.head())
    
    # Convert to cumulative sum
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    print(predictions_ARIMA_diff_cumsum.head())
    
    predictions_ARIMA_log = pd.Series(indexed_Dataset_logScale['price'].iloc[0], \
                                      index = indexed_Dataset_logScale.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value = 0)
    predictions_ARIMA_log.head()
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    plt.plot(indexed_Dataset, c = 'blue', label = 'Original')
    plt.plot(predictions_ARIMA, c = 'red', label = 'Model')
    plt.legend()
    
    # Final Prediction
    results_ARIMA.plot_predict(1,104).savefig('./static/img/ARIMA_Results.png')
    x = results_ARIMA.forecast(steps = 22)
    futurePrices = list(np.exp(x[0]))
    return_value = []
    return_string = []
    
    output_string = ["FUTURE VALUES OF PRICES ARE:"]
    for i in range(21, 0, -1):
        output_string.append(str(i) + " DAY(S) TO DEPARTURE\t:\tRS. " + str(round(futurePrices[21 - i], 2)))
    minFuturePrice = round(min(futurePrices), 2)
    minFuturePriceIndex = futurePrices.index(min(futurePrices))
    todaysPrice = round(np.exp(indexed_Dataset_logScale.tail(1)).iloc[0,0], 2)
    str1=""
    if todaysPrice < minFuturePrice:
        str1 = 'PRICES ARE UNLIKELY TO DECREASE IN THE COMING DAYS'
    else:
        str1 = "LEAST POINT IS RS.", str(minFuturePrice)+ "AT INDEX "+ str(minFuturePriceIndex)
        str1 = str1 , "WE PREDICT THE PRICE WILL BE THE LEAST "+ str(21-minFuturePriceIndex)+ "DAY(S) BEFORE DEPARTURE."

    
    print(output_string)
    return_value.append(str1)
    return_value.append(futurePrices)
    return return_value

# MAIN DRIVER CODE
# fv = 'AI501'            # G8116, SG8906, 6E2132, AI501, SG8718, SG6351/SG6702
# futurePrices = AULVJ(fv)
# print(futurePrices)