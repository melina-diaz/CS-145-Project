from os import stat
import numpy as np
#from numpy.lib.shape_base import _column_stack_dispatcher
import pandas as pd
from pandas.core.frame import DataFrame

# Extract subset of database to file based on the STATE and Field
# Ex. makeCSVfile (f.txt, Alabama, Confirmed) will extract the confirmed data of Alabama, save to f.txt.
def makeCSVfile (filename, StateName, field):
    df = pd.read_csv('train_trendency.csv')
    df = df[df['Province_State'].isin([StateName])]
    df = df[field]
    df.to_csv(filename)
    return

# Extract subset of database to file based on the STATE and Field,
# calcualte the difference between every two consecutive days (which is dF/dt, where dt = 1 day)
# save the result to f.txt.
def makeCSVfile_diff (filename, StateName, field):
    df = pd.read_csv('train_trendency.csv')
    df = df[df['Province_State'].isin([StateName])]
    df = df[field]
    pf = df.values
    pf_2 = np.delete(pf, 0, 0)
    pf = np.delete(pf, pf.size-1, 0)
    pf_diff = pf_2 - pf
    df_diff = pd.DataFrame(pf_diff, columns = [field])
    df_diff.to_csv(filename)
    return

# Read the data from the file named FILENAME
# Return a numpy array contains data of the specific FIELD of the STATE
def state_field(filename, StateName, field):
    df = pd.read_csv(filename)
    df = df.drop("Unnamed: 0", axis=1)
    df = df[df['Province_State'].isin([StateName])]
    #df = df[df['Province_State'] == StateName].fillna(df[df['Province_State'] == StateName].mean())
    df = df[field]
    pf = df.values
    return pf

# Read the data from the file named FILENAME
# Return a numpy array contains data of the difference between every two consecutive days
# of the specific FIELD of the STATE (which is dF/dt, where dt = 1 day)
def state_field_diff(filename, StateName, field):
    df = pd.read_csv(filename)
    df = df.drop("Unnamed: 0", axis=1)
    df = df[df['Province_State'].isin([StateName])]
    #df = df[df['Province_State'] == StateName].fillna(df[df['Province_State'] == StateName].mean())
    df = df[field]
    pf = df.values
    pf_2 = np.delete(pf, 0, 0)
    pf = np.delete(pf, pf.size-1, 0)
    pf_diff = pf_2 - pf
    return pf_diff

# Takes a one dimension Numpy array.
# Return a length-2 list [a, b]
# where a and b is parameter of regression function y = a*x + b  
def RegressionFunctionParams(datasetNpArr):
    day = []
    for x in range(1, datasetNpArr.size+1):
        day.append(x)
    day = np.array(day)

    y_average = np.average( datasetNpArr)
    x_average = np.average(day)
    x_squaresum = np.sum(day**2)
    xy_sum = np.sum(datasetNpArr * day)

    # regression function: y = ax + b, where x is day, y is the change rate of the field in the State
    a = (xy_sum - day.size * x_average * y_average)/(x_squaresum - day.size * x_average**2)
    b = y_average - a * x_average
    param = [a, b]
    return param


def CubicRegression(StateName, filed):
    pf = state_field_diff('train_trendency.csv', StateName, filed)
    pf_2 = np.delete(pf, 0, 0)
    pf = np.delete(pf, pf.size-1, 0)
    pf_diff = pf_2 - pf
    #df_diff = pd.DataFrame(pf_diff, columns=[filed])
    params = RegressionFunctionParams(pf_diff)
    predict_sec_diff = []
    for x in range(1, len(pf_diff)+1+30):
        predict_sec_diff.append(params[0]*x + params[1])
    predict_diff = [pf[0]]
    for x in predict_sec_diff:
        predict_diff.append(predict_diff[len(predict_diff)-1] + x)

    predict = [state_field('train_trendency.csv', StateName, filed)[0]]
    for x in predict_diff:
        predict.append(predict[len(predict)-1] + x)

    Predict_30Days = predict[-30:]
    Predict_30Days = pd.DataFrame(Predict_30Days, columns=[filed])
    return Predict_30Days


# save all prediction data to file
# output_format = 'test' => save data of Province_State, Date, Confirmed and Deaths
# output_format = 'submission' => only save data of Confirmed and Deaths
def CubicRegModel_resultSaveTo (filename, output_format):
    df = pd.read_csv('train_trendency.csv')
    date_list = state_field('test.csv', 'Alabama', 'Date')
    df_date = pd.DataFrame(date_list, columns = ['Date'])
    states = df['Province_State'][0:50]
    output = []

    for x in states:
        output.append(df_date)
        output[len(output) - 1]['Province_State'] = x
        output[len(output) - 1] = pd.concat([output[len(output)-1],CubicRegression( x ,'Confirmed') ], axis=1, join ='inner')
        output[len(output) - 1] = pd.concat([output[len(output)-1],CubicRegression( x ,'Deaths') ], axis=1, join ='inner')

    output = pd.concat(output)
    output = output.sort_values(['Date' , 'Province_State'])
    output = output.reset_index(drop=True) 
    output.index.names = ['ID']
    if output_format == 'test':
        output = output[['Province_State', 'Date', 'Confirmed', 'Deaths']]
    elif output_format == 'submission':
        output = output[[ 'Confirmed', 'Deaths']]
    
    print(output)
    output.to_csv(filename)

CubicRegModel_resultSaveTo('./result_Cubic_Reg.csv', 'submission')