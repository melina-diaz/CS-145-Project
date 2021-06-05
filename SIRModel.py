from os import stat
import numpy as np
#from numpy.lib.shape_base import _column_stack_dispatcher
import pandas as pd

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


def length_of_valid_data (pd_arr):
    len = 0
    for x in pd_arr:
        if np.isnan(x):
            return len
        len = len + 1
    return len

def absent_data_makeup_by_regression(state_name, field):
    pf_diff = state_field_diff('train_trendency.csv', state_name , field)
    cutted_pf_diff = pf_diff[0 : length_of_valid_data (pf_diff)]
    params = RegressionFunctionParams(cutted_pf_diff)
    predict_diff = []
    x = 1
    for i in range(0, len(pf_diff)):
        predict_diff.append(params[0]*x+params[1])
        x = x + 1
    pf = state_field('train_trendency.csv', state_name , field)
    predict_result = [pf[0]]
    predict_result.append(pf[0] + predict_diff[0])
    for i in range(1, len(pf_diff)):
        predict_result.append(predict_result[len(predict_result)-1] + predict_diff[i])
    
    for i in range(0, len(pf)):
        if np.isnan(pf[i]):
            pf[i] = predict_result[i]
    
    return pf


def delta_I(StateName):
    pf_I = absent_data_makeup_by_regression(StateName, 'Active')
    pf_I_2 = np.delete(pf_I, 0, 0)
    pf_I = np.delete(pf_I, pf_I.size-1, 0)
    delta_I = pf_I_2 - pf_I
    return delta_I

def delta_R(StateName):
    pf_R_diff = state_field_diff('train_trendency.csv', StateName, 'Confirmed') - state_field_diff('train_trendency.csv', StateName, 'Active')
    cutted_pf_R_diff = pf_R_diff[0 : length_of_valid_data (pf_R_diff)]
    params = RegressionFunctionParams(cutted_pf_R_diff)
    predict_R_diff = []
    x = 1
    for i in range(0, len(pf_R_diff)):
        predict_R_diff.append(params[0]*x+params[1])
        x = x + 1

    pf_R = state_field('train_trendency.csv', StateName, 'Confirmed') - state_field('train_trendency.csv', StateName, 'Active')
    makeup_R = [pf_R[0]]
    #makeup_R.append(pf_R[0] + predict_R_diff[0])
    for i in range(1, len(pf_R)):
        makeup_R.append(makeup_R[len(makeup_R)-1] + predict_R_diff[i-1])
    
    for i in range(0, len(pf_R)):
        if np.isnan(pf_R[i]):
            pf_R[i] = makeup_R[i]

    pf_R_2 = np.delete(pf_R, 0, 0)
    pf_R = np.delete(pf_R, pf_R.size-1, 0)
    delta_R = pf_R_2 - pf_R
    return delta_R



def delta_S(StateName):
    int_population = int(state_field('state_population.csv', StateName, 'Population'))
    pf_confirmed = state_field('train_trendency.csv', StateName, 'Confirmed')
    pf_S = []
    for i in range(0, len(pf_confirmed)):
        pf_S.append(int_population - pf_confirmed[i])

    pf_S = np.array(pf_S)
    pf_S_2 = np.delete(pf_S, 0, 0)
    pf_S = np.delete(pf_S, pf_S.size-1, 0)
    delta_S = pf_S_2 - pf_S
    return delta_S


def SIR_Model (StateName):
    I = absent_data_makeup_by_regression(StateName, 'Active')
    delta_r = delta_R(StateName)
    delta_s = delta_S(StateName)
    I = np.delete(I, I.size-1, 0)
    
    int_population = int(state_field('state_population.csv', StateName, 'Population'))
    pf_confirmed = state_field('train_trendency.csv', StateName, 'Confirmed')
    S = []
    for i in range(0, len(pf_confirmed)):
        S.append(int_population - pf_confirmed[i])
    S = np.array(S)
    S = np.delete(S, S.size-1, 0)

    r = delta_s * -1 / I * S
    
    
    #a = delta_r / I
    r = pd.DataFrame(r, columns = ['r'])
    return r


df = pd.read_csv('train_trendency.csv')
states = df['Province_State'][0:50]
for x in states:
    pf = SIR_Model (x)
    pf.to_csv('./rr/' + x + '_r.csv')


