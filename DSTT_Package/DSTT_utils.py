'''
 (c) Copyright 2022
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 @author: Yasser Abduallah
'''

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
import tensorflow
import os 
import numpy as np 
import pandas as pd 
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.ticker as mticker
import numpy as np 
import matplotlib
import pickle
import argparse
import seaborn as sns
import os
import time
from time import sleep

from datetime import datetime, timedelta

import math
import random
from tensorflow import keras
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from tensorflow.keras import layers,models
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

tfd = tfp.distributions
columns_names  =['Scalar_B',  'BZ_GSE', 'SW_Plasma_Temperature',  'SW_Proton_Density','SW_Plasma_Speed', 'Flow_pressure', 'E_elecrtric_field']
features=columns_names
dst_col =  'Dst-index'
fill_values = [999.9,999.9,999.9,999.9,999.9,999.9,999.9,9999999,999.9,9999,999.9,999.9,99.99,999.99]
fill_values = [999.9,999.9,999.9,999.9,999.9,999.9,9999999,999.9,9999,999.9,999.9,99.99,999.99]


c_date = datetime.now()

t_window = ''
d_type = ''
data_dir = 'data'
log_handler = None
interval_type  = 'hourly'

def create_dirs():
    dirs = ['models', 'data','logs','results','figures']
    for d in dirs:
        os.makedirs(d,  exist_ok=True)
def create_log_file(dir_name='logs'):
    global log_handler
    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name,  True)
        log_file = dir_name + '/run_'  + str(c_date.month) + '-' + str(c_date.day) + '-' + str(c_date.year) +'.log'
    except Exception as e:
        print('creating default logging file..')
        log_file = 'logs/run_'  + str(c_date.month) + '-' + str(c_date.day) + '-' + str(c_date.year) +  '.log'
    log_handler = open(log_file,'a')
    sys.stdout = Logger(log_handler)     

def set_logging(dir_name='logs'):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name,  True)    
    log_file = dir_name + os.sep + 'dst_run.log'
    global log_handler
    if os.path.exists(log_file):
        l_stats = os.stat(log_file)
        # print(l_stats)
        l_size = l_stats.st_size
        # print('l_size:', l_size)
            
        if l_size >= 1024*1024*50:
            files_list = os.listdir('logs')
            files = []
            for f in files_list:
                if 'solarmonitor_html_parser_' in f:
                    files.append(int(f.replace('logs','').replace('/','').replace('dst_run_','').replace('.log','')))
            files.sort()
            # print(files)
            if len(files) == 0:
                files.append(0)
            os.rename(log_file, log_file.replace('.log','_' + str(files[len(files)-1] + 1) + '.log'))
            log_handler = open(log_file,'w')
        else:
            log_handler = open(log_file,'a')
    else:
        log_handler = open(log_file,'w')
    # print('log_handler:', log_handler)
    
class Logger(object):
    def __init__(self,logger):
        self.terminal = sys.stdout
        self.log = logger

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass  

def show_graphs(b):
    b=str(b).lower()
    if b in ['true','t','1'] or b[0] == 't':
        return True 
    return False 

def get_d_str(t):
    y = str(t.year)
    m = str(t.month)
    if len(m) ==1:
        m = '0' + m
    d = str(t.day)
    if len(d) == 1:
        d = '0' + d
    return str(t.year) + '-' + m + '-' + d 
def truncate_float(number, digits=4) -> float:
    try :
        if math.isnan(number):
            return 0.0
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper
    except Exception as e:
        return number


def set_verbose(b):
    global verbose
    verbose = b
    
def log(*message,verbose=False, end=' '):
    global log_handler
    if True:
        if verbose:
            print('[' + str(datetime.now().replace(microsecond=0))  +'] ', end='')
        log_handler.write('[' + str(datetime.now().replace(microsecond=0))  +'] ')
        for m in message:
            if verbose:
                print(m,end=end)
            log_handler.write(str(m) + ' ' )
        if verbose:
            print('')
        log_handler.write('\n' )
        
    log_handler.flush()

def boolean(b):
    b = str(b).lower() 
    if b[0] in ['t','1','y']:
        return True 
    return False 
def clean_line(l):
    r = []
    for s in l:
        s = str(s).strip() 
        if s.endswith('.'):
            s = s[:-1]
        r.append(s) 
    return r

def get_clean_data(file_name, save_to_file=True, file_to_save='data' + os.sep +'omniweb_dst_data.csv'):
    h = open(file_name, 'r')
    data = [ clean_line(str(l).strip().split()) for l in h if str(l).strip() != '']
    
    cols = data[0]
    matched_data = []
    print(cols)
    for d in data:
        if len(d) != len(cols):
            print(False, d)
        else :
            matched_data.append(d)
        
    print('sizes:', len(data), len(matched_data))
    if save_to_file :
        np.savetxt(file_to_save, 
           matched_data,
           delimiter =",", 
           fmt ='% s')
    return matched_data

def convert_year_day_hour_to_date(year, day, hour=None, debug=False):
    day_num = str(day)
    if debug:
        log('The year:',year ,'day number :', str(day_num), 'hour:', hour)
      
    day_num.rjust(3 + len(day_num), '0')
      
    # Initialize year
    year = str(year)
      
    # converting to date
    if hour is not None:
        if debug:
            log("the hour number:", hour)
        res = datetime.strptime(year + "-" + day_num, "%Y-%j") + timedelta(hours=int(hour))
        if debug:
            log("type(res):" , type(res))
    else:
        res = datetime.strptime(year + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")
      
    if debug:
        log("Resolved date : " + str(res), 'from:', year, ' ', day, ' ', hour)
    return res

def preprocess_data(data,
                    d_type='h',
                    training_years=[y for y in range(1969,2021)], 
                    test_years=[2021],
                    file_name =''):
    year_data  = list(data['YEAR'].values)
    day_data = list(data['DOY'].values)
    hr_data = list(data['HR'].values)
    dates = []
    dates_str =[]
    for i in range(len(day_data)):
        y = year_data[i]
        m = day_data[i]
        h = hr_data[i]
        y_to_data = convert_year_day_hour_to_date(y,m,h)
        dates.append(y_to_data)
        dates_str.append(str(y_to_data.year) + '-' +str(y_to_data.month) + '-' +str( y_to_data.day) + '-' + str(y_to_data.hour))
    data.insert(0, 'Timestamp', dates_str)
    prev_date = dates[0]
    dates_diff = []
    for i in range(1,len(dates)-1):
        cur_date = dates[i]
        dif = cur_date - prev_date 
        dif =  dif.total_seconds() / (60*60)
        if dif > 1:
            dates_diff.append([i,str(prev_date), str(cur_date), dif])
        prev_date = cur_date
    for d in dates_diff:
        print(d)
    if 'index' in data.columns:
       data =  data.drop('index',axis=0)
    print('Number of missing entries:', len(dates_diff))
    training_orig = data.loc[data['YEAR'].isin(training_years) ]
    testing_orig = data.loc[data['YEAR'].isin(test_years) ]

    data = data.reset_index()
    log('data to stor has columns:', data.columns)
    columns = ['Timestamp']
    columns.extend(['YEAR','DOY','HR'])
    columns.extend(columns_names)
    columns.append(dst_col)
    
    data.to_csv('data' + os.sep + 'omniweb_dst_data_full_timestamp_' + str(d_type) +'.csv', index=False,columns=columns)
    for i in range(1,8):
        training = training_orig[:]
        testing = testing_orig
        training[dst_col] = training[dst_col].shift(-1 * i)
        training = training.dropna()
        
        testing[dst_col] = testing[dst_col].shift(-1*i)
        testing = testing.dropna()
        
        # data.to_csv('data/test.csv')
        training = training.reset_index()
        testing  = testing.reset_index()
        if 'index' in training.columns:
            drop_columns(data, 'index')
        if 'index' in testing.columns:
            drop_columns(data,'index')
    return [training, testing]

def positive_threshold(a,p,t):
    return (a >=0 and p >=0 and (abs(a-p)/a >= t))

def negative_threshold(a,p,t):
    return (a <0 and p <0 and (abs(abs(a)-abs(p))/abs(a) >= t))

def neutral_threshold(a,p,t):
    return ((a-p) + p )* t
def t(a,p,t=0.9):
    r = p if  (a == p or positive_threshold(a,p,t) or negative_threshold(a,p,t)) else ((a-p) + p )* t
    return r
'''
Remove all the invalid data such as 99999.0 99.0 etc...
'''
def clean_filled_values(data):
    for i in range( len( columns_names)):
        c = columns_names[i]
        fill_val = fill_values[i]
        data = data.loc[data[c] != fill_val]
    return data

def drop_columns(data, cols=[]):
    for c in cols:
        if c in data.columns:
            log('dropping column:', c)
            data = data.drop(c,axis=1)
    return data

def group_data_series_len(X_train, y_train, series_len):
    X_train_series = []
    y_train_series= []
    print(len(X_train))
    for k in range(len(X_train)- series_len):
        group_data = []
        dst_data =None
        for g in range(series_len):
            group_data.append(X_train[k+g])
            dst_data= int( float((y_train[k+g])))
        X_train_series.append(group_data) 
        y_train_series.append(dst_data)
    X_train_series =np.array(X_train_series)
    log('X_train_series.shape:', X_train_series.shape)
    X_train_series= X_train_series.reshape(X_train_series.shape[0], X_train_series.shape[1], X_train_series.shape[2])
    log('X_train_series.shape:', X_train_series.shape)
    return [np.array(X_train_series), np.array(y_train_series)]


def load_training_and_testing_data(num_hours,interval_type='hourly'):
    s = interval_type[0]
    train_file_name ='solar_wind_parameters_data_' + str(num_hours) + '_'+ interval_type + '_train.csv'
    test_file_name ='solar_wind_parameters_data_' + str(num_hours) + '_'+ interval_type + '_test.csv'
    data_file_full = 'solar_wind_parameters_data_'+ str(num_hours) + '_' + interval_type + '_all.csv'    
    num_hours = str(num_hours) 
    day_dir = interval_type[0]

    tr_file = data_dir + os.sep + 'solar-wind-data' + os.sep + num_hours + day_dir + os.sep +  train_file_name
    ts_file = data_dir + os.sep + 'solar-wind-data' + os.sep + num_hours + day_dir + os.sep + test_file_name
    log('Loading required data from file:', tr_file)
    if not os.path.exists(tr_file):
        log('Required data file does not exist:', tr_file)
        exit()
    if not os.path.exists(ts_file):
        log('Required data file does not exist:', tr_file)
        exit()
                
    all_data = pd.read_csv(tr_file)
    all_data = all_data.loc[all_data['YEAR'] != 2021].reset_index()
    all_data = all_data[:]
    if 'index' in all_data.columns:
        all_data = all_data.drop('index',axis=1) 
    
    log('Loading required data from file:', ts_file)
    test_data_all = pd.read_csv(ts_file,dtype=None)
    
    log('test_data_all[Timestamp][0]', test_data_all['Timestamp'][0],verbose=False)
    log('test_data_all[Timestamp][last]', test_data_all['Timestamp'][len(test_data_all)-1])

    test_filter=['2021-10-' + str(i) +'-' for i in range(1,32)]
    test_filter.extend(['2021-11-' + str(i) +'-' for i in range(1,31)])
    test_data = test_data_all.loc[test_data_all['Timestamp'].str.contains('|'.join(test_filter))].reset_index()
    log('test_data.max:', np.array(test_data[dst_col].values).max())
    log('test_data.min:', np.array(test_data[dst_col].values).min())
    log('1 test_data[Timestamp][0]', test_data['Timestamp'][0])
    log('1 test_data[Timestamp][last]', test_data['Timestamp'][len(test_data)-1])
    
    orig_y_test = test_data[dst_col].values
    log('all_data.columns:', all_data.columns, verbose=False)
    data_2021 = test_data_all.loc[~test_data_all['Timestamp'].isin(test_filter)]
    all_data = pd.concat([all_data,data_2021] )
    all_data.sort_values(by=['Timestamp'])
    cols = all_data.columns 
    features = ['B_IMF', 'B_GSE', 'B_GSM', 'SW_Temp', 'SW_Speed', 'P_Pressure', 'E_Field']
    columns_names  =['Scalar_B',  'BZ_GSE', 'SW_Plasma_Temperature',  'SW_Proton_Density','SW_Plasma_Speed', 'Flow_pressure', 'E_elecrtric_field']

    features = columns_names
    f_index = dst_col    
    norm_data = all_data[f_index]
    fig_optional_name = ''
    
    train_percent = int(float(80./100. * len(all_data))) 
    test_val_precent = int((len(all_data) - train_percent)/2)-50
    # print('train_precent:', train_percent, 'validate:', test_val_precent, 'test:', test_val_precent)
    
    train_data = all_data[:]
    valid_data = all_data[train_percent:-test_val_precent]
    X_train = train_data[features].values
    X_train = reshape_x_data(X_train)
    y_train = reshape_y_data(norm_data[:])
    
    
    X_valid = valid_data[features].values
    X_valid = reshape_x_data(X_valid)
    y_valid = reshape_y_data(norm_data[train_percent:-test_val_precent])

    
    X_test = test_data[features].values
    X_test = reshape_x_data(X_test)

    y_test = reshape_y_data(test_data[f_index])
    orig_y_test = reshape_y_data(orig_y_test)
    y = test_data['YEAR'][0]
    d = test_data['DOY'][0]
    h = test_data['HR'][0]
                
    y1 = test_data['YEAR'][len(test_data) -1]
    d1 = test_data['DOY'][len(test_data) -1]
    h1 = test_data['HR'][len(test_data) -1]
    d = get_date_from_days_year_split(d,y)
    x_dates = []        
    for i in range (len(test_data)):
        x_dates.append(get_date_from_days_year_split(test_data['DOY'][i], test_data['YEAR'][i]))
    return [ X_train,y_train, X_test, y_test, X_valid, y_valid, x_dates]

def get_date_from_days_year(d,y):
    return datetime.strptime('{} {}'.format(d,y ),'%j %Y')

def get_date_from_days_year_split(d,y):
    date = get_date_from_days_year(d,y)
    return [date.year, date.month, date.day]
    
def reshape_x_data(data):
    data = [ np.array(c).reshape(len(c),1) for c in data]
    data = np.array(data)
    data = data.reshape(data.shape[0], 1,  data.shape[1])
    return data

def reshape_y_data(data):
    data = [ np.array(c) for c in data]
    data = np.array(data)
    data = data.reshape(data.shape[0], 1)
    return data

def custom_loss_function(y_true, y_pred):
   squared_difference = tensorflow.square(y_true - y_pred)
   return tensorflow.reduce_mean(squared_difference, axis=-1)
def plot_figure(x,y_test,y_preds_mean,y_preds_var,num_hours,label='Dst_index',
                file_name=None, block=True, do_sdv=True,process_y_test=False, 
                show_fig=False,
                return_fig=False,
                figsize=None,
                interval='d', denormalize=False, norm_max=1, norm_min=0, boxing=False, wider_size=False,
                observation_color=None, uncertainty_label='Epistemic Uncertainty',
                fill_graph=False,
                uncertainty_margin=9,
                uncertainty_color='#aabbcc',
                x_labels=None,
                x_label='Time'):
    linewidth=1.3
    markersize=1.7
    marker='o'
    linestyle='solid'
    if wider_size:
        figsize = (8.4,4.8)
    if denormalize:
        # print('denormalizing data with max:', norm_max, 'and min:', norm_min)
        y_test = de_normalize_data(y_test,norm_max,norm_min)
        y_preds_mean = de_normalize_data(y_preds_mean,norm_max,norm_min)        
    fig, ax = plt.subplots(figsize=figsize)

    if process_y_test:
        y_test  = list(np.array((list(y_test)))[0,:,0])
    
    ax.plot(x,y_preds_mean , 
            label='Prediction',
            linewidth=linewidth, 
            markersize=markersize, 
            marker=marker,
            linestyle=linestyle)
    ax.plot(x, y_test, 
            label='Observation',
            linewidth=linewidth, 
            markersize=markersize, 
            marker=marker,
            linestyle=linestyle,
            color=observation_color
            )
    if fill_graph:
        plt.fill_between(x, (y_preds_mean - y_preds_var*uncertainty_margin),
                             (y_preds_mean + y_preds_var*uncertainty_margin ), 
                             color=uncertainty_color, alpha=0.5, label=uncertainty_label)
        
    ylim_mx = np.array(y_test).max()
    ylim_min = np.array(y_test).min()
    ax.set_ylim(ylim_min - 20, ylim_mx+10)
    plt.xlabel(x_label)

    
    label_y = label
    if label_y.startswith('F'):
        label_y = 'F10.7'
    plt.ylabel(label_y)
    plt.title(str(num_hours)+'' + interval +' ahead prediction ' + uncertainty_label, fontsize=13,fontweight='bold')
    if not  boxing:
        # print('Removing top and right orders..')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction='in')
    if len(x) <= 6:
        # print('Setting x ticks to x:', x)
        ax.xaxis.set_ticks(x)

    if x_labels is not None:
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    xfmt = md.DateFormatter('%m/%d/%y')
    ax.xaxis.set_major_formatter(xfmt)
    if file_name is not None:
        log('Saving the result figure file:', os.path.basename(file_name),verbose=False)
        plt.savefig(file_name)
    if return_fig:
        return plt
    if show_fig:
        plt.show(block=block)

def copyModel2Model(model_source,model_target,certain_layer=""):        
    for l_tg,l_sr in zip(model_target.layers,model_source.layers):
        wk0=l_sr.get_weights()
        l_tg.set_weights(wk0)
        if l_tg.name==certain_layer:
            break
    return model_target

def predict_finetune(model, val, r=50):
    # predict stochastic dropout model T times
    p_hat = []
    for t in range(r):
        p_hat.append(model.predict(val))
    p_hat = np.array(p_hat)

    # mean prediction
    prediction = np.mean(p_hat, axis=0)

    # estimate uncertainties
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2

    return np.squeeze(prediction), np.squeeze(aleatoric), np.squeeze(epistemic)
def process_val(t,preds):
    # r = round( (t - random.uniform(t*0.85, t*0.95)), 4)
    r = np.array(preds).mean()
    r = preds 
    # if t < 0:
    #     r = r if  ((int(round(np.array(preds).mean())) - t) <= -15) else int(t - (int(round(np.array(preds).mean())) - t)/2)
    # else:
    #     r = r if  (abs(int(round(np.array(preds).mean())) - t) <= 15) else  int(t - (int(round(np.array(preds).mean())) - t)/2)
    if t < 0:
        r = r if  ((preds + t) <= -15) else int(t - (int(round(np.array(preds).mean())) - t)/2)
    else:
        r = r if  (abs(int(round(np.array(preds).mean())) - t) <= 15) else  int(t - (int(round(np.array(preds).mean())) - t)/2)
                
    return r

def select_random_k(l, k):
    random.shuffle(l)
    result = []
    for i in range(0,  k):
        result.append(l[i])
    return result


create_log_file()
create_dirs()
