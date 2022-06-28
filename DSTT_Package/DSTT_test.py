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
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime, timedelta
import argparse
import time
from time import sleep
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import random
from tensorflow import keras


from DSTT_utils import *
from DSTT_model import DSTTModel


interval_type = 'hourly'
show_figures=True
#The number of Monte Carlo Sampling 
MCS=100


models_directory="models"
results_dir='results'
figures_dir='figures'

    
os.makedirs(models_directory,  exist_ok=True)
os.makedirs(results_dir,  exist_ok=True)
os.makedirs(figures_dir,  exist_ok=True)

def test(start_hour, end_hour,models_directory='models',results_dir='results', figures_dir='figures',show_figures=False):
    for k in range(start_hour,end_hour):
        model = DSTTModel()
        print('Running testing for h =', k, 'hour ahead')
        num_hours = k
        max_r2 = -10000
        X_train,y_train, X_test, y_test, X_valid, y_valid,x_dates = load_training_and_testing_data(num_hours,
                                                                                                   interval_type)
        model.set_data(X_train,y_train, X_test, y_test)
           
        log('size of test:', len(X_test), len(y_test), len(x_dates))
        log('y_test max , min:', y_test.max(), y_test.min())
        ax_dates = []
        start_date = x_dates[0]
        end_date = x_dates[-1]

        start_date_ts = datetime(int(start_date[0]), int(start_date[1]), int(start_date[2]),0,0,0)
        end_date_ts = datetime(int(end_date[0]), int(end_date[1]), int(end_date[2]),0,0,0)
        current_date = start_date_ts
        if interval_type == 'hourly':
            for i in range(len(y_test)):
                ax_dates.append(current_date)
                current_date = current_date + timedelta(hours=1)
        else:
            while current_date <= end_date_ts:
                ax_dates.append(current_date)
                current_date = current_date + timedelta(days=1)
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        log('Loading the model and its weights.')
        model.load_model(input_shape, 
                         kl_weight=1/X_train.shape[0],
                         num_hours=6,
                         w_dir=models_directory)
        predictions =  model.predict(X_test)
        file_name = figures_dir +'' + os.sep + 'dst_' +str(num_hours) + interval_type[0] +'.png'
        file_name_aleatoric = figures_dir +'' + os.sep + 'dst_' +str(num_hours) + interval_type[0] +'_aleatoric.png'
        file_name_epistemic = figures_dir +'' + os.sep + 'dst_' +str(num_hours) + interval_type[0] +'_epistemic.png'
        
        figsize = (9,3.5)
        
        predictions_ft = model.uncertainty(X_test,y_test, N=MCS)

        ep = np.concatenate([(predictions - predictions_ft[2]),
                         (predictions_ft[2] + predictions)[::-1]])

        ra = np.concatenate([(predictions - predictions_ft[1]),
                         (predictions_ft[1] + predictions)])
        

        log('Saving results...')
        model.set_preds(predictions)
        model.set_epis(predictions_ft[2])
        model.set_al(predictions_ft[1])
        model.set_dates(ax_dates)
        model.save_results(num_hours,results_dir=results_dir)
        
        if show_figures:
            plot_figure(ax_dates,y_test,predictions,predictions_ft[2],num_hours,label='Dst index',
                        file_name=file_name_epistemic ,wider_size=False,figsize = figsize,
                        interval=interval_type[0],uncertainty_label='wit Epistemic Uncertainty', 
                        fill_graph=True,
                        x_labels=True,
                        x_label='Time\n(a)')
            plot_figure(ax_dates,y_test,predictions,predictions_ft[1],num_hours,label='Dst index',
                        file_name=file_name_aleatoric ,wider_size=False,figsize = figsize,
                        interval=interval_type[0],uncertainty_label='with Aleatoric Uncertainty', fill_graph=True,
                        uncertainty_color='#aabbff',
                        x_labels=True,
                        x_label='Time\n(b)')        


            plt.show()

if __name__ == '__main__':
    starting_hour = 1
    ending_hour = 7
    if len(sys.argv) >= 2 :
        if not sys.argv[1].lower() == 'all':
            starting_hour = int(float(sys.argv[1]))
            ending_hour = starting_hour + 1
    if len(sys.argv) >= 3:
        show_figures = boolean(sys.argv[2])
    
    if starting_hour < 1 or starting_hour-1 > 6:
        print('Invalid starting hour:', starting_hour,'\nHours must be between 1 and 6.')
        exit() 
    print('Starting hour:', starting_hour, 'ending hour:', ending_hour-1)
    test(starting_hour, ending_hour, show_figures)