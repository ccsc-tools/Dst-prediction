'''
Custom Monte Carlo Dropout class to avoid using training=True in all training parts.
This is to avoid dropout in any unnecessary layers.
It implements the keras dropout layer.
'''
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')

from tensorflow.keras.layers import Dropout
class DSTDropout(Dropout):
    def call (self, inputs):
        return super().call(inputs, training=True)