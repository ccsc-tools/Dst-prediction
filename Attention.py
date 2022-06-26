'''
        
        @author: Yasser Abduallah
'''
from keras.layers import Layer
import keras.backend as K
class Attention(Layer):
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)
        Attention.name='FocusedAttention'

    def build(self,input_shape): 
        """
        Matrices for creating the context vector.
        """
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(Attention, self).build(input_shape)

    def call(self,x):
        """
        Function which does the computation and is passed through a softmax layer to calculate the attention probabilities and context vector. 
        """
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        """
        For Keras internal compatibility checking.
        """
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        """
        The get_config() method collects the input shape and other information about the model.
        """
        return super(Attention,self).get_config()