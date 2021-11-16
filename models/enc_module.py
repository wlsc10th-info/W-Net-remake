import tensorflow as tf
from tensorflow import keras
 


'''
This class is to generate an encoder model

attribute:
    layer_count: the number of Conv2D layers.
    layer_settings[]: the setting of each layer.

method:
    call: call the model.
        inputs: the input data.
    
    get_mid: get the dataflow between each layers.


    ---This class is not finished---
'''



class Encoder(tf.keras.Model):

    def __init__(self, layer_count, layer_settings):
        super(Encoder, self).__init__()
        
        self.layer_count = layer_count
        self.layer_saver = []
        self.mid = []

        for i in range(layer_count):
            pass   #TODO: generate the Conv2D layers with "layer_settings" attribute, and save in self.layer_saver


    def call(self, inputs):
        
        x = inputs
        midtmp = []

        # run through each layer
        for i in range(self.layer_count):
            x = self.layer_saver[i](x)
            
            # save the mid data everytime we call the model
            if i != self.layer_count - 1:
                midtmp.append(x)
        
        self.mid = midtmp
        return x

    # get the dataflow between each layers
    def get_mid(self):
        return self.mid