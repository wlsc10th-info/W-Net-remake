
'''
This class is to generate an encoder model

attribute:
    layer_count: the number of Conv2D layers.
    layer_settings[]: the setting of each layer.
        dict format: [filters, kernel_size, strides]

method:
    call: call the model.
        inputs: the input data.
    
    get_mid: get the dataflow between each layers.


    ---This class is not finished---
'''

#TODO: residual blocks


import tensorflow as tf
from tensorflow import keras



class Encoder(tf.keras.Model):

    def __init__(self, layer_count, layer_settings):
        super(Encoder, self).__init__()
        
        self.layer_count = layer_count
        self.layer_saver = []
        self.mid = []

        for i in range(layer_count):
            self.layer_saver.append(keras.layers.Conv2D(
                                                filters=layer_settings[i]['filter'],
                                                kernel_size=layer_settings[i]['kernel_size'],
                                                strides=(layer_settings[i]['strides'], layer_settings[i]['strides']),
                                                padding='same'))


    def call(self, inputs):
        
        x = inputs
        midtmp = []

        # run through each layer
        for i in range(self.layer_count):

            x = self.layer_saver[i](x)
            if i != self.layer_count - 1:
                x = keras.layer.BatchNormalization()(x)
                x = keras.layers.LeakyReLU()(x)
            
            # save the mid data everytime we call the model
            if i != self.layer_count - 1:
                midtmp.append(x)

        self.mid = midtmp
        return x

    # get the dataflow between each layers
    def get_mid(self):
        return self.mid