
'''
This class is to generate an decoder model

attribute:
    layer_count: the number of Conv2DTranspose layers.
    layer_settings[]: the setting of each layer.
        dict format: [filters, kernel_size, strides]

method:
    call: call the model.
        inputs: the input data.
        mid1, mid2: mid data from both encoder

    ---This class is not finished---
'''


import tensorflow as tf
from tensorflow import keras



class Decoder(tf.keras.Model):

    def __init__(self, layer_count, layer_settings):
        super(Decoder, self).__init__()

        self.layer_count = layer_count
        self.layer_saver = []

        for i in range(layer_count):
            self.layer_saver.append(keras.layers.Conv2DTranspose(
                                                filters=layer_settings[i]['filter'],
                                                kernel_size=layer_settings[i]['kernel_size'],
                                                strides=(layer_settings[i]['strides'], layer_settings[i]['strides']),
                                                padding='same'))

    def call(self, inputs, mid1, mid2):

        x = inputs

        # run through each layer
        for i in range(self.layer_count):

            x = self.layer_saver[i](x)
            if i != self.layer_count - 1:
                x = keras.layer.BatchNormalization()(x)
                x = keras.layers.LeakyReLU()(x)
            
            # concat the mid data from encoder(s)
            if i != self.layer_count - 1:
                x = tf.concat([x, mid1[-(i+1)], mid2[(i+1)]], len(x.shape)-1)
            print(f"round{i}: shape = {x.shape}")
        
        return x