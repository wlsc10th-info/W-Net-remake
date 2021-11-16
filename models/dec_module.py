import tensorflow as tf
from tensorflow import keras



'''
This class is to generate an decoder model

attribute:
    layer_count: the number of Conv2DTranspose layers.
    layer_settings[]: the setting of each layer.

method:
    call: call the model.
        inputs: the input data.
        mid1, mid2: mid data from both encoder

    ---This class is not finished---
'''



class Decoder(tf.keras.Model):

    def __init__(self, layer_count, layer_settings):
        super(Decoder, self).__init__()

        self.layer_count = layer_count
        self.layer_saver = []

        for i in range(layer_count):
            pass   #TODO: generate the Conv2DTranspose layers with "layer_settings" attribute, and save in self.layer_saver

    def call(self, inputs, mid1, mid2):

        x = inputs

        # run through each layer
        for i in range(self.layer_count):
            x = self.layer_saver[i](x)
            
            # concat the mid data from encoder(s)
            if i != self.layer_count - 1:
                x = tf.concat([x, mid1[-(i+1)], mid2[(i+1)]], 1)
            print(f"round{i}: shape = {x.shape}")
        
        return x