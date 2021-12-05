import tensorflow as tf
from . import enc_module, dec_module

class W_net(tf.keras.Model):

    def __init__(self):
        super(W_net, self).__init__()
        
        self.layer_count = 6
        self.enc_p_layer_settings = []
        self.enc_r_layer_settings = []
        self.dec_layer_settings = []

        self.enc_p_layer_settings.append({'filters' : 64,  'kernel_size' : 5, 'strides' : 2})
        self.enc_p_layer_settings.append({'filters' : 128, 'kernel_size' : 5, 'strides' : 2})
        self.enc_p_layer_settings.append({'filters' : 256, 'kernel_size' : 5, 'strides' : 2})
        self.enc_p_layer_settings.append({'filters' : 512, 'kernel_size' : 5, 'strides' : 2})
        self.enc_p_layer_settings.append({'filters' : 512, 'kernel_size' : 5, 'strides' : 2})
        self.enc_p_layer_settings.append({'filters' : 512, 'kernel_size' : 5, 'strides' : 2})

        self.enc_r_layer_settings.append({'filters' : 64,  'kernel_size' : 5, 'strides' : 2})
        self.enc_r_layer_settings.append({'filters' : 128, 'kernel_size' : 5, 'strides' : 2})
        self.enc_r_layer_settings.append({'filters' : 256, 'kernel_size' : 5, 'strides' : 2})
        self.enc_r_layer_settings.append({'filters' : 512, 'kernel_size' : 5, 'strides' : 2})
        self.enc_r_layer_settings.append({'filters' : 512, 'kernel_size' : 5, 'strides' : 2})
        self.enc_r_layer_settings.append({'filters' : 512, 'kernel_size' : 5, 'strides' : 2})

        self.dec_layer_settings.append({'filters' : 512, 'kernel_size' : 5, 'strides' : 2})
        self.dec_layer_settings.append({'filters' : 512, 'kernel_size' : 5, 'strides' : 2})
        self.dec_layer_settings.append({'filters' : 256, 'kernel_size' : 5, 'strides' : 2})
        self.dec_layer_settings.append({'filters' : 128, 'kernel_size' : 5, 'strides' : 2})
        self.dec_layer_settings.append({'filters' : 64,  'kernel_size' : 5, 'strides' : 2})
        self.dec_layer_settings.append({'filters' : 1,   'kernel_size' : 5, 'strides' : 2})
        
        self.enc_p = enc_module.Encoder(self.layer_count, self.enc_p_layer_settings)
        self.enc_r = enc_module.Encoder(self.layer_count, self.enc_r_layer_settings)
        self.dec = dec_module.Decoder(self.layer_count, self.dec_layer_settings)


    def call(self, inputs_p, inputs_r):

        enc_out_p = self.enc_p(inputs_p)
        enc_out_r = self.enc_r(inputs_r)
        enc_out = tf.concat([enc_out_p, enc_out_r], len(enc_out_p.shape)-1)
        output = self.dec(enc_out, self.enc_p.get_mid(), self.enc_r.get_mid())

        return output

