import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras.layers import Layer, Convolution2D, InputSpec, BatchNormalization, Activation


class SelfAttention(Layer):
    def __init__(self, nb_filters_in=3840, init='normal', **kwargs):
        self.init = initializers.get(init)
        self.nb_filters_in = nb_filters_in
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.input_spec = [InputSpec(shape=input_shape)]

        self.query_conv = Convolution2D(int(self.nb_filters_in/8), (1, 1), border_mode='same', bias=True, init=self.init)
        self.key_conv = Convolution2D(int(self.nb_filters_in/8), (1, 1), border_mode='same', bias=True, init=self.init)
        self.value_conv = Convolution2D(self.nb_filters_in, (1, 1), border_mode='same', bias=True, init=self.init)

        self.query_conv.build((input_shape[0], self.nb_filters_in, input_shape[2], input_shape[3]))
        self.key_conv.build((input_shape[0], self.nb_filters_in, input_shape[2], input_shape[3]))
        self.value_conv.build((input_shape[0], self.nb_filters_in, input_shape[2], input_shape[3]))

        self.query_conv.built = True
        self.key_conv.built = True
        self.value_conv.built = True
		
        self.trainable_weights = []
        self.trainable_weights.extend(self.query_conv.trainable_weights)
        self.trainable_weights.extend(self.key_conv.trainable_weights)
        self.trainable_weights.extend(self.value_conv.trainable_weights)
        self.gama = self.add_weight(name='gama', shape=[1], initializer=tf.constant_initializer(0.0), trainable=True)
        super(SelfAttention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        x_shape = K.shape(x)
        proj_query = tf.transpose(K.reshape(self.query_conv(x), (x_shape[0], -1, x_shape[2] * x_shape[3])), [0, 2, 1])
        proj_key = K.reshape(self.key_conv(x), (x_shape[0], -1, x_shape[2] * x_shape[3]))

        energy = K.batch_dot(proj_query, proj_key)
        attention = K.softmax(energy)
        proj_value = K.reshape(self.value_conv(x), (x_shape[0], -1, x_shape[2] * x_shape[3]))

        out = K.batch_dot(proj_value, tf.transpose(attention, [0, 2, 1]))
        out = K.reshape(out, (x_shape[0], x_shape[1], x_shape[2], x_shape[3]))
        out = out*self.gama + x
        return out
