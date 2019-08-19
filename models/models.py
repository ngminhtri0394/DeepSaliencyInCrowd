import keras
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Lambda, LeakyReLU, add
from keras.layers import merge, Input, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers.convolutional import AtrousConvolution2D
from keras.layers import BatchNormalization, add, GlobalAveragePooling2D
from keras. models import Model
from keras import backend as K
from utils.configs import *
from .SelfAttentionModule import SelfAttention

# KL-Divergence Loss
def kl_divergence(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    return 10 * K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=-1), axis=-1)


# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=2), axis=2)
    sum_x = K.sum(K.sum(y_true, axis=2), axis=2)
    sum_y = K.sum(K.sum(y_pred, axis=2), axis=2)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=2), axis=2)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=2), axis=2)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return -2 * num / den


# Normalized Scanpath Saliency Loss
def nss(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred
    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)),
                                                               shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)),
                                                              shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -(K.sum(K.sum(y_true * y_pred, axis=2), axis=2) / K.sum(K.sum(y_true, axis=2), axis=2))


def upsampling(input, w = -1, h = -1, factor=upsampling_factor):
    if w != -1 and h != -1:
        return Lambda(lambda x: tf.transpose(
            tf.image.resize_bilinear(tf.transpose(x, [0, 2, 3, 1]),
                                     (w, h),
                                     align_corners=True), [0, 3, 1, 2]))(input)
    else:
        return Lambda(lambda x: tf.transpose(
            tf.image.resize_bilinear(tf.transpose(x, [0, 2, 3, 1]),
                                     (x.get_shape()[2] * factor, x.get_shape()[3] * factor),
                                     align_corners=True), [0, 3, 1, 2]))(input)

def upsampling_input(input, factor=upsampling_factor):
    return Lambda(lambda x: tf.transpose(
        tf.image.resize_bilinear(tf.transpose(x, [0, 2, 3, 1]), (x.get_shape()[2] // 2.0, x.get_shape()[3] // 2.0),
                                 align_corners=True), [0, 3, 1, 2]))(input)


def upsampling_shape(s):
    return s[:2] + (s[2] * upsampling_factor, s[3] * upsampling_factor)

def msdensenet(x):
    dcndown = keras.applications.DenseNet201(input_tensor=x[1], include_top=False)
    dcn = keras.applications.DenseNet201(input_tensor=x[0], include_top=False)
    for layer in dcn.layers:
        layer.name = layer.name + str("_2")
    updcndown = upsampling(dcndown.output, factor=2.0)

    x = keras.layers.concatenate([updcndown, dcn.output], axis=1)
    x = Conv2D(32, (5, 5), padding='same', activation='relu', dilation_rate=(4, 4))(x)
    x = Conv2D(64, (7, 7), padding='same', activation='relu', dilation_rate=(8, 8))(x)
    x = Conv2D(64, (9, 9), padding='same', activation='relu', dilation_rate=(16, 16))(x)
    x = upsampling(x)
    x = Conv2D(1, (1, 1), padding='same', activation='relu')(x)
    outs_up = LeakyReLU(alpha=0.1)(x)

    return [outs_up, outs_up, outs_up]

def tsdensenet(x):
    dcndown = keras.applications.DenseNet201(input_tensor=x[1], include_top=False)
    dcndown_1 = keras.applications.DenseNet201(input_tensor=x[2], include_top=False)
    dcn = keras.applications.DenseNet201(input_tensor=x[0], include_top=False)
    for layer in dcn.layers:
        layer.name = layer.name + str("_2")
    for layer in dcndown.layers:
        layer.name = layer.name + str("_3")
    updcndown = upsampling(dcndown.output, factor=2.0)
    updcndown_1 = upsampling(dcndown_1.output, w=30, h=40)

    # x = add([updcndown, dcn.output])
    x = keras.layers.concatenate([updcndown, dcn.output, updcndown_1], axis=1)
    x = Conv2D(32, (5, 5), padding='same', activation='relu', dilation_rate=(4, 4))(x)
    x = Conv2D(64, (7, 7), padding='same', activation='relu', dilation_rate=(8, 8))(x)
    x = Conv2D(64, (9, 9), padding='same', activation='relu', dilation_rate=(16, 16))(x)
    x = upsampling(x)
    x = Conv2D(1, (1, 1), padding='same', activation='relu')(x)
    outs_up = LeakyReLU(alpha=0.1)(x)

    return [outs_up, outs_up, outs_up]


def msdensenet_att(x):
    dcndown = keras.applications.DenseNet201(input_tensor=x[1], include_top=False)
    dcn = keras.applications.DenseNet201(input_tensor=x[0], include_top=False)
    for layer in dcn.layers:
        layer.name = layer.name + str("_2")

    updcndown = upsampling(dcndown.output, factor=2.0)
    x = keras.layers.concatenate([updcndown, dcn.output], axis=1)
    x = SelfAttention(3840)(x)
    x = Conv2D(32, (5, 5), padding='same', activation='relu', dilation_rate=(4, 4))(x)
    x = Conv2D(64, (7, 7), padding='same', activation='relu', dilation_rate=(8, 8))(x)
    x = Conv2D(64, (9, 9), padding='same', activation='relu', dilation_rate=(16, 16))(x)
    # x = SelfAttention(64)(x)
    x = upsampling(x)
    x = Conv2D(1, (1, 1), padding='same', activation='relu')(x)
    outs_up = LeakyReLU(alpha=0.1)(x)

    return [outs_up, outs_up, outs_up]


def msdensenet_non(x):
    dcndown = keras.applications.DenseNet201(input_tensor=x[1], include_top=False)
    dcn = keras.applications.DenseNet201(input_tensor=x[0], include_top=False)
    for layer in dcn.layers:
        layer.name = layer.name + str("_2")
    updcndown = upsampling(dcndown.output, factor=2.0)

    # x = add([updcndown, dcn.output])
    x = keras.layers.concatenate([updcndown, dcn.output], axis=1)
    x = Conv2D(1, (1, 1), padding='same', activation='relu')(x)
    x = upsampling(x)
    outs_up = LeakyReLU(alpha=0.1)(x)

    return [outs_up, outs_up, outs_up]


def sdensenet(x):
    dcn = keras.applications.DenseNet201(input_tensor=x[0], include_top=False)
    x = Conv2D(32, (5, 5), padding='same', activation='relu', dilation_rate=(4, 4))(dcn.output)
    x = Conv2D(64, (7, 7), padding='same', activation='relu', dilation_rate=(8, 8))(x)
    x = Conv2D(64, (9, 9), padding='same', activation='relu', dilation_rate=(16, 16))(x)
    x = upsampling(x)
    x = Conv2D(1, (1, 1), padding='same', activation='relu')(x)
    outs_up = LeakyReLU(alpha=0.1)(x)

    return [outs_up, outs_up, outs_up]

def dense(x):
    dcn = keras.applications.DenseNet201(input_tensor=x[0], include_top=False)
    x = upsampling(dcn.output)
    x = Conv2D(1, (1, 1), padding='same', activation='relu')(x)
    outs_up = LeakyReLU(alpha=0.1)(x)

    return [outs_up, outs_up, outs_up]

