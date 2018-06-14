import tensorflow as tf
import numpy as np
from layer import *
def FC256_4(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense('FC1', x, output_dim=256)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = relu(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense('FC2', x, output_dim=256)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = relu(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense('FC3', x, output_dim=256)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = relu(x)
    # fc4
    with tf.variable_scope('FC4_layer'):
        x = dense('FC4', x, output_dim=256)
        tf.summary.histogram('FC4_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc4')
        tf.summary.histogram('FC4_out', x)
        x = relu(x)

    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def FC512_4(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense('FC1', x, output_dim=512)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = relu(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense('FC2', x, output_dim=512)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = relu(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense('FC3', x, output_dim=512)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = relu(x)
    # fc4
    with tf.variable_scope('FC4_layer'):
        x = dense('FC4', x, output_dim=512)
        tf.summary.histogram('FC4_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc4')
        tf.summary.histogram('FC4_out', x)
        x = relu(x)

    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def FC512_4_bnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense_bwn(name='FC1', x=x, output_dim=512)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = binarize_v2(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense_bwn(name='FC2', x=x, output_dim=512)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = binarize_v2(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense_bwn(name='FC3', x=x, output_dim=512)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = binarize_v2(x)
    with tf.variable_scope('FC4_layer'):
        x = dense_bwn(name='FC4', x=x, output_dim=512)
        tf.summary.histogram('FC4_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc4')
        tf.summary.histogram('FC4_out', x)
        x = binarize_v2(x)
    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense_bwn('FCout', x, output_dim=2)
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def FC512_4_tnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense_twn(name='FC1', x=x, output_dim=512)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = ternary_v3(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense_twn(name='FC2', x=x, output_dim=512)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = ternary_v3(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense_twn(name='FC3', x=x, output_dim=512)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = ternary_v3(x)
    with tf.variable_scope('FC4_layer'):
        x = dense_twn(name='FC4', x=x, output_dim=512)
        tf.summary.histogram('FC4_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc4')
        tf.summary.histogram('FC4_out', x)
        x = ternary_v3(x)
    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense_twn('FCout', x, output_dim=2)
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def FC512_4_bnntnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense_bwn(name='FC1', x=x, output_dim=512)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = ternary_v3(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense_bwn(name='FC2', x=x, output_dim=512)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = ternary_v3(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense_bwn(name='FC3', x=x, output_dim=512)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = ternary_v3(x)
    with tf.variable_scope('FC4_layer'):
        x = dense_bwn(name='FC4', x=x, output_dim=512)
        tf.summary.histogram('FC4_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc4')
        tf.summary.histogram('FC4_out', x)
        x = ternary_v3(x)
    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense_bwn('FCout', x, output_dim=2)
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def FC128_4(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense('FC1', x, output_dim=128)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = relu(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense('FC2', x, output_dim=128)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = relu(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense('FC3', x, output_dim=128)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = relu(x)
    # fc4
    with tf.variable_scope('FC4_layer'):
        x = dense('FC4', x, output_dim=128)
        tf.summary.histogram('FC4_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc4')
        tf.summary.histogram('FC4_out', x)
        x = relu(x)

    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def FC128_twn_3bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    bit=3
    with tf.variable_scope('FC1_layer'):
        x = dense_twn(name='FC1', x=x, output_dim=128)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = quantize(x,bit)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense_twn(name='FC2', x=x, output_dim=128)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = quantize(x,bit)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense_twn(name='FC3', x=x, output_dim=128)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = quantize(x,bit)

    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense_twn('FCout', x, output_dim=2)
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def FC128_tnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense_twn(name='FC1', x=x, output_dim=128)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = ternary(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense_twn(name='FC2', x=x, output_dim=128)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = ternary(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense_twn(name='FC3', x=x, output_dim=128)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = ternary(x)

    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense_twn('FCout', x, output_dim=2)
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def FC128_bnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense_bwn(name='FC1', x=x, output_dim=128)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = binarize_v2(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense_bwn(name='FC2', x=x, output_dim=128)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = binarize_v2(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense_bwn(name='FC3', x=x, output_dim=128)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = binarize_v2(x)

    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense_bwn('FCout', x, output_dim=2)
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def FC128_3(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense('FC1', x, output_dim=128)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = relu(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense('FC2', x, output_dim=128)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = relu(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense('FC3', x, output_dim=128)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = relu(x)

    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def FC128_10(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense('FC1', x, output_dim=128)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = relu(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense('FC2', x, output_dim=128)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = relu(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense('FC3', x, output_dim=128)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = relu(x)

    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense('FCout', x, output_dim=11)
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def FC256_10(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    x = tf.reshape(fingerprint_input,
                                [-1, input_time_size*input_frequency_size])
    # fc1
    with tf.variable_scope('FC1_layer'):
        x = dense('FC1', x, output_dim=256)
        tf.summary.histogram('FC1_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
        tf.summary.histogram('FC1_out', x)
        x = relu(x)
    # fc2
    with tf.variable_scope('FC2_layer'):
        x = dense('FC2', x, output_dim=256)
        tf.summary.histogram('FC2_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc2')
        tf.summary.histogram('FC2_out', x)
        x = relu(x)

    # fc3
    with tf.variable_scope('FC3_layer'):
        x = dense('FC3', x, output_dim=256)
        tf.summary.histogram('FC3_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfc3')
        tf.summary.histogram('FC3_out', x)
        x = relu(x)

    # fcout
    with tf.variable_scope('FCout_layer'):
        x = dense('FCout', x, output_dim=11)
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def hellow_edge_dscnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1]) # N H W C mode

    x = conv2d('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='SAME', stride=(2, 2),
               activation=tf.nn.relu, batchnorm_enabled=True,is_training=is_training)
    x = depthwise_conv2d('DSC1', x,  activation=tf.nn.relu,batchnorm_enabled=True, is_training=is_training)
    x = conv2d('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),
               activation=tf.nn.relu, batchnorm_enabled=True,is_training=is_training)

    x = depthwise_conv2d('DSC2', x,  activation=tf.nn.relu,batchnorm_enabled=True, is_training=is_training)
    x = conv2d('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),
               activation=tf.nn.relu, batchnorm_enabled=True,is_training=is_training)

    x = depthwise_conv2d('DSC3', x,  activation=tf.nn.relu,batchnorm_enabled=True, is_training=is_training)
    x = conv2d('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),
               activation=tf.nn.relu, batchnorm_enabled=True,is_training=is_training)

    x = depthwise_conv2d('DSC4', x,  activation=tf.nn.relu,batchnorm_enabled=True, is_training=is_training)
    x = conv2d('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),
               activation=tf.nn.relu, batchnorm_enabled=True,is_training=is_training)

    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x= flatten(x)
    x = dense('FC', x, output_dim=11,batchnorm_enabled=True,is_training=is_training)

    return x

def hellow_edge_dscnn_bwn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1]) # N H W C mode

    x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='SAME', stride=(2, 2),
               activation=hard_tanh, batchnorm_enabled=True,is_training=is_training,epsilon = 5e-1)
    tf.summary.histogram('conv1', x)

    x = depthwise_conv2d_bwn('DSC1', x,  activation=hard_tanh,batchnorm_enabled=True, is_training=is_training,epsilon = 5e-1)
    tf.summary.histogram('DSC1', x)

    x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),
               activation=hard_tanh, batchnorm_enabled=True,is_training=is_training,epsilon = 5e-1)
    tf.summary.histogram('DSC1_pointwise', x)

    x = depthwise_conv2d_bwn('DSC2', x,  activation=hard_tanh,batchnorm_enabled=True, is_training=is_training,epsilon = 5e-1)
    tf.summary.histogram('DSC2', x)

    x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),
               activation=hard_tanh, batchnorm_enabled=True,is_training=is_training,epsilon = 5e-1)
    tf.summary.histogram('DSC2_pointwise', x)

    x = depthwise_conv2d_bwn('DSC3', x,  activation=hard_tanh,batchnorm_enabled=True, is_training=is_training,epsilon = 5e-1)
    tf.summary.histogram('DSC3', x)

    x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),
               activation=hard_tanh, batchnorm_enabled=True,is_training=is_training,epsilon = 5e-1)
    tf.summary.histogram('DSC3_pointwise', x)

    x = depthwise_conv2d_bwn('DSC4', x,  activation=hard_tanh,batchnorm_enabled=True, is_training=is_training,epsilon = 5e-1)
    tf.summary.histogram('DSC4', x)

    x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),
               activation=hard_tanh, batchnorm_enabled=True,is_training=is_training,epsilon = 5e-1)
    tf.summary.histogram('DSC4_pointwise', x)

    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x= flatten(x)
    x = dense_bwn('FC', x, output_dim=11,batchnorm_enabled=True,is_training=is_training,epsilon = 5e-1)
    tf.summary.histogram('FC', x)

    return x
def variable_summaries_scale(bn_name,num = 64):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    with tf.variable_scope(bn_name,reuse=True):
        mvar1 = tf.get_variable('moving_variance',[num])
        mm1 = tf.get_variable('moving_mean',[num])
    with tf.name_scope('summaries'):
        #mean = tf.reduce_mean(var)
        tf.summary.histogram(bn_name + 'moving_variance', mvar1)
        tf.summary.histogram(bn_name + 'moving_mean', mm1)

def hellow_edge_dscnn_test(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1]) # N H W C mode #8 replace =binarize
    x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
    tf.summary.histogram('CONV1', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BN1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization')
    #dsc1
    x = depthwise_conv2d_bwn('DSC1', x)
    tf.summary.histogram('DSC1', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_1')

    x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC1_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNpw1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_2')

    #dsc2
    x = depthwise_conv2d_bwn('DSC2', x)
    tf.summary.histogram('DSC2', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc2', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_3')

    x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC2_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps2', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_4')

    #dsc3
    x = depthwise_conv2d_bwn('DSC3', x)
    tf.summary.histogram('DSC3', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc3', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_5')

    x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC3_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps3', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_6')


    #dsc4
    x = depthwise_conv2d_bwn('DSC4', x, padding='VALID')
    tf.summary.histogram('DSC4', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc4', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_7')

    x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC4_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps4', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_8')

    #fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x= flatten(x)
    x = dense_bwn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    x = batchnormalization(x,is_training = is_training)
    variable_summaries_scale('batch_normalization_9',11)

    tf.summary.histogram('BNfc', x)
    return x
def fixed_point_bwn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 100])  # N H W C mode #8 replace =binarize
    x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=1024, kernel_size=(10, 4), padding='VALID', stride=(2, 2),bias=-10.0)
    tf.summary.histogram('CONV1', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BN1', x)
    x = binarize(x)
    #variable_summaries_scale('batch_normalization',1024)
    # dsc1
    x = depthwise_conv2d_bwn('DSC1', x)
    tf.summary.histogram('DSC1', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc1', x)
    x = binarize(x)
    #variable_summaries_scale('batch_normalization_1',512)

    x = conv2d_bwn('DSC1_pointwise', x, num_filters=128, kernel_size=(1, 1), padding='SAME', stride=(1, 1),bias=0.0)
    tf.summary.histogram('DSC1_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNpw1', x)
    x = binarize(x)
    # variable_summaries_scale('batch_normalization_2',128)

    # dsc2
    x = depthwise_conv2d_bwn('DSC2', x)
    tf.summary.histogram('DSC2', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc2', x)
    x = binarize(x)
   # variable_summaries_scale('batch_normalization_3')

    x = conv2d_bwn('DSC2_pointwise', x, num_filters=128, kernel_size=(1, 1), padding='SAME', stride=(1, 1),bias=0.0)
    tf.summary.histogram('DSC2_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps2', x)
    x = binarize(x)
    #variable_summaries_scale('batch_normalization_4')

    # dsc3
    x = depthwise_conv2d_bwn('DSC3', x)
    tf.summary.histogram('DSC3', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc3', x)
    x = binarize(x)
   # variable_summaries_scale('batch_normalization_5')

    x = conv2d_bwn('DSC3_pointwise', x, num_filters=128, kernel_size=(1, 1), padding='SAME', stride=(1, 1),bias=0.0)
    tf.summary.histogram('DSC3_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps3', x)
    x = binarize(x)
    #variable_summaries_scale('batch_normalization_6')

    # dsc4
    x = depthwise_conv2d_bwn('DSC4', x, padding='VALID')
    tf.summary.histogram('DSC4', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc4', x)
    x = binarize(x)
    x = conv2d_bwn('DSC4_pointwise', x, num_filters=128, kernel_size=(1, 1), padding='SAME', stride=(1, 1),bias=0.0)
    tf.summary.histogram('DSC4_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps4', x)
    x = binarize(x)
    # fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x = flatten(x)
    x = dense_bwn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    x = batchnormalization(x, is_training=is_training)
    #variable_summaries_scale('batch_normalization_9', 11)

    tf.summary.histogram('BNfc', x)
    return x
def fixed_point_bwn_test(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    x = conv2d_twbwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
    tf.summary.histogram('CONV1', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BN1', x)
    x = quantize(x, 8)
    tf.summary.histogram('BN1_quantize', x)

    variable_summaries_scale('batch_normalization', 128)
    # dsc1
    x = depthwise_conv2d_twbwn('DSC1', x)
    tf.summary.histogram('DSC1', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc1', x)
    x = quantize(x, 8)
    tf.summary.histogram('BNdsc1_quantize', x)

    variable_summaries_scale('batch_normalization_1', 128)

    x = conv2d_twbwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC1_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNpw1', x)
    x = quantize(x, 8)
    tf.summary.histogram('BNpw1_quantize', x)

    # variable_summaries_scale('batch_normalization_2',128)

    # dsc2
    x = depthwise_conv2d_twbwn('DSC2', x)
    tf.summary.histogram('DSC2', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc2', x)
    x = quantize(x, 8)
    # variable_summaries_scale('batch_normalization_3')

    x = conv2d_twbwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC2_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps2', x)
    x = quantize(x, 8)
    # variable_summaries_scale('batch_normalization_4')

    # dsc3
    x = depthwise_conv2d_twbwn('DSC3', x)
    tf.summary.histogram('DSC3', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc3', x)
    x = quantize(x, 8)
    # variable_summaries_scale('batch_normalization_5')

    x = conv2d_twbwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC3_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps3', x)
    x = quantize(x, 8)
    # variable_summaries_scale('batch_normalization_6')

    # dsc4
    x = depthwise_conv2d_twbwn('DSC4', x, padding='VALID')
    tf.summary.histogram('DSC4', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc4', x)
    x = quantize(x, 8)
    # variable_summaries_scale('batch_normalization_7')

    x = conv2d_twbwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC4_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps4', x)
    x = quantize(x, 8)
    # variable_summaries_scale('batch_normalization_8')

    # fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x = flatten(x)
    x = dense_twbwn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    x = batchnormalization(x, is_training=is_training)
    # variable_summaries_scale('batch_normalization_9', 11)

    tf.summary.histogram('BNfc', x)
    return x
def fixed_point_bnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        conv1 = x
        x = batchnormalization_hardware(x, is_training=is_training, name='BN1')
        conv1bn = x
        x = binarize_v2(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        dsc1 = x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNdsc1')
        dsc1bn = x
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        dsc1pw = x
        x = batchnormalization_hardware(x, is_training=is_training, name='BNpw1')
        dsc1pwbn = x
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_bwn('DSC2', x)
        dsc2 = x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNdsc2')
        dsc2bn = x
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc2pw = x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNps2')
        dsc2pwbn = x
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_bwn('DSC3', x)
        dsc3 = x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNdsc3')
        dsc3bn = x
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc3pw = x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNps3')
        dsc3pwbn = x
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d_bwn('DSC4', x)
        dsc4 = x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNdsc4')
        dsc4bn = x
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc4pw = x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNps4')
        dsc4pwbn = x
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        fc =x
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')
        tf.summary.histogram('FCout_out', x)

    return x,conv1,conv1bn,dsc1,dsc1bn,dsc1pw,dsc1pwbn,dsc2,dsc2bn,dsc2pw,dsc2pwbn,dsc3,dsc3bn,dsc3pw,dsc3pwbn,\
           dsc4,dsc4bn,dsc4pw,dsc4pwbn,fc
def dscv1_q16(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    weishu = 16
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_quantize('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2),wei=weishu)
        conv1 =x
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        conv1bn=x
        x = quantize(x,weishu)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_quantize('DSC1', x,wei=weishu)
        dsc1 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        dsc1bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc1pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        dsc1pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_quantize('DSC2', x,wei=weishu)
        dsc2 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        dsc2bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc2pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        dsc2pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_quantize('DSC3', x,wei=weishu)
        dsc3 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        dsc3bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc3pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        dsc3pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d_quantize('DSC4', x,wei=weishu)
        dsc4 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        dsc4bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc4pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        dsc4pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense_quantize('FCout', x, output_dim=model_settings['label_count'],wei=weishu)
        fc =x
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')
        tf.summary.histogram('FCout_out', x)

    return x
def dscv1_q8(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    weishu = 8
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_quantize('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2),wei=weishu)
        conv1 =x
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        conv1bn=x
        x = quantize(x,weishu)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_quantize('DSC1', x,wei=weishu)
        dsc1 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        dsc1bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc1pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        dsc1pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_quantize('DSC2', x,wei=weishu)
        dsc2 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        dsc2bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc2pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        dsc2pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_quantize('DSC3', x,wei=weishu)
        dsc3 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        dsc3bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc3pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        dsc3pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d_quantize('DSC4', x,wei=weishu)
        dsc4 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        dsc4bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc4pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        dsc4pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense_quantize('FCout', x, output_dim=model_settings['label_count'],wei=weishu)
        fc =x
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')
        tf.summary.histogram('FCout_out', x)

    return x
def dscv1_q4(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    weishu = 4
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_quantize('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2),wei=weishu)
        conv1 =x
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        conv1bn=x
        x = quantize(x,weishu)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_quantize('DSC1', x,wei=weishu)
        dsc1 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        dsc1bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc1pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        dsc1pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_quantize('DSC2', x,wei=weishu)
        dsc2 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        dsc2bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc2pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        dsc2pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_quantize('DSC3', x,wei=weishu)
        dsc3 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        dsc3bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc3pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        dsc3pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d_quantize('DSC4', x,wei=weishu)
        dsc4 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        dsc4bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc4pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        dsc4pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense_quantize('FCout', x, output_dim=model_settings['label_count'],wei=weishu)
        fc =x
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')
        tf.summary.histogram('FCout_out', x)

    return x
def dscv1_q3(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    weishu = 3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_quantize('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2),wei=weishu)
        conv1 =x
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        conv1bn=x
        x = quantize(x,weishu)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_quantize('DSC1', x,wei=weishu)
        dsc1 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        dsc1bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc1pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        dsc1pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_quantize('DSC2', x,wei=weishu)
        dsc2 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        dsc2bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc2pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        dsc2pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_quantize('DSC3', x,wei=weishu)
        dsc3 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        dsc3bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc3pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        dsc3pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d_quantize('DSC4', x,wei=weishu)
        dsc4 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        dsc4bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc4pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        dsc4pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense_quantize('FCout', x, output_dim=model_settings['label_count'],wei=weishu)
        fc =x
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')
        tf.summary.histogram('FCout_out', x)

    return x
def dscv1_q2(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    weishu = 2
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_quantize('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2),wei=weishu)
        conv1 =x
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        conv1bn=x
        x = quantize(x,weishu)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_quantize('DSC1', x,wei=weishu)
        dsc1 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        dsc1bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc1pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        dsc1pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_quantize('DSC2', x,wei=weishu)
        dsc2 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        dsc2bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc2pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        dsc2pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_quantize('DSC3', x,wei=weishu)
        dsc3 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        dsc3bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc3pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        dsc3pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d_quantize('DSC4', x,wei=weishu)
        dsc4 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        dsc4bn =x
        x = quantize(x,weishu)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_quantize('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),wei=weishu)
        dsc4pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        dsc4pwbn =x
        x = quantize(x,weishu)
        tf.summary.histogram('pointwise_out', x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense_quantize('FCout', x, output_dim=model_settings['label_count'],wei=weishu)
        fc =x
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')
        tf.summary.histogram('FCout_out', x)

    return x
def dscnn_twn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        conv1 =x
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        conv1bn=x
        x = relu(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d('DSC1', x)
        dsc1 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        dsc1bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc1pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        dsc1pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d('DSC2', x)
        dsc2 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        dsc2bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc2pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        dsc2pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d('DSC3', x)
        dsc3 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        dsc3bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc3pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        dsc3pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d('DSC4', x)
        dsc4 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        dsc4bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc4pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        dsc4pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense('FCout', x, output_dim=model_settings['label_count'])
        fc =x
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
    #return x,conv1,conv1bn,dsc1,dsc1bn,dsc1pw,dsc1pwbn,dsc2,dsc2bn,dsc2pw,dsc2pwbn,dsc3,dsc3bn,dsc3pw,dsc3pwbn,\
    #       dsc4,dsc4bn,dsc4pw,dsc4pwbn,fc
def dscv1(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        conv1 =x
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        conv1bn=x
        x = relu(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d('DSC1', x)
        dsc1 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        dsc1bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc1pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        dsc1pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d('DSC2', x)
        dsc2 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        dsc2bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc2pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        dsc2pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d('DSC3', x)
        dsc3 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        dsc3bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc3pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        dsc3pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d('DSC4', x)
        dsc4 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        dsc4bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc4pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        dsc4pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense('FCout', x, output_dim=model_settings['label_count'])
        fc =x
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
    #return x,conv1,conv1bn,dsc1,dsc1bn,dsc1pw,dsc1pwbn,dsc2,dsc2bn,dsc2pw,dsc2pwbn,dsc3,dsc3bn,dsc3pw,dsc3pwbn,\
    #       dsc4,dsc4bn,dsc4pw,dsc4pwbn,fc
def dscv2(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d('CONV1', fingerprint_4d, num_filters=172, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        conv1 =x
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        conv1bn=x
        x = relu(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d('DSC1', x)
        dsc1 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        dsc1bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC1_pointwise', x, num_filters=172, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc1pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        dsc1pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d('DSC2', x)
        dsc2 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        dsc2bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC2_pointwise', x, num_filters=172, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc2pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        dsc2pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d('DSC3', x)
        dsc3 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        dsc3bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC3_pointwise', x, num_filters=172, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc3pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        dsc3pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d('DSC4', x)
        dsc4 =x
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        dsc4bn =x
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC4_pointwise', x, num_filters=172, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        dsc4pw =x
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        dsc4pwbn =x
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense('FCout', x, output_dim=model_settings['label_count'])
        fc =x
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')
        tf.summary.histogram('FCout_out', x)

    return x,conv1,conv1bn,dsc1,dsc1bn,dsc1pw,dsc1pwbn,dsc2,dsc2bn,dsc2pw,dsc2pwbn,dsc3,dsc3bn,dsc3pw,dsc3pwbn,\
           dsc4,dsc4bn,dsc4pw,dsc4pwbn,fc
def cnn_twn_3bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    wei =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=24, kernel_size=(10, 4), padding='VALID', stride=(2, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,wei)
    with tf.variable_scope('CONV2_layer'):
        x = conv2d_twn('CONV2', x, num_filters=16, kernel_size=(10, 4), padding='VALID', stride=(1, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN2')
        x = quantize(x,wei)

    with tf.variable_scope('FCout_layer'):
        x = flatten(x)
        x = dense_twn('lin', x, output_dim=16)
        x = dense_twn('dnn', x, output_dim=128)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN3')
        x = quantize(x,wei)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'])
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')

    return x
def cnn_tnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=24, kernel_size=(10, 4), padding='VALID', stride=(2, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = ternary_v3(x)
    with tf.variable_scope('CONV2_layer'):
        x = conv2d_twn('CONV2', x, num_filters=16, kernel_size=(10, 4), padding='VALID', stride=(1, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN2')
        x = ternary_v3(x)

    with tf.variable_scope('FCout_layer'):
        x = flatten(x)
        x = dense_twn('lin', x, output_dim=16)
        x = dense_twn('dnn', x, output_dim=128)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN3')
        x = ternary_v3(x)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'])
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')

    return x
def cnn_bnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=24, kernel_size=(10, 4), padding='VALID', stride=(2, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = binarize_v2(x)
    with tf.variable_scope('CONV2_layer'):
        x = conv2d_bwn('CONV2', x, num_filters=16, kernel_size=(10, 4), padding='VALID', stride=(1, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN2')
        x = binarize_v2(x)

    with tf.variable_scope('FCout_layer'):
        x = flatten(x)
        x = dense_bwn('lin', x, output_dim=16)
        x = dense_bwn('dnn', x, output_dim=128)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN3')
        x = binarize_v2(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')

    return x
def cnn_bnntnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=24, kernel_size=(10, 4), padding='VALID', stride=(2, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = ternary_v3(x)
    with tf.variable_scope('CONV2_layer'):
        x = conv2d_bwn('CONV2', x, num_filters=16, kernel_size=(10, 4), padding='VALID', stride=(1, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN2')
        x = ternary_v3(x)

    with tf.variable_scope('FCout_layer'):
        x = flatten(x)
        x = dense_bwn('lin', x, output_dim=16)
        x = dense_bwn('dnn', x, output_dim=128)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN3')
        x = ternary_v3(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')

    return x

def cnn_trad_fpool3(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(1, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = relu(x)
    with tf.variable_scope('CONV2_layer'):
        x = conv2d('CONV2', x, num_filters=48, kernel_size=(10, 4), padding='VALID', stride=(2, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN2')
        x = relu(x)

    with tf.variable_scope('FCout_layer'):
        x = flatten(x)
        x = dense('lin', x, output_dim=16)
        x = dense('dnn', x, output_dim=128)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN3')
        x = relu(x)
        x = dense('FCout', x, output_dim=model_settings['label_count'])
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')

    return x
def cnnv1(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d('CONV1', fingerprint_4d, num_filters=24, kernel_size=(10, 4), padding='VALID', stride=(2, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = relu(x)
    with tf.variable_scope('CONV2_layer'):
        x = conv2d('CONV2', x, num_filters=16, kernel_size=(10, 4), padding='VALID', stride=(1, 1))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN2')
        x = relu(x)

    with tf.variable_scope('FCout_layer'):
        x = flatten(x)
        x = dense('lin', x, output_dim=16)
        x = dense('dnn', x, output_dim=128)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN3')
        x = relu(x)
        x = dense('FCout', x, output_dim=model_settings['label_count'])
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')

    return x

def fixed_point_bnn_v2(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = binarize_v2(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNfcout')
        tf.summary.histogram('FCout_out', x)

    return x

def fixed_point_twn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = relu(x)
        tf.summary.histogram('BN1', x)
    with tf.variable_scope('CONV2_layer'):
        x = conv2d('CONV2', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training, name='BN2')
        x = relu(x)
        tf.summary.histogram('BN2', x)
    # dsc1
    # with tf.variable_scope('DSC1_layer'):
    #     x = depthwise_conv2d('DSC1', x)
    #     tf.summary.histogram('depthwise', x)
    #     x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
    #     x = relu(x)
    #     tf.summary.histogram('depthwise_out', x)
    #
    #     x = conv2d('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    #     tf.summary.histogram('pointwise', x)
    #     x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
    #     x = relu(x)
    #     tf.summary.histogram('pointwise_out', x)
    #
    # # dsc2
    # with tf.variable_scope('DSC2_layer'):
    #     x = depthwise_conv2d('DSC2', x)
    #     tf.summary.histogram('depthwise', x)
    #     x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
    #     x = relu(x)
    #     tf.summary.histogram('depthwise_out', x)
    #
    #     x = conv2d('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    #     tf.summary.histogram('pointwise', x)
    #     x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
    #     x = relu(x)
    #     tf.summary.histogram('pointwise_out', x)

    # #dsc3
    # with tf.variable_scope('DSC3_layer'):
    #     x = depthwise_conv2d('DSC3', x)
    #     tf.summary.histogram('depthwise', x)
    #     x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
    #     x = relu(x)
    #     tf.summary.histogram('depthwise_out', x)
    #
    #     x = conv2d('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    #     tf.summary.histogram('pointwise', x)
    #     x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
    #     x = relu(x)
    #     tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_twn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = relu(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = relu(x)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def twn_2bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =2
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,bits)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_twn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_twn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_twn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d_twn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def twn_3bit_theta(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    the = 0.5
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2),the=the)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,bits)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_twn(name='DSC1', x=x,the=the)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),the=the)
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_twn('DSC2', x,the=the)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),the=the)
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_twn('DSC3', x,the=the)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),the=the)
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x= depthwise_conv2d_twn('DSC4', x, padding='VALID',the=the)
        #x = depthwise_conv2d_twn('DSC4', x,the=the)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1),the=the)
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'],the=the)
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def dscnn_tnn2(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =2
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = ternary_v3(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_twn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = ternary_v3(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = ternary_v3(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_twn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = ternary_v3(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = ternary_v3(x)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_twn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = ternary_v3(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = ternary_v3(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_twn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = ternary_v3(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = ternary_v3(x)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def dscnn_bnntnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =2
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = ternary_v3(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = ternary_v3(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = ternary_v3(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_bwn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = ternary_v3(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = ternary_v3(x)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_bwn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = ternary_v3(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = ternary_v3(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_bwn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_twn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = ternary_v3(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = ternary_v3(x)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def dscnn_tnn(fingerprint_input, model_settings, is_training,batch):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = ternary_v2(x,batch)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_twn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = ternary_v2(x,batch)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = ternary_v2(x,batch)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_twn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = ternary_v2(x,batch)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = ternary_v2(x,batch)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_twn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = ternary_v2(x,batch)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = ternary_v2(x,batch)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_twn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = ternary_v2(x,batch)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = ternary_v2(x,batch)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def dscnn_bnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = binarize_v2(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_bwn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_bwn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def dscnn_bnn_h5(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 4))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = binarize_v2(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_bwn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_bwn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def dscnn_bnn_3dsc(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = binarize_v2(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_bwn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)
    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def dscnn_bnn_2dsc(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = binarize_v2(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def dscnn_bnn_1dsc(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = binarize_v2(x)
        tf.summary.histogram('BN1', x)
    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)
    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def dscnn_bnn_2dscv2(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=32, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = binarize_v2(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=32, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=32, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def dscnn_bnn_1dscv2(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=48, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = binarize_v2(x)
        tf.summary.histogram('BN1', x)
    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = binarize_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=48, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = binarize_v2(x)
        tf.summary.histogram('pointwise_out', x)
    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x


def dscnn_bnn_3bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,bits)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_bwn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_bwn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def dscnn_bnn_4bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =4
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,bits)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_bwn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_bwn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def dscnn_bnn_6bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =6
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,bits)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_bwn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_bwn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def dscnn_bnn_8bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =8
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,bits)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_bwn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_bwn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_bwn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_bwn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_bwn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def twn_3bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    bits =3
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,bits)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_twn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_twn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_twn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_twn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = quantize(x,bits)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = quantize(x,bits)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def twn_4bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,4)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_twn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = quantize(x,4)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = quantize(x,4)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_twn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = quantize(x,4)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = quantize(x,4)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_twn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = quantize(x,4)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = quantize(x,4)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d_twn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = quantize(x,4)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = quantize(x,4)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def twn_6bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,6)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_twn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = quantize(x,6)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = quantize(x,6)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_twn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = quantize(x,6)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = quantize(x,6)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_twn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = quantize(x,6)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = quantize(x,6)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d_twn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = quantize(x,6)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = quantize(x,6)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x
def twn_8bit(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = quantize(x,8)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_twn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = quantize(x,8)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = quantize(x,8)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_twn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = quantize(x,8)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = quantize(x,8)
        tf.summary.histogram('pointwise_out', x)

    #dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_twn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = quantize(x,8)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = quantize(x,8)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        #x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        x = depthwise_conv2d_twn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = quantize(x,8)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = quantize(x,8)
        tf.summary.histogram('pointwise_out', x)

    # fc
    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        #x = max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling', padding='VALID')
        x = flatten(x)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)
    return x

def fixed_point_twn_bnn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    with tf.variable_scope('CONV1_layer'):
        x = conv2d_twn('CONV1', fingerprint_4d, num_filters=128, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BN1')
        x = ternary_v2(x)
        tf.summary.histogram('BN1', x)
    # dsc1
    with tf.variable_scope('DSC1_layer'):
        x = depthwise_conv2d_twn('DSC1', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc1')
        x = ternary_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNpw1')
        x = ternary_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc2
    with tf.variable_scope('DSC2_layer'):
        x = depthwise_conv2d_twn('DSC2', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc2')
        x = ternary_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps2')
        x = ternary_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc3
    with tf.variable_scope('DSC3_layer'):
        x = depthwise_conv2d_twn('DSC3', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc3')
        x = ternary_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps3')
        x = ternary_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # dsc4
    with tf.variable_scope('DSC4_layer'):
        x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
        #x = depthwise_conv2d_twn('DSC4', x)
        tf.summary.histogram('depthwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNdsc4')
        x = ternary_v2(x)
        tf.summary.histogram('depthwise_out', x)

        x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        tf.summary.histogram('pointwise', x)
        x = batchnormalization_hardware(x, is_training=is_training,name = 'BNps4')
        x = ternary_v2(x)
        tf.summary.histogram('pointwise_out', x)

    # fc
    # with tf.variable_scope('FC_layer'):
    #     x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    #     x = flatten(x)
    #     x = dense_twn('FC', x, output_dim=11)
    #     tf.summary.histogram('FC_before_bn', x)
    #     x = batchnormalization_hardware(x, is_training=is_training, name='BNfc')
    #     tf.summary.histogram('FC_out', x)

    #with tf.variable_scope('FC1_layer'):
    #    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    #    x = flatten(x)
    #    x = dense_twn('FC1', x, output_dim=256)
    #    tf.summary.histogram('FC1_before_bn', x)
    #    x = batchnormalization_hardware(x, is_training=is_training, name='BNfc1')
    #    x = binarize_v2(x)

    with tf.variable_scope('FCout_layer'):
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = dense_twn('FCout', x, output_dim=model_settings['label_count'])
        tf.summary.histogram('FCout_before_bn', x)
        x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
        tf.summary.histogram('FCout_out', x)

    return x


def fixed_point_twn_bntest(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
    tf.summary.histogram('CONV1', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BN1')
    x = binarize_v2(x)
    tf.summary.histogram('BN1', x)
    variable_summaries_scale('BN1')
    # dsc1
    x = depthwise_conv2d_twn('DSC1', x)
    tf.summary.histogram('DSC1', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNdsc1')
    x = binarize_v2(x)
    tf.summary.histogram('BNdsc1', x)
    variable_summaries_scale('BNdsc1')

    x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC1_pointwise', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNpw1')
    x = binarize_v2(x)
    tf.summary.histogram('BNpw1', x)
    variable_summaries_scale('BNpw1')

    # dsc2
    x = depthwise_conv2d_twn('DSC2', x)
    tf.summary.histogram('DSC2', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNdsc2')
    x = binarize_v2(x)
    tf.summary.histogram('BNdsc2', x)
    variable_summaries_scale('BNdsc2')

    x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC2_pointwise', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNps2')
    x = binarize_v2(x)
    tf.summary.histogram('BNps2', x)
    variable_summaries_scale('BNps2')

    # dsc3
    x = depthwise_conv2d_twn('DSC3', x)
    tf.summary.histogram('DSC3', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNdsc3')
    x = binarize_v2(x)
    tf.summary.histogram('BNdsc3', x)
    variable_summaries_scale('BNdsc3')

    x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC3_pointwise', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNps3')
    x = binarize_v2(x)
    tf.summary.histogram('BNps3', x)
    variable_summaries_scale('BNps3')

    # dsc4
    x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
    tf.summary.histogram('DSC4', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNdsc4')
    x = binarize_v2(x)
    tf.summary.histogram('BNdsc4', x)
    variable_summaries_scale('BNdsc4')

    x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC4_pointwise', x)
   # x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNps4')
    x = binarize_v2(x)
    tf.summary.histogram('BNps4', x)
    variable_summaries_scale('BNps4')

    # fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x = flatten(x)
    x = dense_twn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNfc')
    variable_summaries_scale('BNfc', 11)

    tf.summary.histogram('BNfc', x)
    return x

def fixed_point_fwn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    x = conv2d_fwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
    tf.summary.histogram('CONV1', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BN1')
    x = quantize(x,2)
    tf.summary.histogram('BN1', x)
    variable_summaries_scale('BN1')
    # dsc1
    x = depthwise_conv2d_fwn('DSC1', x)
    tf.summary.histogram('DSC1', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNdsc1')
    x = quantize(x,2)
    tf.summary.histogram('BNdsc1', x)
    variable_summaries_scale('BNdsc1')

    x = conv2d_fwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC1_pointwise', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNpw1')
    x = quantize(x,2)
    tf.summary.histogram('BNpw1', x)
    variable_summaries_scale('BNpw1')

    # dsc2
    x = depthwise_conv2d_fwn('DSC2', x)
    tf.summary.histogram('DSC2', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNdsc2')
    x = quantize(x,2)
    tf.summary.histogram('BNdsc2', x)
    variable_summaries_scale('BNdsc2')

    x = conv2d_fwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC2_pointwise', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNps2')
    x = quantize(x,2)
    tf.summary.histogram('BNps2', x)
    variable_summaries_scale('BNps2')

    # dsc3
    x = depthwise_conv2d_fwn('DSC3', x)
    tf.summary.histogram('DSC3', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNdsc3')
    x = quantize(x,2)
    tf.summary.histogram('BNdsc3', x)
    variable_summaries_scale('BNdsc3')

    x = conv2d_fwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC3_pointwise', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNps3')
    x = quantize(x,2)
    tf.summary.histogram('BNps3', x)
    variable_summaries_scale('BNps3')

    # dsc4
    x = depthwise_conv2d_fwn('DSC4', x, padding='VALID')
    tf.summary.histogram('DSC4', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNdsc4')
    x = quantize(x,2)
    tf.summary.histogram('BNdsc4', x)
    variable_summaries_scale('BNdsc4')

    x = conv2d_fwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC4_pointwise', x)
   # x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNps4')
    x = quantize(x,2)
    tf.summary.histogram('BNps4', x)
    variable_summaries_scale('BNps4')

    # fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x = flatten(x)
    x = dense_fwn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    #x = batchnormalization(x, is_training=is_training)
    x = batchnormalization_quantize(x, is_training=is_training,name = 'BNfc')
    variable_summaries_scale('BNfc', 11)
    tf.summary.histogram('BNfc', x)
    return x
















def fixed_point_xwn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
    tf.summary.histogram('CONV1', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BN1', x)
    x = quantize(x,8)
    variable_summaries_scale('batch_normalization')
    # dsc1
    x = depthwise_conv2d_twn('DSC1', x)
    tf.summary.histogram('DSC1', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc1', x)
    x = quantize(x,8)
    variable_summaries_scale('batch_normalization_1')

    x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC1_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNpw1', x)
    x = quantize(x,8)
    variable_summaries_scale('batch_normalization_2')

    # dsc2
    x = depthwise_conv2d_twn('DSC2', x)
    tf.summary.histogram('DSC2', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc2', x)
    x = quantize(x,8)
    variable_summaries_scale('batch_normalization_3')

    x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC2_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps2', x)
    x = quantize(x,8)
    variable_summaries_scale('batch_normalization_4')

    # dsc3
    x = depthwise_conv2d_bwn('DSC3', x)
    tf.summary.histogram('DSC3', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc3', x)
    x = quantize(x,8)
    variable_summaries_scale('batch_normalization_5')

    x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC3_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps3', x)
    x = quantize(x,8)
    variable_summaries_scale('batch_normalization_6')

    # dsc4
    x = depthwise_conv2d_bwn('DSC4', x, padding='VALID')
    tf.summary.histogram('DSC4', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNdsc4', x)
    x = quantize(x,8)
    variable_summaries_scale('batch_normalization_7')

    x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC4_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps4', x)
    x = quantize(x,8)
    variable_summaries_scale('batch_normalization_8')

    # fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x = flatten(x)
    x = dense_twn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    x = batchnormalization(x, is_training=is_training)
    variable_summaries_scale('batch_normalization_9', 11)

    tf.summary.histogram('BNfc', x)
    return x
def group2_conv_twn(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
    x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
    tf.summary.histogram('CONV1', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BN1', x)
    x = quantize(x,8)
    variable_summaries_scale('batch_normalization')
    # dsc1
    x = grouped_conv2d_twn('GC1', x = x,num_filters=32,num_groups=32)
    tf.summary.histogram('GC1', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNgc1', x)
    x = quantize(x,8)

    x = conv2d_twn('GC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('GC1_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNpw1', x)
    x = quantize(x,8)

    # dsc2
    x = grouped_conv2d_twn('GC2', x = x,num_filters=32,num_groups=32)
    tf.summary.histogram('GC2', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNgc2', x)
    x = quantize(x,8)

    x = conv2d_twn('GC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('GC2_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps2', x)
    x = quantize(x,8)

    # dsc3
    x = grouped_conv2d_twn('GC3', x = x,num_filters=32,num_groups=32)
    tf.summary.histogram('GC3', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNgc3', x)
    x = quantize(x,8)

    x = conv2d_twn('GC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('GC3_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps3', x)
    x = quantize(x,8)

    # dsc4
    x = grouped_conv2d_twn('GC4', x = x,num_filters=32,num_groups=32,padding='VALID')
    tf.summary.histogram('GC4', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNgc4', x)
    x = quantize(x,8)

    x = conv2d_twn('GC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('GC4_pointwise', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNps4', x)
    x = quantize(x,8)

    # fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x = flatten(x)
    x = dense_twn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    x = batchnormalization(x, is_training=is_training)
    tf.summary.histogram('BNfc', x)
    return x


def model3(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1]) # N H W C mode #8 replace =binarize
    x = conv2d_twn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
    tf.summary.histogram('CONV1', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BN1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization')
    #dsc1
    x = depthwise_conv2d_twn('DSC1', x)
    tf.summary.histogram('DSC1', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_1')

    x = conv2d_twn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC1_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNpw1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_2')

    #dsc2
    x = depthwise_conv2d_twn('DSC2', x)
    tf.summary.histogram('DSC2', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc2', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_3')

    x = conv2d_twn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC2_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps2', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_4')

    #dsc3
    x = depthwise_conv2d_twn('DSC3', x)
    tf.summary.histogram('DSC3', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc3', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_5')

    x = conv2d_twn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC3_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps3', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_6')


    #dsc4
    x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
    tf.summary.histogram('DSC4', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc4', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_7')

    x = conv2d_twn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC4_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps4', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_8')

    #fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x= flatten(x)
    x = dense_twn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    x = batchnormalization(x,is_training = is_training)
    variable_summaries_scale('batch_normalization_9',11)

    tf.summary.histogram('BNfc', x)
    return x



def model4(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1]) # N H W C mode #8 replace =binarize
    x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=128, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
    tf.summary.histogram('CONV1', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BN1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization',128)
    #dsc1
    x = depthwise_conv2d_bwn('DSC1', x)
    tf.summary.histogram('DSC1', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_1',128)

    x = conv2d_bwn('DSC1_pointwise', x, num_filters=128, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC1_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNpw1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_2',128)

    #dsc2
    x = depthwise_conv2d_bwn('DSC2', x)
    tf.summary.histogram('DSC2', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc2', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_3',128)

    x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC2_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps2', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_4')

    #dsc3
    x = depthwise_conv2d_bwn('DSC3', x)
    tf.summary.histogram('DSC3', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc3', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_5')

    x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC3_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps3', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_6')

    #dscout
    x = depthwise_conv2d_bwn('DSCout', x, padding='VALID')
    tf.summary.histogram('DSCout', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdscout', x)
    x = hard_tanh(x)

    x = conv2d_bwn('DSCout_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSCout_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNpsout', x)
    x = hard_tanh(x)

    #fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x= flatten(x)
    x = dense_bwn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    x = batchnormalization(x,is_training = is_training)

    tf.summary.histogram('BNfc', x)
    return x

def model5(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1]) # N H W C mode #8 replace =binarize
    x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=128, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
    tf.summary.histogram('CONV1', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BN1', x)
    x = binarize(x)
    #variable_summaries_scale('batch_normalization',128)
    #dsc1
    x = depthwise_conv2d_bwn('DSC1', x)
    tf.summary.histogram('DSC1', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc1', x)
    x = binarize(x)
    #variable_summaries_scale('batch_normalization_1')

    x = conv2d_bwn('DSC1_pointwise', x, num_filters=128, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC1_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNpw1', x)
    x = binarize(x)
    #variable_summaries_scale('batch_normalization_2')

    #dsc2
    x = depthwise_conv2d_bwn('DSC2', x)
    tf.summary.histogram('DSC2', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc2', x)
    x = binarize(x)
    #variable_summaries_scale('batch_normalization_3')

    x = conv2d_bwn('DSC2_pointwise', x, num_filters=128, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC2_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps2', x)
    x = binarize(x)
   # variable_summaries_scale('batch_normalization_4')

    #dsc3
    x = depthwise_conv2d_bwn('DSC3', x)
    tf.summary.histogram('DSC3', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc3', x)
    x = binarize(x)
   #variable_summaries_scale('batch_normalization_5')

    x = conv2d_bwn('DSC3_pointwise', x, num_filters=128, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC3_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps3', x)
    x = binarize(x)
    #variable_summaries_scale('batch_normalization_6')


    #dsc4
    x = depthwise_conv2d_bwn('DSC4', x, padding='VALID')
    tf.summary.histogram('DSC4', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc4', x)
    x = binarize(x)
   # variable_summaries_scale('batch_normalization_7')

    x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC4_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps4', x)
    x = binarize(x)
   # variable_summaries_scale('batch_normalization_8')

    #fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x= flatten(x)
    x = dense_bwn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    x = batchnormalization(x,is_training = is_training)
    #variable_summaries_scale('batch_normalization_9',11)

    tf.summary.histogram('BNfc', x)
    return x
def model6(fingerprint_input, model_settings, is_training):
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1]) # N H W C mode #8 replace =binarize
    x = conv2d_bwn('CONV1', fingerprint_4d, num_filters=64, kernel_size=(10, 4), padding='VALID', stride=(2, 2))
    tf.summary.histogram('CONV1', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BN1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization')
    #dsc1
    x = depthwise_conv2d_bwn('DSC1', x)
    tf.summary.histogram('DSC1', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_1')

    x = conv2d_bwn('DSC1_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC1_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNpw1', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_2')

    #dsc2
    x = depthwise_conv2d_bwn('DSC2', x)
    tf.summary.histogram('DSC2', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc2', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_3')

    x = conv2d_bwn('DSC2_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC2_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps2', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_4')

    #dsc3
    x = depthwise_conv2d_bwn('DSC3', x)
    tf.summary.histogram('DSC3', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc3', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_5')

    x = conv2d_bwn('DSC3_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC3_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps3', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_6')


    #dsc4
    x = depthwise_conv2d_bwn('DSC4', x, padding='VALID')
    tf.summary.histogram('DSC4', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNdsc4', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_7')

    x = conv2d_bwn('DSC4_pointwise', x, num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
    tf.summary.histogram('DSC4_pointwise', x)
    x = batchnormalization(x,is_training = is_training)
    tf.summary.histogram('BNps4', x)
    x = hard_tanh(x)
    variable_summaries_scale('batch_normalization_8')

    #fc
    x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
    x= flatten(x)
    x = dense_bwn('FC', x, output_dim=11)
    tf.summary.histogram('FC', x)
    x = batchnormalization(x,is_training = is_training)
    variable_summaries_scale('batch_normalization_9',11)

    tf.summary.histogram('BNfc', x)
    return x
