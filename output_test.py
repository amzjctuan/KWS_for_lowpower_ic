from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import input_data_filler
import models
import matplotlib.pyplot as plt
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from layer import *
wanted_words='five,happy,left,marvin,nine,seven,sheila,six,stop,zero'

model_architecture = 'fixed_point_twn_test'

def np_round_and_clip(x):
    x = np.round(x)
    x = np.clip(x, -50, 50)/64
    return x
def prepare_processing_graph(file,window_size_samples,window_stride_samples,dct_coefficient_count):
    desired_samples = 16000
    wav_filename_placeholder_ = file
    wav_loader = io_ops.read_file(wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(
        # wav_loader, desired_channels=1, desired_samples=desired_samples)
        wav_loader, desired_channels=1, desired_samples=16000)
    # Allow the audio sample's volume to be adjusted.
    # Shift the sample's start position, and pad any gaps with zeros.
    wave_input = tf.clip_by_value(wav_decoder.audio, -1.0, 1.0)
    ######################  M F C C #################################
    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = contrib_audio.audio_spectrogram(
        wave_input,
        window_size=window_size_samples,
        stride=window_stride_samples,
        magnitude_squared=True)
    mfcc_ = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=dct_coefficient_count)
    return mfcc_

filepath = 'happy.wav'
clip_duration_ms = 1000
window_size_ms = 40
window_stride_ms = 20
dct_coefficient_count = 16
sample_rate = 16000
###     model_settings     ###
desired_samples = int(sample_rate * clip_duration_ms / 1000)
window_size_samples = int(sample_rate * window_size_ms / 1000)
window_stride_samples = int(sample_rate * window_stride_ms / 1000)
length_minus_window = (desired_samples - window_size_samples)
if length_minus_window < 0:
    spectrogram_length = 0
else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
fingerprint_size = dct_coefficient_count * spectrogram_length
##########################################################################
print("1/ model_setting allready！")

sess = tf.InteractiveSession()
mfcc = prepare_processing_graph(filepath,window_size_samples,window_stride_samples,dct_coefficient_count)
mfcc = sess.run(mfcc)
print("MFCC shape: " +str(mfcc.shape))
features = np_round_and_clip(mfcc[0])

print("2/ MFCC feature allready！")
##########################################################################
print("3/ Read Weights ！")
print("******************")
start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/speech_commands_train/h5_40_20ms16_1word_happy_dscv1_mix1Q6nobias/dscv1/dscv1.ckpt-2000'
reader = tf.train.NewCheckpointReader(start_checkpoint)
all_variables = reader.get_variable_to_shape_map()
def read_weight(name,reader,sess):
    weights = reader.get_tensor(name)
    weights = ternary(tf.convert_to_tensor(weights))
    w = sess.run(weights)
    print(name + " shape :" + str(w.shape))
    #print(name + " sample :" + str(w[:, :, 0, :]))
    return w
def read_weight_v2(name,reader):  #not ternary
    w = reader.get_tensor(name)
    print(name + " shape :" + str(w.shape))
    return w

def read_bn(name,reader):
    beta = read_weight_v2(name + '/beta',reader)
    gamma = read_weight_v2(name +'/gamma',reader)
    moving_mean = read_weight_v2(name + '/moving_mean',reader)
    moving_variance = read_weight_v2(name +'/moving_variance',reader)
    divider = np.sqrt(moving_variance + 0.001)
    mul = gamma/divider
    # print(gamma)
    # print(beta)
    bias = -1.0 * moving_mean + beta/mul
    return mul,bias
#############weights##################
conv1_w = read_weight_v2('CONV1_layer/CONV1/conv/weights',reader)

DSC1_w = read_weight_v2('DSC1_layer/DSC1/conv/weights',reader)
DSC1pw_w = read_weight_v2('DSC1_layer/DSC1_pointwise/conv/weights',reader)

DSC2_w = read_weight_v2('DSC2_layer/DSC2/conv/weights',reader)
DSC2pw_w = read_weight_v2('DSC2_layer/DSC2_pointwise/conv/weights',reader)

DSC3_w = read_weight_v2('DSC3_layer/DSC3/conv/weights',reader)
DSC3pw_w = read_weight_v2('DSC3_layer/DSC3_pointwise/conv/weights',reader)

DSC4_w = read_weight_v2('DSC4_layer/DSC4/conv/weights',reader)
DSC4pw_w = read_weight_v2('DSC4_layer/DSC4_pointwise/conv/weights',reader)

FCout_w = read_weight_v2('FCout_layer/FCout/dense/weights',reader)
##############################################################
#############bn parameters##################
conv1_bn_mul,conv1_bn_bias = read_bn('CONV1_layer/BN1',reader)

dsc1_bn_mul,dsc1_bn_bias = read_bn('DSC1_layer/BNdsc1',reader)
dsc1pw_bn_mul,dsc1pw_bn_bias = read_bn('DSC1_layer/BNpw1',reader)

dsc2_bn_mul,dsc2_bn_bias = read_bn('DSC2_layer/BNdsc2',reader)
dsc2pw_bn_mul,dsc2pw_bn_bias = read_bn('DSC2_layer/BNps2',reader)

dsc3_bn_mul,dsc3_bn_bias = read_bn('DSC3_layer/BNdsc3',reader)
dsc3pw_bn_mul,dsc3pw_bn_bias = read_bn('DSC3_layer/BNps3',reader)

dsc4_bn_mul,dsc4_bn_bias = read_bn('DSC4_layer/BNdsc4',reader)
dsc4pw_bn_mul,dsc4pw_bn_bias = read_bn('DSC4_layer/BNps4',reader)

FC_bn_mul,FC_bn_bias = read_bn('FCout_layer/BNfcout',reader)
def plot_hist(name,x,xx,i):
    plt.subplot(2,10,i)
    plt.title(name + 'mul')
    plt.hist(x, bins=60, color='steelblue', normed=True)
    plt.subplot(2,10,i+10)
    plt.title(name + 'bias')
    plt.hist(xx, bins=60, color='steelblue', normed=True)
plot_hist('CONV1_layer',conv1_bn_mul,conv1_bn_bias,1)
plot_hist('dsc1',dsc1_bn_mul,dsc1_bn_bias,2)
plot_hist('dsc1pw',dsc1pw_bn_mul,dsc1pw_bn_bias,3)
plot_hist('dsc2',dsc2_bn_mul,dsc2_bn_bias,4)
plot_hist('dsc2pw',dsc2pw_bn_mul,dsc2pw_bn_bias,5)
plot_hist('dsc3',dsc3_bn_mul,dsc3_bn_bias,6)
plot_hist('dsc3pw',dsc3pw_bn_mul,dsc3pw_bn_bias,7)
plot_hist('dsc4',dsc4_bn_mul,dsc4_bn_bias,8)
plot_hist('dsc4pw',dsc4pw_bn_mul,dsc4pw_bn_bias,9)
plot_hist('fc',FC_bn_mul,FC_bn_bias,10)

plt.show()
print("3/ Read weights complete ！")
##############################################################
##############################################################

print("******************")
print("4/ Creat Model ！")

input_frequency_size = dct_coefficient_count
input_time_size = spectrogram_length
label_count = 11
is_training = False
fingerprint_4d = tf.reshape(features,[-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
x = tf.nn.conv2d(fingerprint_4d, conv1_w, [1, 2, 2, 1], padding='VALID')
x = tf.multiply(tf.add(x,conv1_bn_bias),conv1_bn_mul)
x = quantize(x,8)
conv1_out = sess.run(x)
print(conv1_out.shape)
x = tf.nn.depthwise_conv2d(x, DSC1_w, [1, 1, 1, 1], padding='VALID')

'''
print("******************")
input_frequency_size = dct_coefficient_count
input_time_size = spectrogram_length
label_count = 11
is_training = False
fingerprint_4d = tf.reshape(features,[-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
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
    x = depthwise_conv2d_twn('DSC4', x, padding='VALID')
    #x = depthwise_conv2d_twn('DSC4', x)
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
    x = dense_twn('FCout', x, output_dim=label_count)
    tf.summary.histogram('FCout_before_bn', x)
    x = batchnormalization_hardware(x, is_training=is_training, name='BNfcout')
    tf.summary.histogram('FCout_out', x)
'''