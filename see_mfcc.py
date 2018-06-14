from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf

import input_data_filler
import models
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
#from confuse_test import *
import matplotlib.pyplot as plt
from numpy import asfarray
data_url = None
data_dir = 'tmp/speech_dataset_raw'
silence_percentage = 3
unknown_percentage = 80
validation_percentage = 10
testing_percentage = 10
background_volume = 0.1
background_frequency = 0.8
sample_rate = 16000
batch_size = 100
time_shift_ms = 100.0

#wanted_words ='yes,no,up,down,left,right,on,off,stop,go'
#wanted_words ='yes,no'
wanted_words='five,happy,left,marvin,nine,seven,sheila,six,stop,zero'

model_architecture = 'hellow_edge_dscnn_test'
clip_duration_ms = 1000
window_size_ms = 40
window_stride_ms = 20
dct_coefficient_count = 10
start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/speech_commands_train/hellow_edge_dscnn_test/hellow_edge_dscnn_test.ckpt-10400'
#start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/moxing/ds_conv_2_5000_0.09filler/l_ds_conv_2.ckpt-5000'
#start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech/tmp/moxing/ds_conv_2_5000_0.09filler/l_ds_conv_2.ckpt-5000'
pbs_path = ''





# We want to see all the logging messages for this tutorial.
tf.logging.set_verbosity(tf.logging.INFO)

# Start a new TensorFlow session.
sess = tf.InteractiveSession()

model_settings = models.prepare_model_settings(
  len(input_data_filler.prepare_words_list_my(wanted_words.split(','))),
  sample_rate, clip_duration_ms, window_size_ms,
  window_stride_ms, dct_coefficient_count)
audio_processor = input_data_filler.AudioProcessor(
  data_url, data_dir, silence_percentage,
  unknown_percentage,
  wanted_words.split(','), validation_percentage,
  testing_percentage, model_settings)
time_shift_samples = int((time_shift_ms * sample_rate) / 1000)

fingerprint_size = model_settings['fingerprint_size']
label_count = model_settings['label_count']
# fingerprint_input = tf.placeholder(
#       tf.float32, [None, fingerprint_size], name='fingerprint_input')
# ground_truth_input = tf.placeholder(
#     tf.float32, [None, label_count], name='groundtruth_input')
#
# logits= models.create_model(
#       fingerprint_input,
#       model_settings,
#       model_architecture,
#       is_training=False)
# softmax = tf.nn.softmax(logits, name='labels_softmax')
#
# with tf.name_scope('cross_entropy'):
#     cross_entropy_mean = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(
#             labels=ground_truth_input, logits=logits))
# predicted_indices = tf.argmax(logits, 1)
# expected_indices = tf.argmax(ground_truth_input, 1)
# correct_prediction = tf.equal(predicted_indices, expected_indices)
# confusion_matrix = tf.confusion_matrix(
#       expected_indices, predicted_indices, num_classes=label_count)
# evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# global_step = tf.train.get_or_create_global_step()
# increment_global_step = tf.assign(global_step, global_step + 1)
# tf.global_variables_initializer().run()
# if start_checkpoint:
#     models.load_variables_from_checkpoint(sess, start_checkpoint)
#     start_step = global_step.eval(session=sess)
#*********************************************************************
print(" *****************  audio processor  ********************")

training_datas = len(audio_processor.data_index['training']) + len(audio_processor.unknown_index['training'])
validation_datas = len(audio_processor.data_index['validation']) + len(audio_processor.unknown_index['validation'])
testing_datas = len(audio_processor.data_index['testing']) + len(audio_processor.unknown_index['testing'])
print("* total      samples :  " + str(training_datas+validation_datas + testing_datas))
print("* training   samples :  "+str(len(audio_processor.data_index['training']))  + ' + ' \
                                 + str(len(audio_processor.unknown_index['training']))  + '(unknowns)' + ' = ' + str(training_datas))
print("* validation samples :  "+str(len(audio_processor.data_index['validation']))+ ' +  ' \
                                 + str(len(audio_processor.unknown_index['validation']))+ ' (unknowns)' + ' = ' + str(validation_datas))
print("* testing    samples :  "+str(len(audio_processor.data_index['testing']))   + ' +  ' \
                                 + str(len(audio_processor.unknown_index['testing']))   + ' (unknowns)' + ' = ' + str(testing_datas))
print(" ********************************************************" + '\n')
#*********************************************************************
print(" ***************  Features generator  *******************")
test_fingerprints, test_ground_truth= audio_processor.get_data_my(
    2000, 0, model_settings, 0, 0, 0, 'testing', sess)


print("* fingerprint size          ； " + str(model_settings['fingerprint_size']))
print("* test set examples number  ； " + str(np.sum(np.sum(test_ground_truth, axis=0))))
print("* test set features size    ； " + str(test_fingerprints.shape))
print("* test set labels size      ； " + str(test_ground_truth.shape))
if model_settings['fingerprint_size']==test_fingerprints.shape[1] and \
        len(input_data_filler.prepare_words_list_my(wanted_words.split(','))) == test_ground_truth.shape[1] :
    print("------------->  ALL CORRECT <--------------")
else:
    print("------------->  DATA WRONG! <--------------")
print(" ********************************************************" + '\n')
print(" ***************       import ckpt    *******************")
#print(np.round(test_fingerprints[0].reshape([49,10])))
plt.hist(test_fingerprints.flatten(), bins=200, color='steelblue', normed=True)
plt.show()
# draw = test_fingerprints[1:1000]
# draw = draw.flatten()
# plt.hist(draw, bins=100)
# plt.show()
# print(np.max(test_fingerprints))
# print(np.min(test_fingerprints))
# print(np.mean(test_fingerprints))
# mid = np.abs(np.max(test_fingerprints)+np.min(test_fingerprints))/2
# min = np.min(test_fingerprints)
# print("mid = " + str(mid))
# gap = np.max(test_fingerprints)-np.min(test_fingerprints)
# print("half = " + str(gap))
# scale = ((test_fingerprints-min)/gap*255).astype(np.uint8)
# print(scale[0][0])
# print(np.max(scale))
# print(np.min(scale))
# print(np.mean(scale))
# shape = np.shape(scale)
# scale_re = np.reshape(scale,[-1,49,10])
# print(scale_re.shape)
# scale_re = scale_re[:,:,:,np.newaxis]
# print(scale_re.shape)
# scale_re = np.array(scale_re,dtype=np.uint8)
# print(scale_re.shape)
# final_scale = np.unpackbits(scale_re,axis=3)
# print(final_scale.shape)
# print(final_scale[0][0][0])
# def fingerprints_to_binary(fingerprints_in,time_size,frequence_size):
#     # print('fingerprints input MAX: ' + str(np.max(fingerprints_in)))
#     # print('fingerprints input MIN: ' + str(np.min(fingerprints_in)))
#     x = np.round(fingerprints_in)
#     x = np.clip(fingerprints_in, -100, 100) + 100
#     scale = x.astype(np.uint8)
#     scale_re = np.reshape(scale, [-1, time_size, frequence_size])
#     scale_re = scale_re[:, :, :, np.newaxis]
#     scale_re = np.array(scale_re, dtype=np.uint8)
#     final_scale = np.unpackbits(scale_re, axis=3)
#     return final_scale
# final = fingerprints_to_binary(test_fingerprints,49,10)
# print("test:" + str(final[0][0][0]))
# print(test_fingerprints[0][0])


# x = np.round(test_fingerprints)
# x = np.clip(x, -50, 49) + 50
# x = np.reshape(x, [-1, 49, 10])
# print(x)
# print(x.shape)
# shape_onehot=list(x.shape)+[100]
# onehot_x = np.zeros(shape_onehot)
# print(onehot_x.shape)
# for n in range(x.shape[0]):
#     for i in range(49):
#         for j in range(10):
#             k = int(x[n,i,j])
#             onehot_x[n,i,j,k]=1
# print(onehot_x.shape)
# print(onehot_x[0,0,0])
# print(x[0,0,0])

# array = np.array(np.reshape(test_fingerprints[:3],[3,49,10]))
# a = array[0]
# b = array[1]
# c = array[2]
#
# save_path = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/'
# np.savetxt(save_path +"10_mfcc_data_array.csv",(a,b,c),delimiter=',')
#
# def mixup_data(x,y,alpha = 1):
#     if alpha >0:
#         lam = np.random.beta(alpha,alpha)
#     else:
#         lam = 1
#     batch_size = len(list(np.array(x,dtype=np.float)))
#     index = list(np.arange(batch_size))
#     np.random.shuffle(index)
#     mixed_x = lam*x + (1-lam)*x[index,:]
#     mixed_y = lam*y + (1-lam)*y[index,:]
#     return mixed_x,mixed_y
#
# x,y = mixup_data(test_fingerprints, test_ground_truth,1)
# print(x.shape)
# print(test_fingerprints)
# print('//////////')
# print(x)
#
# print(y.shape)
# print(y)
# print(test_ground_truth)
# test_fingerprints = np.append(test_fingerprints,x,axis=0)
# test_ground_truth = np.append(test_ground_truth,y,axis=0)
# random_index = list(np.arange(200))
# np.random.shuffle(random_index)
# test_fingerprints = test_fingerprints[random_index,:]
# test_ground_truth = test_ground_truth[random_index,:]
# print(random_index)
# print(test_fingerprints)
# print(test_fingerprints.shape)
# print(test_ground_truth)
# print(test_ground_truth.shape)