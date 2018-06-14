from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf

import new_features_input
import models
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
from numpy import asfarray


data_dir ='/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/KWSFeature/'
silence_percentage = 3
unknown_percentage = 80
validation_percentage = 10
testing_percentage = 10
sample_rate = 16000
batch_size = 100
#wanted_words ='yes,no,up,down,left,right,on,off,stop,go'
#wanted_words ='yes,no'
wanted_words='five,happy,left,marvin,nine,seven,sheila,six,stop,zero'

clip_duration_ms = 1000
window_size_ms = 25
window_stride_ms = 10
dct_coefficient_count = 16
start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/speech_commands_train/hellow_edge_dscnn_test/hellow_edge_dscnn_test.ckpt-10400'
#start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/moxing/ds_conv_2_5000_0.09filler/l_ds_conv_2.ckpt-5000'
#start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech/tmp/moxing/ds_conv_2_5000_0.09filler/l_ds_conv_2.ckpt-5000'
pbs_path = ''





# We want to see all the logging messages for this tutorial.
tf.logging.set_verbosity(tf.logging.INFO)

# Start a new TensorFlow session.
sess = tf.InteractiveSession()

model_settings = models.prepare_model_settings(
  len(new_features_input.prepare_words_list_my(wanted_words.split(','))),
  sample_rate, clip_duration_ms, window_size_ms,
  window_stride_ms, dct_coefficient_count)
audio_processor = new_features_input.AudioProcessor(data_dir, silence_percentage,
    wanted_words.split(','), validation_percentage,
  testing_percentage)

fingerprint_size = model_settings['fingerprint_size']
label_count = model_settings['label_count']
print("spectrogram_length: " + str(model_settings['spectrogram_length']))
print("dct_coefficient_count: " + str(model_settings['dct_coefficient_count']))
print('fingerprint_size: ' + str(fingerprint_size))

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
    -1, 0, model_settings,'training')


print("* fingerprint size          ； " + str(model_settings['fingerprint_size']))
print("* test set examples number  ； " + str(np.sum(np.sum(test_ground_truth, axis=0))))
print("* test set features size    ； " + str(test_fingerprints.shape))
print("* test set labels size      ； " + str(test_ground_truth.shape))
if model_settings['fingerprint_size']==test_fingerprints.shape[1] and \
        len(new_features_input.prepare_words_list_my(wanted_words.split(','))) == test_ground_truth.shape[1] :
    print("------------->  ALL CORRECT <--------------")
else:
    print("------------->  DATA WRONG! <--------------")
print(" ********************************************************" + '\n')
print(" ***************       import ckpt    *******************")
print(test_fingerprints)

