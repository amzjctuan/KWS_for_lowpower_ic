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

data_url = None
data_dir = 'tmp/speech_dataset_raw'
silence_percentage = 2
unknown_percentage = 80
validation_percentage = 10
testing_percentage = 10
background_volume = 0.1
background_frequency = 0.8
sample_rate = 16000
batch_size = 100
time_shift_ms = 200.0

wanted_words ='yes,no,up,down,left,right,on,off,stop,go'
#wanted_words ='yes,no'
#wanted_words='five,happy,left,marvin,nine,seven,sheila,six,stop,zero'

model_architecture = 'l_ds_bnn_conv'
clip_duration_ms = 1000
window_size_ms = 40
window_stride_ms = 20
dct_coefficient_count = 10

pbs_path = 'tmp/pbs/l_ds_conv5000.pb'




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
        -1, 0, model_settings, background_frequency,
        background_volume, time_shift_samples, 'testing', sess)

print("* fingerprint size          ； " + str(model_settings['fingerprint_size']))
print("* test set examples number  ； " + str(np.sum(np.sum(test_ground_truth, axis=0))))
print("* test set features size    ； " + str(test_fingerprints.shape))
print("* test set labels size      ； " + str(test_ground_truth.shape))
if model_settings['fingerprint_size']==test_fingerprints.shape[1] and \
        len(input_data_filler.prepare_words_list_my(wanted_words.split(','))) == test_ground_truth.shape[1] and \
        test_fingerprints.shape[0] == testing_datas:
    print("------------->  ALL CORRECT <--------------")
else:
    print("------------->  DATA WRONG! <--------------")
print(" ********************************************************" + '\n')
print(" ***************       import pbs     *******************")
print(tf.get_default_graph())
with tf.Graph().as_default():
    graph_def = tf.GraphDef()
    with open(pbs_path,"rb") as f:
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def,name = "")
    # sess.run(tf.global_variables_initializer())

    # print("graph_def:",graph_def)
    # with graph_def.as_default():
    #     print(tf.get_default_graph())
    print(graph_def)

    input_x = sess.graph.get_tensor_by_name("ds4_pointwise_conv/moments/variance/reduction_indices")
    print(input_x)
    # input_label = sess.graph.get_tensor_by_name("ground_truth_input")
    # output_y = sess.graph.get_tensor_by_name("accuracy")
    # print(input_x)
    # test_acc =  sess.run(output_y,feed_dict = {input_x:test_fingerprints,input_label: test_ground_truth})
    # print(test_acc)
print(" ********************************************************" + '\n')

