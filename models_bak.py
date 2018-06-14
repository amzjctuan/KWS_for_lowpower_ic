# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'single_fc':
    return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'l_bn_conv':
    return l_bn_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'l_bnn_conv':
    return l_bnn_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'l_conv':
    return l_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'l_ds_conv':
    return l_ds_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'l_ds_conv_2':
    return l_ds_conv_2_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'l_ds_bnn_conv':
    return l_ds_bnn_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'l_ds_bnn_conv_2':
    return l_ds_bnn_conv_2_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'cnn_one_fpool3':
    return cnn_one_fpool3(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'l_allbnn_conv':
    return l_allbnn_conv_model(fingerprint_input, model_settings,
                                         is_training)

  elif model_architecture == 'low_latency_svdf':
    return create_low_latency_svdf_model(fingerprint_input, model_settings,
                                         is_training, runtime_settings)
  elif model_architecture == 'dnn':
    return create_2017Google_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'dnn_bnn_retrain':
      return dnn_bnn_retrain(fingerprint_input, model_settings,
                                     is_training)
  elif model_architecture == 'dnn_bn_bnn_3n':
      return dnn_bn_bnn_3n(fingerprint_input, model_settings,
                                     is_training)
  elif model_architecture == 'dnn_bn_bnn_4n':
      return dnn_bn_bnn_4n(fingerprint_input, model_settings,
                                     is_training)
  elif model_architecture == 'dnn_bn_bnn_5n':
      return dnn_bn_bnn_5n(fingerprint_input, model_settings,
                      is_training)
  elif model_architecture == 'dnn_test':
      return dnn_test(fingerprint_input, model_settings,
                                     is_training)
  elif model_architecture == 'dnn_dp':
      return dnn_dp(fingerprint_input, model_settings,
                      is_training)
  elif model_architecture == 'dnn_bn_all_bnn_4n':
      return dnn_bn_all_bnn_4n(fingerprint_input, model_settings,
                      is_training)
  elif model_architecture == 'dnn_bn_all_bnn_5n':
      return dnn_bn_all_bnn_5n(fingerprint_input, model_settings,
                      is_training)
  elif model_architecture == 'dnn_4n_dp':
      return dnn_4n_dp(fingerprint_input, model_settings,
                      is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_single_fc_model(fingerprint_input, model_settings, is_training):
  """Builds a model with a single hidden fully-connected layer.

  This is a very simple model with just one matmul and bias layer. As you'd
  expect, it doesn't produce very accurate results, but it is very fast and
  simple, so it's useful for sanity testing.

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  weights = tf.Variable(
      tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
  bias = tf.Variable(tf.zeros([label_count]))
  logits = tf.matmul(fingerprint_input, weights) + bias
  if is_training:
    return logits, dropout_prob
  else:
    return logits


def create_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.

  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
   dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 4
  first_filter_height = 10
  first_filter_count = 28
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 30
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc
def l_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.

  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
   dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 10
  first_filter_height = 4
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  print("first_conv weight:"+str(first_conv.get_shape()))
  first_relu = tf.nn.relu(first_conv)
  # if is_training:
  #   first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  # else:
  #   first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  print("first_pool weight:"+str(max_pool.get_shape()))
  second_filter_width = 10
  second_filter_height = 4
  second_filter_count = 48
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))

  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 2, 1, 1],
                             'SAME') + second_bias
  print("second_conv weight:"+str(second_conv.get_shape()))
  second_relu = tf.nn.relu(second_conv)
  max_pool2 = tf.nn.max_pool(second_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  print("second_pool weight:"+str(max_pool2.get_shape()))
  # if is_training:
  #   second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  # else:
  #   second_dropout = second_relu
  second_conv_shape = max_pool2.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(max_pool2,
                                     [-1, second_conv_element_count])
  print("element count FC:"+ str(second_conv_element_count))
  label_count = model_settings['label_count']
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, 16], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([16]))
  first_fc = tf.matmul(flattened_second_conv, first_fc_weights) + first_fc_bias

  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [16, 128], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([128]))
  second_fc = tf.matmul(first_fc, second_fc_weights) + second_fc_bias
  second_fc = tf.nn.relu(second_fc)

  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [128, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(second_fc, final_fc_weights) + final_fc_bias
  weight_num = first_filter_height*first_filter_width*1*first_filter_count\
               + second_filter_height*second_filter_width*first_filter_count*second_filter_count\
               +second_conv_element_count*16+16*128                    #weights num
  print("Weights number:"+str(weight_num))
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc

def cnn_one_fpool3(fingerprint_input, model_settings, is_training):

  if is_training:
   dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 32
  first_filter_height = 8
  first_filter_count = 54
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME')
  #first_conv = batch_normalization_layer(first_conv, 'first_conv', is_training)  #batchnormalization
  first_relu = tf.nn.relu(first_conv)
  max_pool = tf.nn.max_pool(first_relu, [1, 1, 3, 1], [1, 1, 3, 1], 'SAME')
  
  conv_shape = max_pool.get_shape()
  conv_output_width = conv_shape[2]
  conv_output_height = conv_shape[1]
  conv_element_count = int(
      conv_output_width * conv_output_height *
      first_filter_count)
  flattened_conv = tf.reshape(max_pool,
                                     [-1, conv_element_count])
  label_count = model_settings['label_count']
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [conv_element_count, 32], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([32]))
  first_fc = tf.matmul(flattened_conv, first_fc_weights) + first_fc_bias

  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [32, 128], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([128]))
  second_fc = tf.matmul(first_fc, second_fc_weights) + second_fc_bias
  second_fc = tf.nn.relu(second_fc)
  
  third_fc_weights = tf.Variable(
      tf.truncated_normal(
          [128, 128], stddev=0.01))
  third_fc_bias = tf.Variable(tf.zeros([128]))
  third_fc = tf.matmul(second_fc, third_fc_weights) + third_fc_bias
  third_fc = tf.nn.relu(third_fc)
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [128, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(third_fc, final_fc_weights) + final_fc_bias
  weight_num  =   first_filter_width *first_filter_height *first_filter_count+conv_element_count*32+32*128+128*128+128*11
  print("weight num :" + str(weight_num))
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def l_bn_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.

  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
   dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 10
  first_filter_height = 4
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME')
  first_conv = batch_normalization_layer(first_conv, 'first_conv', is_training)  #batchnormalization
  first_relu = tf.nn.relu(first_conv)
  max_pool = tf.nn.max_pool(first_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 10
  second_filter_height =4
  second_filter_count = 48
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 2, 1, 1],
                             'SAME')
  second_conv = batch_normalization_layer(second_conv, 'second_conv',is_training)  #batchnormalization
  second_relu = tf.nn.relu(second_conv)
  max_pool2 = tf.nn.max_pool(second_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_conv_shape = max_pool2.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(max_pool2,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, 16], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([16]))
  first_fc = tf.matmul(flattened_second_conv, first_fc_weights) + first_fc_bias

  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [16, 128], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([128]))
  second_fc = tf.matmul(first_fc, second_fc_weights) + second_fc_bias
  second_fc = tf.nn.relu(second_fc)

  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [128, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(second_fc, final_fc_weights) + final_fc_bias

  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc
def l_allbnn_conv_model(fingerprint_input, model_settings, is_training):

  if is_training:
   dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 10
  first_filter_height = 4
  first_filter_count = 80
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_weights = binarize(first_weights)
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME')
  first_conv = batch_normalization_layer(first_conv, 'first_conv', is_training)  #batchnormalization
  first_relu = binarize(first_conv)
  max_pool = tf.nn.max_pool(first_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 10
  second_filter_height =4
  second_filter_count = 96
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_weights = binarize(second_weights)
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 2, 1, 1],
                             'SAME')
  second_conv = batch_normalization_layer(second_conv, 'second_conv',is_training)  #batchnormalization
  second_relu = binarize(second_conv)
  max_pool2 = tf.nn.max_pool(second_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_conv_shape = max_pool2.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(max_pool2,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  lin_num = 16
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, lin_num], stddev=0.01))
  first_fc_weights = binarize(first_fc_weights)

  first_fc = tf.matmul(flattened_second_conv, first_fc_weights)

  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [lin_num, 128], stddev=0.01))
  second_fc_weights = binarize(second_fc_weights)

  second_fc = tf.matmul(first_fc, second_fc_weights)
  second_fc = batch_normalization_layer(second_fc,'second_fc',is_training)
  second_fc = binarize(second_fc)

  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [128, label_count], stddev=0.01))
  final_fc_weights = binarize(final_fc_weights)
  final_fc = tf.matmul(second_fc, final_fc_weights)
  final_fc = batch_normalization_layer(final_fc,'final_fc',is_training)
  final_fc = tf.clip_by_value(final_fc,-1,1)
  weight_num = first_filter_height*first_filter_width*1*first_filter_count\
               + second_filter_height*second_filter_width*first_filter_count*second_filter_count\
               +second_conv_element_count*lin_num+lin_num*128                    #weights num
  print("Weights number:"+str(weight_num))
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc

def l_bnn_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.
  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
   dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 10
  first_filter_height = 4
  first_filter_count = 28
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_weights = binarize(first_weights)
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME')
  first_conv = batch_normalization_layer(first_conv, 'first_conv', is_training)  #batchnormalization
  first_relu = tf.clip_by_value(first_conv,-1,1)
  max_pool = tf.nn.max_pool(first_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 10
  second_filter_height =4
  second_filter_count = 30
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_weights = binarize(second_weights)
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 2, 1, 1],
                             'SAME')
  second_conv = batch_normalization_layer(second_conv, 'second_conv',is_training)  #batchnormalization
  second_relu = tf.clip_by_value(second_conv,-1,1)
  max_pool2 = tf.nn.max_pool(second_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_conv_shape = max_pool2.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(max_pool2,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, 16], stddev=0.01))
  first_fc_weights = binarize(first_fc_weights)

  first_fc = tf.matmul(flattened_second_conv, first_fc_weights)

  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [16, 128], stddev=0.01))
  second_fc_weights = binarize(second_fc_weights)

  second_fc = tf.matmul(first_fc, second_fc_weights)
  second_fc = batch_normalization_layer(second_fc,'second_fc',is_training)
  second_fc = tf.clip_by_value(second_fc,-1,1)

  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [128, label_count], stddev=0.01))
  final_fc_weights = binarize(final_fc_weights)
  final_fc = tf.matmul(second_fc, final_fc_weights)
  final_fc = batch_normalization_layer(final_fc,'final_fc',is_training)
  final_fc = tf.clip_by_value(final_fc,-1,1)
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc

def l_ds_conv_model(fingerprint_input, model_settings, is_training):
  if is_training:
   dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  #######    conv1     ########
  first_filter_width = 10
  first_filter_height = 4
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 2, 2, 1], 'SAME')
  first_conv = batch_normalization_layer(first_conv, 'first_conv', is_training)  #batchnormalization
  first_relu = tf.nn.relu(first_conv)
  max_pool = tf.nn.max_pool(first_relu, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
  #######    dsconv1   ########
  ds1conv_filter_width = 3
  ds1conv_filter_height =3
  ds1conv_filter_count = 2
  ds1conv_weights = tf.Variable(
      tf.truncated_normal(
          [ds1conv_filter_height, ds1conv_filter_width, first_filter_count, ds1conv_filter_count],   stddev=0.01))
  #depthwise conv
  ds1_conv = tf.nn.depthwise_conv2d(max_pool, ds1conv_weights, strides= [1, 1, 1, 1], padding = 'SAME')
  #batchnormalization
  ds1_conv = batch_normalization_layer(ds1_conv, 'ds1_conv',is_training)  
  ds1_relu1 = tf.nn.relu(ds1_conv)
  #pointwise conv
  ds1_out_channels = 64
  ds1_pointwise_weights = tf.Variable(
      tf.truncated_normal(
          [1, 1,  first_filter_count*ds1conv_filter_count , ds1_out_channels],   stddev=0.01))
  ds1_pointwise_conv = tf.nn.conv2d(ds1_relu1, ds1_pointwise_weights,  [1, 1 ,1, 1],  'SAME')
  #batchnormalization
  ds1_pointwise_conv =  batch_normalization_layer(ds1_pointwise_conv, 'ds1_pointwise_conv',is_training)  
  ds1_relu2 = tf.nn.relu(ds1_pointwise_conv)
  
  #######    dsconv2   ########
  ds2conv_filter_width = 3
  ds2conv_filter_height =3
  ds2conv_filter_count = 2
  ds2conv_weights = tf.Variable(
  tf.truncated_normal(
     [ds2conv_filter_height, ds2conv_filter_width, ds1_out_channels, ds2conv_filter_count],   stddev=0.01))
  #depthwise conv
  ds2_conv = tf.nn.depthwise_conv2d(ds1_relu2, ds2conv_weights, strides= [1, 1, 1, 1], padding = 'SAME')
  #batchnormalization
  ds2_conv = batch_normalization_layer(ds2_conv, 'ds2_conv',is_training)  
  ds2_relu1 = tf.nn.relu(ds2_conv)
  #pointwise conv
  ds2_out_channels = 64
  ds2_pointwise_weights = tf.Variable(
  tf.truncated_normal(
      [1, 1,  ds1_out_channels*ds2conv_filter_count , ds2_out_channels],   stddev=0.01))
  ds2_pointwise_conv = tf.nn.conv2d(ds2_relu1, ds2_pointwise_weights,  [1, 1 ,1, 1],  'SAME')
  #batchnormalization
  ds2_pointwise_conv =  batch_normalization_layer(ds2_pointwise_conv, 'ds2_pointwise_conv',is_training)  
  ds2_relu2 = tf.nn.relu(ds2_pointwise_conv)
  
  #######    dsconv3   ########
  ds3conv_filter_width = 3
  ds3conv_filter_height = 3
  ds3conv_filter_count = 2
  ds3conv_weights = tf.Variable(
      tf.truncated_normal(
          [ds3conv_filter_height, ds3conv_filter_width, ds2_out_channels, ds3conv_filter_count], stddev=0.01))
  # depthwise conv
  ds3_conv = tf.nn.depthwise_conv2d(ds2_relu2, ds3conv_weights, strides=[1, 1, 1, 1], padding='SAME')
  # batchnormalization
  ds3_conv = batch_normalization_layer(ds3_conv, 'ds3_conv', is_training)
  ds3_relu1 = tf.nn.relu(ds3_conv)
  # pointwise conv
  ds3_out_channels = 64
  ds3_pointwise_weights = tf.Variable(
      tf.truncated_normal(
          [1, 1, ds2_out_channels * ds3conv_filter_count, ds3_out_channels], stddev=0.01))
  ds3_pointwise_conv = tf.nn.conv2d(ds3_relu1, ds3_pointwise_weights, [1, 1, 1, 1], 'SAME')
  # batchnormalization
  ds3_pointwise_conv = batch_normalization_layer(ds3_pointwise_conv, 'ds3_pointwise_conv', is_training)
  ds3_relu2 = tf.nn.relu(ds3_pointwise_conv)
  
  #######    dsconv4   ########
  ds4conv_filter_width = 3
  ds4conv_filter_height = 3
  ds4conv_filter_count = 2
  ds4conv_weights = tf.Variable(
      tf.truncated_normal(
          [ds4conv_filter_height, ds4conv_filter_width, ds3_out_channels, ds4conv_filter_count], stddev=0.01))
  # depthwise conv
  ds4_conv = tf.nn.depthwise_conv2d(ds3_relu2, ds4conv_weights, strides=[1, 1, 1, 1], padding='SAME')
  # batchnormalization
  ds4_conv = batch_normalization_layer(ds4_conv, 'ds4_conv', is_training)
  ds4_relu1 = tf.nn.relu(ds4_conv)
  # pointwise conv
  ds4_out_channels = 64
  ds4_pointwise_weights = tf.Variable(
      tf.truncated_normal(
          [1, 1, ds3_out_channels * ds4conv_filter_count, ds4_out_channels], stddev=0.01))
  ds4_pointwise_conv = tf.nn.conv2d(ds4_relu1, ds4_pointwise_weights, [1, 1, 1, 1], 'SAME')
  # batchnormalization
  ds4_pointwise_conv = batch_normalization_layer(ds4_pointwise_conv, 'ds4_pointwise_conv', is_training)
  ds4_relu2 = tf.nn.relu(ds4_pointwise_conv)
  
  #average pool
  avg_pool = tf.nn.avg_pool(ds4_relu2, [1, 2, 2, 1],  [1, 2, 2, 1], 'SAME')
  
  ######  FC  ########
  avg_pool_shape = avg_pool.get_shape()

  avg_pool_element_count = int(
       avg_pool_shape[2] * avg_pool_shape[1] *avg_pool_shape[3])
  
  flattened_avg_pool = tf.reshape(avg_pool, [-1, avg_pool_element_count])
  label_count = model_settings['label_count']
  fc_weights = tf.Variable(
      tf.truncated_normal(
          [avg_pool_element_count, label_count], stddev=0.01))
  fc_bias = tf.Variable(tf.zeros([label_count]))
  fc = tf.add(tf.matmul(flattened_avg_pool, fc_weights) , fc_bias)
  weights_num = first_filter_count*ds1conv_filter_count*ds1_out_channels+ds1_out_channels*ds2conv_filter_count * ds2_out_channels\
                  +ds2_out_channels * ds3conv_filter_count* ds3_out_channels + ds3_out_channels * ds4conv_filter_count*ds4_out_channels\
                  + avg_pool_element_count*label_count
  print("weights num:"+str(weights_num))
  if is_training:
    return fc, dropout_prob
  else:
    return fc


def l_ds_conv_2_model(fingerprint_input, model_settings, is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    #######    conv1     ########
    first_filter_width = 10
    first_filter_height = 4
    first_filter_count = 172
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 2, 1, 1], 'SAME')
    first_conv = batch_normalization_layer(first_conv, 'first_conv', is_training)  # batchnormalization
    first_relu = tf.nn.relu(first_conv)
    max_pool = tf.nn.max_pool(first_relu, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
    #######    dsconv1   ########
    ds1conv_filter_width = 3
    ds1conv_filter_height = 3
    ds1conv_filter_count = 2
    ds1conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds1conv_filter_height, ds1conv_filter_width, first_filter_count, ds1conv_filter_count], stddev=0.01))
    # depthwise conv
    ds1_conv = tf.nn.depthwise_conv2d(max_pool, ds1conv_weights, strides=[1, 2, 2, 1], padding='SAME')
    # batchnormalization
    ds1_conv = batch_normalization_layer(ds1_conv, 'ds1_conv', is_training)
    ds1_relu1 = tf.nn.relu(ds1_conv)
    # pointwise conv
    ds1_out_channels = 172
    ds1_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, first_filter_count * ds1conv_filter_count, ds1_out_channels], stddev=0.01))
    ds1_pointwise_conv = tf.nn.conv2d(ds1_relu1, ds1_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds1_pointwise_conv = batch_normalization_layer(ds1_pointwise_conv, 'ds1_pointwise_conv', is_training)
    ds1_relu2 = tf.nn.relu(ds1_pointwise_conv)

    #######    dsconv2   ########
    ds2conv_filter_width = 3
    ds2conv_filter_height = 3
    ds2conv_filter_count = 2
    ds2conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds2conv_filter_height, ds2conv_filter_width, ds1_out_channels, ds2conv_filter_count], stddev=0.01))
    # depthwise conv
    ds2_conv = tf.nn.depthwise_conv2d(ds1_relu2, ds2conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # batchnormalization
    ds2_conv = batch_normalization_layer(ds2_conv, 'ds2_conv', is_training)
    ds2_relu1 = tf.nn.relu(ds2_conv)
    # pointwise conv
    ds2_out_channels = 172
    ds2_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, ds1_out_channels * ds2conv_filter_count, ds2_out_channels], stddev=0.01))
    ds2_pointwise_conv = tf.nn.conv2d(ds2_relu1, ds2_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds2_pointwise_conv = batch_normalization_layer(ds2_pointwise_conv, 'ds2_pointwise_conv', is_training)
    ds2_relu2 = tf.nn.relu(ds2_pointwise_conv)

    #######    dsconv3   ########
    ds3conv_filter_width = 3
    ds3conv_filter_height = 3
    ds3conv_filter_count = 2
    ds3conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds3conv_filter_height, ds3conv_filter_width, ds2_out_channels, ds3conv_filter_count], stddev=0.01))
    # depthwise conv
    ds3_conv = tf.nn.depthwise_conv2d(ds2_relu2, ds3conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # batchnormalization
    ds3_conv = batch_normalization_layer(ds3_conv, 'ds3_conv', is_training)
    ds3_relu1 = tf.nn.relu(ds3_conv)
    # pointwise conv
    ds3_out_channels = 172
    ds3_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, ds2_out_channels * ds3conv_filter_count, ds3_out_channels], stddev=0.01))
    ds3_pointwise_conv = tf.nn.conv2d(ds3_relu1, ds3_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds3_pointwise_conv = batch_normalization_layer(ds3_pointwise_conv, 'ds3_pointwise_conv', is_training)
    ds3_relu2 = tf.nn.relu(ds3_pointwise_conv)

    #######    dsconv4   ########
    ds4conv_filter_width = 3
    ds4conv_filter_height = 3
    ds4conv_filter_count = 2
    ds4conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds4conv_filter_height, ds4conv_filter_width, ds3_out_channels, ds4conv_filter_count], stddev=0.01))
    # depthwise conv
    ds4_conv = tf.nn.depthwise_conv2d(ds3_relu2, ds4conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # batchnormalization
    ds4_conv = batch_normalization_layer(ds4_conv, 'ds4_conv', is_training)
    ds4_relu1 = tf.nn.relu(ds4_conv)
    # pointwise conv
    ds4_out_channels = 172
    ds4_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, ds3_out_channels * ds4conv_filter_count, ds4_out_channels], stddev=0.01))
    ds4_pointwise_conv = tf.nn.conv2d(ds4_relu1, ds4_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds4_pointwise_conv = batch_normalization_layer(ds4_pointwise_conv, 'ds4_pointwise_conv', is_training)
    ds4_relu2 = tf.nn.relu(ds4_pointwise_conv)

    # average pool
    avg_pool = tf.nn.avg_pool(ds4_relu2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    ######  FC  ########
    avg_pool_shape = avg_pool.get_shape()

    avg_pool_element_count = int(
        avg_pool_shape[2] * avg_pool_shape[1] * avg_pool_shape[3])

    flattened_avg_pool = tf.reshape(avg_pool, [-1, avg_pool_element_count])
    label_count = model_settings['label_count']
    fc_weights = tf.Variable(
        tf.truncated_normal(
            [avg_pool_element_count, label_count], stddev=0.01))
    fc_bias = tf.Variable(tf.zeros([label_count]))
    fc = tf.add(tf.matmul(flattened_avg_pool, fc_weights), fc_bias)
    weights_num = first_filter_count * ds1conv_filter_count * ds1_out_channels + ds1_out_channels * ds2conv_filter_count * ds2_out_channels \
                  + ds2_out_channels * ds3conv_filter_count * ds3_out_channels + ds3_out_channels * ds4conv_filter_count * ds4_out_channels \
                  + avg_pool_element_count * label_count
    print("weights num:" + str(weights_num))
    if is_training:
        return fc, dropout_prob
    else:
        return fc
def l_ds_bnn_conv_model(fingerprint_input, model_settings, is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    #######    conv1     ########
    first_filter_width = 10
    first_filter_height = 4
    first_filter_count = 64
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_weights = binarize(first_weights)
    print(first_weights.get_shape())
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 2, 2, 1], 'SAME')
    first_conv = batch_normalization_layer(first_conv, 'first_conv', is_training)  # batchnormalization
    first_relu = tf.clip_by_value(first_conv,-1,1)
    max_pool = tf.nn.max_pool(first_relu, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
    #######    dsconv1   ########
    ds1conv_filter_width = 3
    ds1conv_filter_height = 3
    ds1conv_filter_count = 10
    ds1conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds1conv_filter_height, ds1conv_filter_width, first_filter_count, ds1conv_filter_count], stddev=0.01))
    ds1conv_weights = binarize(ds1conv_weights)
    # depthwise conv
    ds1_conv = tf.nn.depthwise_conv2d(max_pool, ds1conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # batchnormalization
    ds1_conv = batch_normalization_layer(ds1_conv, 'ds1_conv', is_training)
    ds1_relu1 = tf.clip_by_value(ds1_conv,-1,1)
    # pointwise conv
    ds1_out_channels = 64
    ds1_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, first_filter_count * ds1conv_filter_count, ds1_out_channels], stddev=0.01))
    ds1_pointwise_weights = binarize(ds1_pointwise_weights)
    ds1_pointwise_conv = tf.nn.conv2d(ds1_relu1, ds1_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds1_pointwise_conv = batch_normalization_layer(ds1_pointwise_conv, 'ds1_pointwise_conv', is_training)
    ds1_relu2 = tf.clip_by_value(ds1_pointwise_conv,-1,1)

    #######    dsconv2   ########
    ds2conv_filter_width = 3
    ds2conv_filter_height = 3
    ds2conv_filter_count = 10
    ds2conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds2conv_filter_height, ds2conv_filter_width, ds1_out_channels, ds2conv_filter_count], stddev=0.01))
    ds2conv_weights = binarize(ds2conv_weights)
    # depthwise conv
    ds2_conv = tf.nn.depthwise_conv2d(ds1_relu2, ds2conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # batchnormalization
    ds2_conv = batch_normalization_layer(ds2_conv, 'ds2_conv', is_training)
    ds2_relu1 = tf.clip_by_value(ds2_conv,-1,1)
    # pointwise conv
    ds2_out_channels = 64
    ds2_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, ds1_out_channels * ds2conv_filter_count, ds2_out_channels], stddev=0.01))
    ds2_pointwise_weights = binarize(ds2_pointwise_weights)
    ds2_pointwise_conv = tf.nn.conv2d(ds2_relu1, ds2_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds2_pointwise_conv = batch_normalization_layer(ds2_pointwise_conv, 'ds2_pointwise_conv', is_training)
    ds2_relu2 = tf.clip_by_value(ds2_pointwise_conv,-1,1)

    #######    dsconv3   ########
    ds3conv_filter_width = 3
    ds3conv_filter_height = 3
    ds3conv_filter_count = 10
    ds3conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds3conv_filter_height, ds3conv_filter_width, ds2_out_channels, ds3conv_filter_count], stddev=0.01))
    ds3conv_weights = binarize(ds3conv_weights)
    # depthwise conv
    ds3_conv = tf.nn.depthwise_conv2d(ds2_relu2, ds3conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # batchnormalization
    ds3_conv = batch_normalization_layer(ds3_conv, 'ds3_conv', is_training)
    ds3_relu1 = tf.clip_by_value(ds3_conv,-1,1)
    # pointwise conv
    ds3_out_channels = 64
    ds3_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, ds2_out_channels * ds3conv_filter_count, ds3_out_channels], stddev=0.01))
    ds3_pointwise_weights = binarize(ds3_pointwise_weights)
    ds3_pointwise_conv = tf.nn.conv2d(ds3_relu1, ds3_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds3_pointwise_conv = batch_normalization_layer(ds3_pointwise_conv, 'ds3_pointwise_conv', is_training)
    ds3_relu2 = tf.clip_by_value(ds3_pointwise_conv,-1,1)

    #######    dsconv4   ########
    ds4conv_filter_width = 3
    ds4conv_filter_height = 3
    ds4conv_filter_count = 10
    ds4conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds4conv_filter_height, ds4conv_filter_width, ds3_out_channels, ds4conv_filter_count], stddev=0.01))
    ds4conv_weights = binarize(ds4conv_weights)
    # depthwise conv
    ds4_conv = tf.nn.depthwise_conv2d(ds3_relu2, ds4conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # batchnormalization
    ds4_conv = batch_normalization_layer(ds4_conv, 'ds4_conv', is_training)
    ds4_relu1 = tf.clip_by_value(ds4_conv,-1,1)
    # pointwise conv
    ds4_out_channels = 64
    ds4_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, ds3_out_channels * ds4conv_filter_count, ds4_out_channels], stddev=0.01))
    ds4_pointwise_weights = binarize(ds4_pointwise_weights)
    ds4_pointwise_conv = tf.nn.conv2d(ds4_relu1, ds4_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds4_pointwise_conv = batch_normalization_layer(ds4_pointwise_conv, 'ds4_pointwise_conv', is_training)
    ds4_relu2 = tf.clip_by_value(ds4_pointwise_conv,-1,1)

    # average pool
    avg_pool = tf.nn.avg_pool(ds4_relu2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    ######  FC  ########
    avg_pool_shape = avg_pool.get_shape()

    avg_pool_element_count = int(
        avg_pool_shape[2] * avg_pool_shape[1] * avg_pool_shape[3])

    flattened_avg_pool = tf.reshape(avg_pool, [-1, avg_pool_element_count])
    label_count = model_settings['label_count']
    fc_weights = tf.Variable(
        tf.truncated_normal(
            [avg_pool_element_count, label_count], stddev=0.01))
    fc_weights = binarize(fc_weights)
    fc_bias = tf.Variable(tf.zeros([label_count]))
    fc_bias = binarize(fc_bias)
    fc = tf.add(tf.matmul(flattened_avg_pool, fc_weights), fc_bias)
    #fc = tf.clip_by_value(fc,-1,1)
    weights_num = first_filter_count * ds1conv_filter_count * ds1_out_channels + ds1_out_channels * ds2conv_filter_count * ds2_out_channels \
                  + ds2_out_channels * ds3conv_filter_count * ds3_out_channels + ds3_out_channels * ds4conv_filter_count * ds4_out_channels \
                  + avg_pool_element_count * label_count
    print("weights num:" + str(weights_num))
    if is_training:
        return fc, dropout_prob
    else:
        return fc
def l_ds_bnn_conv_2_model(fingerprint_input, model_settings, is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    #######    conv1     ########
    first_filter_width = 10
    first_filter_height = 4
    first_filter_count = 172
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_weights = binarize(first_weights)
    print(first_weights.get_shape())
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 2, 1, 1], 'SAME')
    first_conv = batch_normalization_layer(first_conv, 'first_conv', is_training)  # batchnormalization
    first_relu = tf.clip_by_value(first_conv,-1,1)
    max_pool = tf.nn.max_pool(first_relu, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
    #######    dsconv1   ########
    ds1conv_filter_width = 3
    ds1conv_filter_height = 3
    ds1conv_filter_count = 10
    ds1conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds1conv_filter_height, ds1conv_filter_width, first_filter_count, ds1conv_filter_count], stddev=0.01))
    ds1conv_weights = binarize(ds1conv_weights)
    # depthwise conv
    ds1_conv = tf.nn.depthwise_conv2d(max_pool, ds1conv_weights, strides=[1, 2, 2, 1], padding='SAME')
    # batchnormalization
    ds1_conv = batch_normalization_layer(ds1_conv, 'ds1_conv', is_training)
    ds1_relu1 = tf.clip_by_value(ds1_conv,-1,1)
    # pointwise conv
    ds1_out_channels = 172
    ds1_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, first_filter_count * ds1conv_filter_count, ds1_out_channels], stddev=0.01))
    ds1_pointwise_weights = binarize(ds1_pointwise_weights)
    ds1_pointwise_conv = tf.nn.conv2d(ds1_relu1, ds1_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds1_pointwise_conv = batch_normalization_layer(ds1_pointwise_conv, 'ds1_pointwise_conv', is_training)
    ds1_relu2 = tf.clip_by_value(ds1_pointwise_conv,-1,1)

    #######    dsconv2   ########
    ds2conv_filter_width = 3
    ds2conv_filter_height = 3
    ds2conv_filter_count = 10
    ds2conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds2conv_filter_height, ds2conv_filter_width, ds1_out_channels, ds2conv_filter_count], stddev=0.01))
    ds2conv_weights = binarize(ds2conv_weights)
    # depthwise conv
    ds2_conv = tf.nn.depthwise_conv2d(ds1_relu2, ds2conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # batchnormalization
    ds2_conv = batch_normalization_layer(ds2_conv, 'ds2_conv', is_training)
    ds2_relu1 = tf.clip_by_value(ds2_conv,-1,1)
    # pointwise conv
    ds2_out_channels = 172
    ds2_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, ds1_out_channels * ds2conv_filter_count, ds2_out_channels], stddev=0.01))
    ds2_pointwise_weights = binarize(ds2_pointwise_weights)
    ds2_pointwise_conv = tf.nn.conv2d(ds2_relu1, ds2_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds2_pointwise_conv = batch_normalization_layer(ds2_pointwise_conv, 'ds2_pointwise_conv', is_training)
    ds2_relu2 = tf.clip_by_value(ds2_pointwise_conv,-1,1)

    #######    dsconv3   ########
    ds3conv_filter_width = 3
    ds3conv_filter_height = 3
    ds3conv_filter_count = 10
    ds3conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds3conv_filter_height, ds3conv_filter_width, ds2_out_channels, ds3conv_filter_count], stddev=0.01))
    ds3conv_weights = binarize(ds3conv_weights)
    # depthwise conv
    ds3_conv = tf.nn.depthwise_conv2d(ds2_relu2, ds3conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # batchnormalization
    ds3_conv = batch_normalization_layer(ds3_conv, 'ds3_conv', is_training)
    ds3_relu1 = tf.clip_by_value(ds3_conv,-1,1)
    # pointwise conv
    ds3_out_channels = 172
    ds3_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, ds2_out_channels * ds3conv_filter_count, ds3_out_channels], stddev=0.01))
    ds3_pointwise_weights = binarize(ds3_pointwise_weights)
    ds3_pointwise_conv = tf.nn.conv2d(ds3_relu1, ds3_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds3_pointwise_conv = batch_normalization_layer(ds3_pointwise_conv, 'ds3_pointwise_conv', is_training)
    ds3_relu2 = tf.clip_by_value(ds3_pointwise_conv,-1,1)

    #######    dsconv4   ########
    ds4conv_filter_width = 3
    ds4conv_filter_height = 3
    ds4conv_filter_count = 10
    ds4conv_weights = tf.Variable(
        tf.truncated_normal(
            [ds4conv_filter_height, ds4conv_filter_width, ds3_out_channels, ds4conv_filter_count], stddev=0.01))
    ds4conv_weights = binarize(ds4conv_weights)
    # depthwise conv
    ds4_conv = tf.nn.depthwise_conv2d(ds3_relu2, ds4conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # batchnormalization
    ds4_conv = batch_normalization_layer(ds4_conv, 'ds4_conv', is_training)
    ds4_relu1 = tf.clip_by_value(ds4_conv,-1,1)
    # pointwise conv
    ds4_out_channels = 172
    ds4_pointwise_weights = tf.Variable(
        tf.truncated_normal(
            [1, 1, ds3_out_channels * ds4conv_filter_count, ds4_out_channels], stddev=0.01))
    ds4_pointwise_weights = binarize(ds4_pointwise_weights)
    ds4_pointwise_conv = tf.nn.conv2d(ds4_relu1, ds4_pointwise_weights, [1, 1, 1, 1], 'SAME')
    # batchnormalization
    ds4_pointwise_conv = batch_normalization_layer(ds4_pointwise_conv, 'ds4_pointwise_conv', is_training)
    ds4_relu2 = tf.clip_by_value(ds4_pointwise_conv,-1,1)

    # average pool
    avg_pool = tf.nn.avg_pool(ds4_relu2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    ######  FC  ########
    avg_pool_shape = avg_pool.get_shape()

    avg_pool_element_count = int(
        avg_pool_shape[2] * avg_pool_shape[1] * avg_pool_shape[3])

    flattened_avg_pool = tf.reshape(avg_pool, [-1, avg_pool_element_count])
    label_count = model_settings['label_count']
    fc_weights = tf.Variable(
        tf.truncated_normal(
            [avg_pool_element_count, label_count], stddev=0.01))
    fc_weights = binarize(fc_weights)
    fc_bias = tf.Variable(tf.zeros([label_count]))
    fc_bias = binarize(fc_bias)
    fc = tf.add(tf.matmul(flattened_avg_pool, fc_weights), fc_bias)
    #fc = tf.clip_by_value(fc,-1,1)
    weights_num = first_filter_count * ds1conv_filter_count * ds1_out_channels + ds1_out_channels * ds2conv_filter_count * ds2_out_channels \
                  + ds2_out_channels * ds3conv_filter_count * ds3_out_channels + ds3_out_channels * ds4conv_filter_count * ds4_out_channels \
                  + avg_pool_element_count * label_count
    print("weights num:" + str(weights_num))
    if is_training:
        return fc, dropout_prob
    else:
        return fc


def s_create_conv_model_b_1(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.

  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
   dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 5
  first_filter_height = 5
  first_filter_count = 32
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)

  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  
  max_pool1 = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  
  second_filter_width = 5
  second_filter_height = 5
  second_filter_count = 32
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool1, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu

  max_pool2 = tf.nn.max_pool(second_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  
  third_filter_width = 3
  third_filter_height = 3
  third_filter_count = 32
  third_weights = tf.Variable(
      tf.truncated_normal(
          [
              third_filter_height, third_filter_width, second_filter_count,
              third_filter_count
          ],
          stddev=0.01))
  third_bias = tf.Variable(tf.zeros([third_filter_count]))
  third_conv = tf.nn.conv2d(max_pool2, third_weights, [1, 1, 1, 1],
                             'SAME') + third_bias
  third_relu = tf.nn.relu(third_conv)
  if is_training:
    third_dropout = tf.nn.dropout(third_relu, dropout_prob)
  else:
    third_dropout = third_relu


  third_conv_shape = third_dropout.get_shape()
  third_conv_output_width = third_conv_shape[2]
  third_conv_output_height = third_conv_shape[1]
  third_conv_element_count = int(
      third_conv_output_width * third_conv_output_height *
      third_filter_count)
  flattened_third_conv = tf.reshape(third_dropout,
                                     [-1, third_conv_element_count])



  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [third_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_third_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc

def create_low_latency_conv_model(fingerprint_input, model_settings,
                                  is_training):
  """Builds a convolutional model with low compute requirements.

  This is roughly the network labeled as 'cnn-one-fstride4' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces slightly lower quality results than the 'conv' model, but needs
  fewer weight parameters and computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = input_time_size
  first_filter_count = 186
  first_filter_stride_x = 1
  first_filter_stride_y = 1
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu


  first_conv_output_width = math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x)
  first_conv_output_height = math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y)
  first_conv_element_count = int(
      first_conv_output_width * first_conv_output_height * first_filter_count)
  flattened_first_conv = tf.reshape(first_dropout,
                                    [-1, first_conv_element_count])


  first_fc_output_channels = 128
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_conv_element_count, first_fc_output_channels], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc


  second_fc_output_channels = 128
  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc


  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_fc_output_channels, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc
def create_low_latency_svdf_model(fingerprint_input, model_settings,
                                  is_training, runtime_settings):
  """Builds an SVDF model with low compute requirements.

  This is based in the topology presented in the 'Compressing Deep Neural
  Networks using a Rank-Constrained Topology' paper:
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
        [SVDF]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This model produces lower recognition accuracy than the 'conv' model above,
  but requires fewer weight parameters and, significantly fewer computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    The node is expected to produce a 2D Tensor of shape:
      [batch, model_settings['dct_coefficient_count'] *
              model_settings['spectrogram_length']]
    with the features corresponding to the same time slot arranged contiguously,
    and the oldest slot at index [:, 0], and newest at [:, -1].
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
      ValueError: If the inputs tensor is incorrectly shaped.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']

  # Validation.
  input_shape = fingerprint_input.get_shape()
  if len(input_shape) != 2:
    raise ValueError('Inputs to `SVDF` should have rank == 2.')
  if input_shape[-1].value is None:
    raise ValueError('The last dimension of the inputs to `SVDF` '
                     'should be defined. Found `None`.')
  if input_shape[-1].value % input_frequency_size != 0:
    raise ValueError('Inputs feature dimension %d must be a multiple of '
                     'frame size %d', fingerprint_input.shape[-1].value,
                     input_frequency_size)

  # Set number of units (i.e. nodes) and rank.
  rank = 2
  num_units = 1280
  # Number of filters: pairs of feature and time filters.
  num_filters = rank * num_units
  # Create the runtime memory: [num_filters, batch, input_time_size]
  batch = 1
  memory = tf.Variable(tf.zeros([num_filters, batch, input_time_size]),
                       trainable=False, name='runtime-memory')
  # Determine the number of new frames in the input, such that we only operate
  # on those. For training we do not use the memory, and thus use all frames
  # provided in the input.
  # new_fingerprint_input: [batch, num_new_frames*input_frequency_size]
  if is_training:
    num_new_frames = input_time_size
  else:
    window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                           model_settings['sample_rate'])
    num_new_frames = tf.cond(
        tf.equal(tf.count_nonzero(memory), 0),
        lambda: input_time_size,
        lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))
  new_fingerprint_input = fingerprint_input[
      :, -num_new_frames*input_frequency_size:]
  # Expand to add input channels dimension.
  new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

  # Create the frequency filters.
  weights_frequency = tf.Variable(
      tf.truncated_normal([input_frequency_size, num_filters], stddev=0.01))
  # Expand to add input channels dimensions.
  # weights_frequency: [input_frequency_size, 1, num_filters]
  weights_frequency = tf.expand_dims(weights_frequency, 1)
  # Convolve the 1D feature filters sliding over the time dimension.
  # activations_time: [batch, num_new_frames, num_filters]
  activations_time = tf.nn.conv1d(
      new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
  # Rearrange such that we can perform the batched matmul.
  # activations_time: [num_filters, batch, num_new_frames]
  activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

  # Runtime memory optimization.
  if not is_training:
    # We need to drop the activations corresponding to the oldest frames, and
    # then add those corresponding to the new frames.
    new_memory = memory[:, :, num_new_frames:]
    new_memory = tf.concat([new_memory, activations_time], 2)
    tf.assign(memory, new_memory)
    activations_time = new_memory

  # Create the time filters.
  weights_time = tf.Variable(
      tf.truncated_normal([num_filters, input_time_size], stddev=0.01))
  # Apply the time filter on the outputs of the feature filters.
  # weights_time: [num_filters, input_time_size, 1]
  # outputs: [num_filters, batch, 1]
  weights_time = tf.expand_dims(weights_time, 2)
  outputs = tf.matmul(activations_time, weights_time)
  # Split num_units and rank into separate dimensions (the remaining
  # dimension is the input_shape[0] -i.e. batch size). This also squeezes
  # the last dimension, since it's not used.
  # [num_filters, batch, 1] => [num_units, rank, batch]
  outputs = tf.reshape(outputs, [num_units, rank, -1])
  # Sum the rank outputs per unit => [num_units, batch].
  units_output = tf.reduce_sum(outputs, axis=1)
  # Transpose to shape [batch, num_units]
  units_output = tf.transpose(units_output)

  # Appy bias.
  bias = tf.Variable(tf.zeros([num_units]))
  first_bias = tf.nn.bias_add(units_output, bias)

  # Relu.
  first_relu = tf.nn.relu(first_bias)

  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu

  first_fc_output_channels = 256
  first_fc_weights = tf.Variable(
      tf.truncated_normal([num_units, first_fc_output_channels], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 256
  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_fc_output_channels, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc
def binarize_3(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()  # 获取当前默认计算图

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}): #覆盖当前图中的梯度算法
            x=tf.clip_by_value(x,-1,1)                      #限制x范围在-1和1之间
            return tf.sign(x)

def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()  # 获取当前默认计算图

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}): #覆盖当前图中的梯度算法
            x=tf.clip_by_value(x,-1,1)                      #限制x范围在-1和1之间
            return tf.sign(x)

def binarize_u_sigma(x,u,sigma):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()  # 获取当前默认计算图

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}): #覆盖当前图中的梯度算法
            x=tf.clip_by_value(x,u-1*sigma,u+1*sigma)                      #限制x范围在u-3sigma和u+3sigma之间
            return u+1*sigma*tf.sign(x-u)
def binarize_u_3sigma(x,u,sigma):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()  # 获取当前默认计算图

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}): #覆盖当前图中的梯度算法
            x=tf.clip_by_value(x,u-3*sigma,u+3*sigma)                      #限制x范围在u-3sigma和u+3sigma之间
            return u+3*sigma*tf.sign(x-u)

def create_2017Google_model(fingerprint_input, model_settings, is_training):
      dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
      fingerprint_size = model_settings['fingerprint_size']
      label_count = model_settings['label_count']

      #hidden_layer1 = 128
      #hidden_layer2 = 128
      #hidden_layer3 = 128
      hidden_layer1 = 512
      hidden_layer2 = 512
      hidden_layer3 = 512
      hidden_layer4 = 512
      hidden_layer5 = 512

      layer1_weights = tf.Variable(
        tf.truncated_normal([fingerprint_size, hidden_layer1], stddev=0.01),dtype=tf.float32,name="L1_weights")
      layer1_bias = tf.Variable(tf.random_normal([hidden_layer1]),dtype=tf.float32,name="L1_bias")

      layer2_weights = tf.Variable(
        tf.truncated_normal([hidden_layer1, hidden_layer2], stddev=0.02),dtype=tf.float32,name="L2_weights")
      layer2_bias = tf.Variable(tf.random_normal([hidden_layer2]),dtype=tf.float32,name="L2_bias")

      layer3_weights = tf.Variable(
        tf.truncated_normal([hidden_layer2, hidden_layer3], stddev=0.03),dtype=tf.float32,name="L3_weights")
      layer3_bias = tf.Variable(tf.random_normal([hidden_layer3]),dtype=tf.float32,name="L3_bias")

      layer4_weights = tf.Variable(
        tf.truncated_normal([hidden_layer3, hidden_layer4], stddev=0.04),dtype=tf.float32,name="L4_weights")
      layer4_bias = tf.Variable(tf.random_normal([hidden_layer4]),dtype=tf.float32,name="L4_bias")

      layer5_weights = tf.Variable(
        tf.truncated_normal([hidden_layer4, hidden_layer5], stddev=0.05),dtype=tf.float32,name="L5_weights")
      layer5_bias = tf.Variable(tf.random_normal([hidden_layer5]),dtype=tf.float32,name="L5_bias")

      output_weights = tf.Variable(
        tf.truncated_normal([hidden_layer5, label_count], stddev=0.04),dtype=tf.float32,name="Lout_weights")
      output_bias = tf.Variable(tf.random_normal([label_count]),dtype=tf.float32,name="Lout_bias")

      a1 = tf.nn.relu(tf.add(tf.matmul(fingerprint_input, layer1_weights),layer1_bias))
      a2 = tf.nn.relu(tf.add(tf.matmul(a1, layer2_weights) , layer2_bias))
      a3 = tf.nn.relu(tf.add(tf.matmul(a2, layer3_weights) , layer3_bias))
      a4 = tf.nn.relu(tf.add(tf.matmul(a3, layer4_weights) , layer4_bias))
      a5 = tf.nn.relu(tf.add(tf.matmul(a4, layer5_weights) , layer5_bias))
      out = tf.add(tf.matmul(a5, output_weights) , output_bias)
      #loss_norm = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(
      #    layer3_weights) +  tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer5_weights) + tf.nn.l2_loss(output_weights)
      if is_training:
          return out, dropout_prob
      else:
          return out

def dnn_dp(fingerprint_input, model_settings, is_training):
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']

    hidden_layer1 = 128
    hidden_layer2 = 128
    hidden_layer3 = 128
    layer1_weights = tf.Variable(
        tf.truncated_normal([fingerprint_size, hidden_layer1], stddev=0.01), dtype=tf.float32, name="L1_weights")
    layer1_bias = tf.Variable(tf.random_normal([hidden_layer1]), dtype=tf.float32, name="L1_bias")

    layer2_weights = tf.Variable(
        tf.truncated_normal([hidden_layer1, hidden_layer2], stddev=0.02), dtype=tf.float32, name="L2_weights")
    layer2_bias = tf.Variable(tf.random_normal([hidden_layer2]), dtype=tf.float32, name="L2_bias")

    layer3_weights = tf.Variable(
        tf.truncated_normal([hidden_layer2, hidden_layer3], stddev=0.03), dtype=tf.float32, name="L3_weights")
    layer3_bias = tf.Variable(tf.random_normal([hidden_layer3]), dtype=tf.float32, name="L3_bias")

    output_weights = tf.Variable(
        tf.truncated_normal([hidden_layer3, label_count], stddev=0.04), dtype=tf.float32, name="Lout_weights")
    output_bias = tf.Variable(tf.random_normal([label_count]), dtype=tf.float32, name="Lout_bias")

    a1 = tf.nn.relu(tf.add(tf.matmul(fingerprint_input, layer1_weights), layer1_bias))
    if is_training:
        a1 = tf.nn.dropout(a1, dropout_prob)

    a2 = tf.nn.relu(tf.add(tf.matmul(a1, layer2_weights), layer2_bias))
    if is_training:
        a2 = tf.nn.dropout(a2, dropout_prob)

    a3 = tf.nn.relu(tf.add(tf.matmul(a2, layer3_weights), layer3_bias))
    if is_training:
        a3 = tf.nn.dropout(a3, dropout_prob)
    out = tf.add(tf.matmul(a3, output_weights), output_bias)
    loss_norm = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(
        layer3_weights) + tf.nn.l2_loss(output_weights)
    if is_training:
        return out, dropout_prob#, loss_norm
    else:
        return out, dropout_prob#, loss_norm
def dnn_4n_dp(fingerprint_input, model_settings, is_training):
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']

    hidden_layer1 = 128
    hidden_layer2 = 128
    hidden_layer3 = 128
    hidden_layer4 = 128

    layer1_weights = tf.Variable(
        tf.truncated_normal([fingerprint_size, hidden_layer1], stddev=0.01), dtype=tf.float32, name="L1_weights")
    layer1_bias = tf.Variable(tf.random_normal([hidden_layer1]), dtype=tf.float32, name="L1_bias")

    layer2_weights = tf.Variable(
        tf.truncated_normal([hidden_layer1, hidden_layer2], stddev=0.02), dtype=tf.float32, name="L2_weights")
    layer2_bias = tf.Variable(tf.random_normal([hidden_layer2]), dtype=tf.float32, name="L2_bias")

    layer3_weights = tf.Variable(
        tf.truncated_normal([hidden_layer2, hidden_layer3], stddev=0.03), dtype=tf.float32, name="L3_weights")
    layer3_bias = tf.Variable(tf.random_normal([hidden_layer3]), dtype=tf.float32, name="L3_bias")

    layer4_weights = tf.Variable(
        tf.truncated_normal([hidden_layer3, hidden_layer4], stddev=0.04), dtype=tf.float32, name="L4_weights")
    layer4_bias = tf.Variable(tf.random_normal([hidden_layer4]), dtype=tf.float32, name="L4_bias")


    output_weights = tf.Variable(
        tf.truncated_normal([hidden_layer4, label_count], stddev=0.05), dtype=tf.float32, name="Lout_weights")
    output_bias = tf.Variable(tf.random_normal([label_count]), dtype=tf.float32, name="Lout_bias")

    a1 = tf.nn.relu(tf.add(tf.matmul(fingerprint_input, layer1_weights), layer1_bias))
    if is_training:
        a1 = tf.nn.dropout(a1, dropout_prob)

    a2 = tf.nn.relu(tf.add(tf.matmul(a1, layer2_weights), layer2_bias))
    if is_training:
        a2 = tf.nn.dropout(a2, dropout_prob)

    a3 = tf.nn.relu(tf.add(tf.matmul(a2, layer3_weights), layer3_bias))
    if is_training:
        a3 = tf.nn.dropout(a3, dropout_prob)

    a4 = tf.nn.relu(tf.add(tf.matmul(a3, layer4_weights), layer4_bias))
    if is_training:
        a4 = tf.nn.dropout(a4, dropout_prob)

    out = tf.add(tf.matmul(a4, output_weights), output_bias)
    loss_norm = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(
        layer3_weights) + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(output_weights)

    if is_training:
        return out, dropout_prob#, loss_norm
    else:
        return out, dropout_prob#, loss_norm

def dnn_bnn_retrain(fingerprint_input, model_settings, is_training):
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']

    hidden_layer1 = 128
    hidden_layer2 = 128
    hidden_layer3 = 128
    reader = tf.train.NewCheckpointReader("/home/zhangs/tensorflow-master/tensorflow/examples/speech/tmp/dnn128_650ms_50_20/dnn.ckpt-16000")
    #all_variables = reader.get_variable_to_shape_map()
    w1 = reader.get_tensor("L1_weights")
    w2 = reader.get_tensor("L2_weights")
    w3 = reader.get_tensor("L3_weights")
    w4 = reader.get_tensor("Lout_weights")
    w1_variance = np.sqrt(np.var(w1))
    w2_variance = np.sqrt(np.var(w2))
    w3_variance = np.sqrt(np.var(w3))
    w4_variance = np.sqrt(np.var(w4))
    w1_mean = np.mean(w1)
    w2_mean = np.mean(w2)
    w3_mean = np.mean(w3)
    w4_mean = np.mean(w4)
    # w1_variance = 1
    # w2_variance = 1/3
    # w3_variance = 0.5/3
    # w4_variance = 1/3
    # w1_mean = 0
    # w2_mean = 0
    # w3_mean = 0
    # w4_mean = 0
    layer1_weights = tf.Variable(
        w1, name="L1_weights")

    layer2_weights = tf.Variable(
        w2, name="L2_weights")

    layer3_weights = tf.Variable(
        w3, name="L3_weights")

    output_weights = tf.Variable(
        w4, name="Lout_weights")

    #bin_w1 = binarize(layer1_weights)
    bin_w1 = binarize_u_3sigma(layer1_weights, w1_mean, w1_variance)
    z1 = tf.matmul(fingerprint_input, bin_w1)
    z1_hat = Batchnormalize('hidden1', z1, hidden_layer1, is_training)
    a1 = tf.clip_by_value(z1_hat, -1, 1)

    #bin_w2 = binarize(layer2_weights)
    bin_w2 = binarize_u_3sigma(layer2_weights,w2_mean,w2_variance)
    z2 = tf.matmul(a1, bin_w2)
    z2_hat = Batchnormalize('hidden2', z2, hidden_layer2, is_training)
    # a2 = binarize(z2_hat)
    a2 = tf.clip_by_value(z2_hat, -1, 1)
    bin_w3 = binarize_u_3sigma(layer3_weights, w3_mean, w3_variance)
    #bin_w3 = binarize(layer3_weights)
    z3 = tf.matmul(a2, bin_w3)
    z3_hat = Batchnormalize('hidden3', z3, hidden_layer3, is_training)
    # a3 = binarize(z3_hat)
    a3 = tf.clip_by_value(z3_hat, -1, 1)
    bin_w4 = binarize_u_3sigma(output_weights, w4_mean, w4_variance)
    #bin_w4 = binarize(output_weights)
    z4 = tf.matmul(a3, bin_w4)
    out = Batchnormalize('out', z4, label_count, is_training)
    # out = binarize(out)
    out = tf.clip_by_value(out, -1, 1)
    loss_norm = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(
        layer3_weights) + tf.nn.l2_loss(output_weights)

    if is_training:
        return out, dropout_prob, loss_norm
    else:
        return out

def dnn_bn_all_bnn_4n(fingerprint_input, model_settings, is_training):
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        fingerprint_size = model_settings['fingerprint_size']
        label_count = model_settings['label_count']

        hidden_layer1 = 128
        hidden_layer2 = 512
        hidden_layer3 = 256
        hidden_layer4 = 256

        layer1_weights = tf.Variable(
            tf.truncated_normal([fingerprint_size, hidden_layer1], stddev=0.01), dtype=tf.float32, name="L1_weights")

        layer2_weights = tf.Variable(
            tf.truncated_normal([hidden_layer1, hidden_layer2], stddev=0.02), dtype=tf.float32, name="L2_weights")

        layer3_weights = tf.Variable(
            tf.truncated_normal([hidden_layer2, hidden_layer3], stddev=0.03), dtype=tf.float32, name="L3_weights")

        layer4_weights = tf.Variable(
            tf.truncated_normal([hidden_layer3, hidden_layer4], stddev=0.03), dtype=tf.float32, name="L4_weights")

        output_weights = tf.Variable(
            tf.truncated_normal([hidden_layer4, label_count], stddev=0.04), dtype=tf.float32, name="Lout_weights")

        bin_w1 = binarize(layer1_weights)
        z1 = tf.matmul(fingerprint_input, bin_w1)
        z1_hat = Batchnormalize('hidden1', z1, hidden_layer1, is_training)
        a1 = binarize(z1_hat)

        bin_w2 = binarize(layer2_weights)
        z2 = tf.matmul(a1, bin_w2)
        z2_hat = Batchnormalize('hidden2', z2, hidden_layer2, is_training)
        a2 = binarize(z2_hat)

        bin_w3 = binarize(layer3_weights)
        z3 = tf.matmul(a2, bin_w3)
        z3_hat = Batchnormalize('hidden3', z3, hidden_layer3, is_training)
        a3 = binarize(z3_hat)

        bin_w4 = binarize(layer4_weights)
        z4 = tf.matmul(a3, bin_w4)
        z4_hat = Batchnormalize('hidden4', z4, hidden_layer4, is_training)
        a4 = binarize(z4_hat)

        bin_w5 = binarize(output_weights)
        z5 = tf.matmul(a4, bin_w5)
        out = Batchnormalize('out', z5, label_count, is_training)
        out = tf.clip_by_value(out,-1,1)
        loss_norm = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(
            layer3_weights) + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(output_weights)

        if is_training:
            return out, dropout_prob, loss_norm
        else:
            return out, dropout_prob, loss_norm
def dnn_bn_all_bnn_5n(fingerprint_input, model_settings, is_training):
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']

    hidden_layer1 = 128
    hidden_layer2 = 256
    hidden_layer3 = 512
    hidden_layer4 = 256
    hidden_layer5 = 128

    layer1_weights = tf.Variable(
        tf.truncated_normal([fingerprint_size, hidden_layer1], stddev=0.01), dtype=tf.float32, name="L1_weights")

    layer2_weights = tf.Variable(
        tf.truncated_normal([hidden_layer1, hidden_layer2], stddev=0.02), dtype=tf.float32, name="L2_weights")

    layer3_weights = tf.Variable(
        tf.truncated_normal([hidden_layer2, hidden_layer3], stddev=0.03), dtype=tf.float32, name="L3_weights")

    layer4_weights = tf.Variable(
        tf.truncated_normal([hidden_layer3, hidden_layer4], stddev=0.04), dtype=tf.float32, name="L4_weights")

    layer5_weights = tf.Variable(
        tf.truncated_normal([hidden_layer4, hidden_layer5], stddev=0.05), dtype=tf.float32, name="L5_weights")

    output_weights = tf.Variable(
        tf.truncated_normal([hidden_layer5, label_count], stddev=0.04), dtype=tf.float32, name="Lout_weights")

    bin_w1 = binarize(layer1_weights)
    z1 = tf.matmul(fingerprint_input, bin_w1)
    z1_hat = Batchnormalize('hidden1', z1, hidden_layer1, is_training)
    a1 = binarize(z1_hat)

    bin_w2 = binarize(layer2_weights)
    z2 = tf.matmul(a1, bin_w2)
    z2_hat = Batchnormalize('hidden2', z2, hidden_layer2, is_training)
    a2 = binarize(z2_hat)

    bin_w3 = binarize(layer3_weights)
    z3 = tf.matmul(a2, bin_w3)
    z3_hat = Batchnormalize('hidden3', z3, hidden_layer3, is_training)
    a3 = binarize(z3_hat)

    bin_w4 = binarize(layer4_weights)
    z4 = tf.matmul(a3, bin_w4)
    z4_hat = Batchnormalize('hidden4', z4, hidden_layer4, is_training)
    a4 = binarize(z4_hat)

    bin_w5 = binarize(layer5_weights)
    z5 = tf.matmul(a4, bin_w5)
    z5_hat = Batchnormalize('hidden5', z5, hidden_layer5, is_training)
    a5 = binarize(z5_hat)

    bin_w6 = binarize(output_weights)
    z6 = tf.matmul(a5, bin_w6)
    out = Batchnormalize('out', z6, label_count, is_training)
    out = tf.clip_by_value(out, -1, 1)
    loss_norm = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(
        layer3_weights) + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer5_weights)+tf.nn.l2_loss(output_weights)

    if is_training:
        return out, dropout_prob, loss_norm
    else:
        return out, dropout_prob, loss_norm

def dnn_bn_bnn_4n(fingerprint_input, model_settings, is_training):
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        fingerprint_size = model_settings['fingerprint_size']
        label_count = model_settings['label_count']

        hidden_layer1 = 256
        hidden_layer2 = 256
        hidden_layer3 = 128
        hidden_layer4 = 128

        layer1_weights = tf.Variable(
            tf.truncated_normal([fingerprint_size, hidden_layer1], stddev=0.01), dtype=tf.float32, name="L1_weights")

        layer2_weights = tf.Variable(
            tf.truncated_normal([hidden_layer1, hidden_layer2], stddev=0.02), dtype=tf.float32, name="L2_weights")

        layer3_weights = tf.Variable(
            tf.truncated_normal([hidden_layer2, hidden_layer3], stddev=0.03), dtype=tf.float32, name="L3_weights")

        layer4_weights = tf.Variable(
            tf.truncated_normal([hidden_layer3, hidden_layer4], stddev=0.03), dtype=tf.float32, name="L4_weights")

        output_weights = tf.Variable(
            tf.truncated_normal([hidden_layer4, label_count], stddev=0.04), dtype=tf.float32, name="Lout_weights")

        bin_w1 = binarize(layer1_weights)
        z1 = tf.matmul(fingerprint_input, bin_w1)
        z1_hat = Batchnormalize('hidden1', z1, hidden_layer1, is_training)
        a1 = binarize(z1_hat)

        bin_w2 = binarize(layer2_weights)
        z2 = tf.matmul(a1, bin_w2)
        z2_hat = Batchnormalize('hidden2', z2, hidden_layer2, is_training)
        a2 = binarize(z2_hat)

        bin_w3 = binarize(layer3_weights)
        z3 = tf.matmul(a2, bin_w3)
        z3_hat = Batchnormalize('hidden3', z3, hidden_layer3, is_training)
        a3 = binarize(z3_hat)

        bin_w4 = binarize(layer4_weights)
        z4 = tf.matmul(a3, bin_w4)
        z4_hat = Batchnormalize('hidden4', z4, hidden_layer4, is_training)
        a4 = binarize(z4_hat)

        bin_w5 = binarize(output_weights)
        z5 = tf.matmul(a4, bin_w5)
        out = Batchnormalize('out', z5, label_count, is_training)
        out = tf.clip_by_value(out,-1,1)
        loss_norm = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(
            layer3_weights) + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(output_weights)

        if is_training:
            return out, dropout_prob, loss_norm
        else:
            return out
def dnn_bn_bnn_5n(fingerprint_input, model_settings, is_training):
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']

    hidden_layer1 = 128
    hidden_layer2 = 256
    hidden_layer3 = 512
    hidden_layer4 = 256
    hidden_layer5 = 128

    layer1_weights = tf.Variable(
        tf.truncated_normal([fingerprint_size, hidden_layer1], stddev=0.01), dtype=tf.float32, name="L1_weights")

    layer2_weights = tf.Variable(
        tf.truncated_normal([hidden_layer1, hidden_layer2], stddev=0.02), dtype=tf.float32, name="L2_weights")

    layer3_weights = tf.Variable(
        tf.truncated_normal([hidden_layer2, hidden_layer3], stddev=0.03), dtype=tf.float32, name="L3_weights")

    layer4_weights = tf.Variable(
        tf.truncated_normal([hidden_layer3, hidden_layer4], stddev=0.04), dtype=tf.float32, name="L4_weights")

    layer5_weights = tf.Variable(
        tf.truncated_normal([hidden_layer4, hidden_layer5], stddev=0.05), dtype=tf.float32, name="L5_weights")

    output_weights = tf.Variable(
        tf.truncated_normal([hidden_layer5, label_count], stddev=0.04), dtype=tf.float32, name="Lout_weights")

    bin_w1 = binarize(layer1_weights)
    z1 = tf.matmul(fingerprint_input, bin_w1)
    z1_hat = Batchnormalize('hidden1', z1, hidden_layer1, is_training)
    a1 = tf.clip_by_value(z1_hat,-1,1)

    bin_w2 = binarize(layer2_weights)
    z2 = tf.matmul(a1, bin_w2)
    z2_hat = Batchnormalize('hidden2', z2, hidden_layer2, is_training)
    a2 =  tf.clip_by_value(z2_hat,-1,1)

    bin_w3 = binarize(layer3_weights)
    z3 = tf.matmul(a2, bin_w3)
    z3_hat = Batchnormalize('hidden3', z3, hidden_layer3, is_training)
    a3 =  tf.clip_by_value(z3_hat,-1,1)

    bin_w4 = binarize(layer4_weights)
    z4 = tf.matmul(a3, bin_w4)
    z4_hat = Batchnormalize('hidden4', z4, hidden_layer4, is_training)
    a4 =  tf.clip_by_value(z4_hat,-1,1)

    bin_w5 = binarize(layer5_weights)
    z5 = tf.matmul(a4, bin_w5)
    z5_hat = Batchnormalize('hidden5', z5, hidden_layer5, is_training)
    a5 =  tf.clip_by_value(z5_hat,-1,1)

    bin_w6 = binarize(output_weights)
    z6 = tf.matmul(a5, bin_w6)
    out = Batchnormalize('out', z6, label_count, is_training)
    out = tf.clip_by_value(out, -1, 1)
    loss_norm = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(
        layer3_weights) + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer5_weights)+tf.nn.l2_loss(output_weights)

    if is_training:
        return out, dropout_prob, loss_norm
    else:
        return out, dropout_prob, loss_norm
def dnn_bn_bnn_3n(fingerprint_input, model_settings, is_training):
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        fingerprint_size = model_settings['fingerprint_size']
        label_count = model_settings['label_count']

        hidden_layer1 = 128
        hidden_layer2 = 128
        hidden_layer3 = 128


        layer1_weights = tf.Variable(
            tf.truncated_normal([fingerprint_size, hidden_layer1], stddev=0.01), dtype=tf.float32, name="L1_weights")

        layer2_weights = tf.Variable(
            tf.truncated_normal([hidden_layer1, hidden_layer2], stddev=0.02), dtype=tf.float32, name="L2_weights")

        layer3_weights = tf.Variable(
            tf.truncated_normal([hidden_layer2, hidden_layer3], stddev=0.03), dtype=tf.float32, name="L3_weights")

        output_weights = tf.Variable(
            tf.truncated_normal([hidden_layer3, label_count], stddev=0.04), dtype=tf.float32, name="Lout_weights")

        bin_w1 = binarize(layer1_weights)
        z1 = tf.matmul(fingerprint_input, bin_w1)
        z1_hat = Batchnormalize('hidden1', z1, hidden_layer1, is_training)
        a1 = tf.clip_by_value(z1_hat,-1,1)

        bin_w2 = binarize(layer2_weights)
        z2 = tf.matmul(a1, bin_w2)
        z2_hat = Batchnormalize('hidden2', z2, hidden_layer2, is_training)
       # a2 = binarize(z2_hat)
        a2 = tf.clip_by_value(z2_hat, -1, 1)

        bin_w3 = binarize(layer3_weights)
        z3 = tf.matmul(a2, bin_w3)
        z3_hat = Batchnormalize('hidden3', z3, hidden_layer3, is_training)
        #a3 = binarize(z3_hat)
        a3 = tf.clip_by_value(z3_hat, -1, 1)

        bin_w4 = binarize(output_weights)
        z4 = tf.matmul(a3, bin_w4)
        out = Batchnormalize('out', z4, label_count, is_training)
        #out = binarize(out)
        out = tf.clip_by_value(out, -1, 1)
        loss_norm = tf.nn.l2_loss(layer1_weights)+tf.nn.l2_loss(layer2_weights)+tf.nn.l2_loss(layer3_weights)+tf.nn.l2_loss(output_weights)

        if is_training:
            return out, dropout_prob,loss_norm
        else:
            return out

def dnn_test(fingerprint_input, model_settings, is_training):
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']

    hidden_layer1 = 128
    hidden_layer2 = 128
    hidden_layer3 = 128
    layer1_weights = tf.Variable(
        tf.truncated_normal([fingerprint_size, hidden_layer1], stddev=0.01), dtype=tf.float32, name="L1_weights")
    layer1_bias = tf.Variable(tf.random_normal([hidden_layer1]), dtype=tf.float32, name="L1_bias")

    layer2_weights = tf.Variable(
        tf.truncated_normal([hidden_layer1, hidden_layer2], stddev=0.02), dtype=tf.float32, name="L2_weights")
    layer2_bias = tf.Variable(tf.random_normal([hidden_layer2]), dtype=tf.float32, name="L2_bias")

    layer3_weights = tf.Variable(
        tf.truncated_normal([hidden_layer2, hidden_layer3], stddev=0.03), dtype=tf.float32, name="L3_weights")
    layer3_bias = tf.Variable(tf.random_normal([hidden_layer3]), dtype=tf.float32, name="L3_bias")

    output_weights = tf.Variable(
        tf.truncated_normal([hidden_layer3, label_count], stddev=0.04), dtype=tf.float32, name="Lout_weights")
    output_bias = tf.Variable(tf.random_normal([label_count]), dtype=tf.float32, name="Lout_bias")

    l1_bin = binarize(layer1_weights)
    l2_bin = binarize(layer2_weights)
    l3_bin = binarize(layer3_weights)
    a1 = tf.nn.relu(tf.add(tf.matmul(fingerprint_input, l1_bin), layer1_bias))
    a2 = tf.nn.relu(tf.add(tf.matmul(a1, l2_bin), layer2_bias))
    a3 = tf.nn.relu(tf.add(tf.matmul(a2, l3_bin), layer3_bias))
    out = tf.add(tf.matmul(a3, output_weights), output_bias)
    loss_norm = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(
        layer3_weights) + tf.nn.l2_loss(output_weights)
    if is_training:
        return out, dropout_prob, loss_norm
    else:
        return out
def Batchnormalize(layer,linear_output,num_out_nodes,is_training):

    # Batch normalization adds additional trainable variables:
    # gamma (for scaling) and beta (for shifting).
    gamma = tf.Variable(tf.ones([num_out_nodes]),name='gamma_'+layer)
    beta = tf.Variable(tf.zeros([num_out_nodes]),name='beta'+layer)

    # These variables will store the mean and variance for this layer over the entire training set,
    # which we assume represents the general population distribution.
    # By setting `trainable=False`, we tell TensorFlow not to modify these variables during
    # back propagation. Instead, we will assign values to these variables ourselves.
    pop_mean = tf.Variable(tf.zeros([num_out_nodes]), trainable=False)
    pop_variance = tf.Variable(tf.ones([num_out_nodes]), trainable=False)

    # Batch normalization requires a small constant epsilon, used to ensure we don't divide by zero.
    # This is the default value TensorFlow uses.
    epsilon = 1e-3


    def batch_norm_training():
        # Calculate the mean and variance for the data coming out of this layer's linear-combination step.
        # The [0] defines an array of axes to calculate over.
        batch_mean, batch_variance = tf.nn.moments(linear_output, [0])

        # Calculate a moving average of the training data's mean and variance while training.
        # These will be used during inference.
        # Decay should be some number less than 1. tf.layers.batch_normalization uses the parameter
        # "momentum" to accomplish this and defaults it to 0.99
        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))

        # The 'tf.control_dependencies' context tells TensorFlow it must calculate 'train_mean'
        # and 'train_variance' before it calculates the 'tf.nn.batch_normalization' layer.
        # This is necessary because the those two operations are not actually in the graph
        # connecting the linear_output and batch_normalization layers,
        # so TensorFlow would otherwise just skip them.
        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(linear_output, batch_mean, batch_variance, beta, gamma, epsilon)

    def batch_norm_inference():
        # During inference, use the our estimated population mean and variance to normalize the layer
        return tf.nn.batch_normalization(linear_output, pop_mean, pop_variance, beta, gamma, epsilon)


    # Use `tf.cond` as a sort of if-check. When self.is_training is True, TensorFlow will execute
    # the operation returned from `batch_norm_training`; otherwise it will execute the graph
    # operation returned from `batch_norm_inference`.
    batch_normalized_output = tf.cond(tf.cast(is_training,tf.bool), batch_norm_training, batch_norm_inference)
    return(batch_normalized_output)


def batch_normalization_layer(inputs, name, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary

    '''
    Here I referred this website:
    https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
    '''

    with tf.variable_scope(name):
        axis = list(range(len(inputs.get_shape()) - 1))

        mean, variance = tf.nn.moments(inputs, axis)

        beta = tf.get_variable('beta', initializer=tf.zeros_initializer, shape=inputs.get_shape()[-1], dtype=tf.float32)
        gamma = tf.get_variable('gamma', initializer=tf.ones_initializer, shape=inputs.get_shape()[-1],
                                dtype=tf.float32)

        moving_mean = tf.get_variable(name='moving_mean', initializer=tf.zeros_initializer,
                                      shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)
        moving_variance = tf.get_variable(name='moving_var', initializer=tf.zeros_initializer,
                                          shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.999,zero_debias=True) ##
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.999,zero_debias=True)##

        if isTrain:
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):  ####
                inputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)
                tf.add_to_collection('mean', update_moving_mean)
                tf.add_to_collection('variance', update_moving_variance)
        else:
            mean = update_moving_mean
            variance = update_moving_variance
            inputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)
        #inputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)

        return inputs
