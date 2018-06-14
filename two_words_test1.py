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
r"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data_filler
import models
from tensorflow.python.platform import gfile
from layer import mixup_data
#from confuse_test import *
FLAGS = None


def input_normalization(train_fingerprints):
    mean = np.mean(train_fingerprints)
    max =  np.max(train_fingerprints)
    train_fingerprints = (train_fingerprints - mean)/ max
    return train_fingerprints

def np_round_and_clip_7bit(x):
    x = np.round(x)
    x = np.clip(x, -64, 63)/64
    return x
def np_round_and_clip_6bit(x):
    x = np.round(x)
    x = np.clip(x, -32, 31)/32
    return x
def np_round_and_clip_5bit(x):
    x = np.round(x)
    x = np.clip(x, -16, 15)/16
    return x
def np_round_and_clip_4bit(x):
    x = np.round(x)
    x = np.clip(x, -8, 7)/8
    return x
def main(_):
  best_acc = 0
  best_step = 0
  best_acc_istrain = 0
  best_step_istrain = 0
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(input_data_filler.prepare_words_list_my(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
  audio_processor = input_data_filler.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))
##############################################
  ############tensorflow modules##########

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  # ############ 模型创建 ##########
  istrain = tf.placeholder(tf.bool, name='istrain')
  logits= models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=istrain)
  ############ 模型创建 ##########
  # logits, dropout_prob= models.create_model(
  #     fingerprint_input,
  #     model_settings,
  #     FLAGS.model_architecture,
  #     is_training=True)
  # Define loss and optimizer

  ############ 真实值 ##########
  ground_truth_input = tf.placeholder(
      tf.float32, [None, label_count], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  ############ 交叉熵计算 ##########
  # with tf.name_scope('cross_entropy'):
  #   cross_entropy_mean = tf.reduce_mean(
  #       tf.nn.softmax_cross_entropy_with_logits(
  #           labels=ground_truth_input, logits=logits)) + beta*loss_norm
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits))
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  ############ 学习率、准确率、混淆矩阵 ##########
  # learning_rate_input    学习率输入（tf.placeholder）
  # train_step             训练过程 （优化器）
  # predicted_indices      预测输出索引
  # expected_indices       实际希望输出索引
  # correct_prediction     正确预测矩阵
  # confusion_matrix       混淆矩阵
  # evaluation_step        正确分类概率（每个阶段）
  # global_step            全局训练阶段
  # increment_global_step  全局训练阶段递增

  learning_rate_input = tf.placeholder(
      tf.float32, [], name='learning_rate_input')
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(
        learning_rate_input).minimize(cross_entropy_mean)
  # with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
  #   learning_rate_input = tf.placeholder(
  #       tf.float32, [], name='learning_rate_input')
  #  # train_step = tf.train.GradientDescentOptimizer(
  #     #  learning_rate_input).minimize(cross_entropy_mean)
  #   with tf.control_dependencies(update_ops):
  #       train_step = tf.train.AdamOptimizer(
  #           learning_rate_input).minimize(cross_entropy_mean)
  predicted_indices = tf.argmax(logits, 1)
  expected_indices = tf.argmax(ground_truth_input, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(
      expected_indices, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  acc = tf.summary.scalar('accuracy', evaluation_step)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)


  saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)# max keep file // moren 5

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  validation_merged_summaries = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'accuracy'),tf.get_collection(tf.GraphKeys.SUMMARIES,'cross_entropy')])
  test_summaries = tf.summary.merge([acc])
  test_summaries_istrain = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'accuracy'),tf.get_collection(tf.GraphKeys.SUMMARIES,'cross_entropy')])

  #test_summaries_istrain = tf.summary.merge([acc])
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  # validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
  test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
  test_istrain_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test_istrain')
  tf.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  tf.logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                       FLAGS.model_architecture + '.pbtxt')

  # Save list of words.
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))
###
  # model1: fc
  # model2: conv :940k个parameter
  # model3:low_latancy_conv:~~model1
  # model4: 750k
  # Training loop.
    #############################################
    ########            主循环              ######
    #############################################
  training_steps_max = np.sum(training_steps_list)
  for training_step in xrange(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    #######       自动切换学习率      #######
    if training_step <12000+1:
        learning_rate_value = learning_rates_list[0]*0.02**(training_step/12000)
    else:
        learning_rate_value = learning_rates_list[0]*0.02    #0.015 12000
    training_steps_sum = 0
    # for i in range(len(training_steps_list)):
    #   training_steps_sum += training_steps_list[i]
    #   if training_step <= training_steps_sum:
    #     learning_rate_value = learning_rates_list[i]
    #     break

    # Pull the audio samples we'll use for training.
    #######       audio处理器导入数据      ##################################
    ##get_data(self, how_many, offset, model_settings, background_frequency,
    ##         background_volume_range, time_shift, mode, sess)
    ########################################################################
    train_fingerprints, train_ground_truth = audio_processor.get_data_my(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'training', sess)
    #mid = np.abs(np.max(train_fingerprints) + np.min(train_fingerprints)) / 2
    #half = np.max(train_fingerprints) - np.min(train_fingerprints)
    # train_fingerprints = ((train_fingerprints + mid) / half * 255).astype(int)
    train_fingerprints = np_round_and_clip_7bit(train_fingerprints)

    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy_mean, train_step,
            increment_global_step
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            istrain:True
        })
    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     cross_entropy_value))
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:

      #############################################
      ########  测试集重复计算正确率和混淆矩阵  ######
      set_size = audio_processor.set_size('testing')
      tf.logging.info('set_size=%d', set_size)
      test_fingerprints, test_ground_truth = audio_processor.get_data_my(
          -3, 0, model_settings, 0.0, 0.0, 0, 'testing', sess)
      #mid = np.abs(np.max(test_fingerprints) + np.min(test_fingerprints)) / 2
      #half = np.max(test_fingerprints) - np.min(test_fingerprints)
      test_fingerprints = np_round_and_clip_7bit(test_fingerprints)
      final_summary,test_accuracy, conf_matrix = sess.run(
          [test_summaries,evaluation_step, confusion_matrix],
          feed_dict={
              fingerprint_input: test_fingerprints,
              ground_truth_input: test_ground_truth,
              istrain : False
          })
      final_summary_istrain,test_accuracy_istrain= sess.run(
          [test_summaries_istrain,evaluation_step],
          feed_dict={
              fingerprint_input: test_fingerprints,
              ground_truth_input: test_ground_truth,
              istrain : True
          })

      if test_accuracy > best_acc:
          best_acc = test_accuracy
          best_step = training_step
      if test_accuracy_istrain > best_acc_istrain:
          best_acc_istrain = test_accuracy_istrain
          best_step_istrain = training_step
      test_writer.add_summary(final_summary, training_step)
      test_istrain_writer.add_summary(final_summary_istrain, training_step)
      tf.logging.info('Confusion Matrix:\n %s' % (conf_matrix))
      tf.logging.info('test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100,6882))
      tf.logging.info('test_istrain accuracy = %.1f%% (N=%d)' % (test_accuracy_istrain * 100,6882))

      tf.logging.info('Best test accuracy before now = %.1f%% (N=%d)' % (best_acc * 100,6882) + '  at step of ' + str(best_step))
      tf.logging.info('Best test_istrain accuracy before now = %.1f%% (N=%d)' % (best_acc_istrain * 100,6882) + '  at step of ' + str(best_step_istrain))
    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir + '/'+FLAGS.model_architecture,
                                     FLAGS.model_architecture + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)
    print_line = 'Best test accuracy before now = %.1f%% (N=%d)' % (best_acc * 100,6882) + '  at step of ' + str(best_step) + '\n' + \
                 'Best test_istrain accuracy before now = %.1f%% (N=%d)' % (best_acc_istrain * 100,6882) + '  at step of ' + str(best_step_istrain)
    if training_step == training_steps_max:
        with open(FLAGS.train_dir + '/' +FLAGS.model_architecture+ '/details.txt', 'w') as f:
            f.write(print_line)
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      # default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      default=None,
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      #default='/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/speech_dataset_raw/',
      # default='tmp/speech_dataset/',
      # default='/home/zhangs/zs_data_cut/tmp/train_data',
      default='/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/speech_dataset_v2/',

      help="""\
        Where to download the speech training data to.
        """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      # default=0.1,
      help="""\
        How loud the background noise should be, between 0 and 1.
        """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      # default=0.8,
      help="""\
        How many of the training samples have background noise mixed in.
        """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=3.0,
      # default=0.0,
      help="""\
        How much of the training data should be silence.
        """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      # default=85.0,
      # default=48.0,
      default=48,
      help="""\
        How much of the training data should be unknown words.
        """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      # default=100.0,
      default=150.0,
      help="""\
        Range to randomly shift the training audio by in time.
        """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=0,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs', )
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      # default=1000,
      default=1000,
      # default=500,
      help='Expected duration in milliseconds of the wavs', )
  parser.add_argument(
      '--window_size_ms',
      type=float,
      # default=30.0,
      # default=50.0,
      default=40.0,
      help='How long each spectrogram timeslice is', )
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      # default=10.0,
      default=20.0,
      help='How long each spectrogram timeslice is', )
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=10,
      # default=13,
      help='How many bins to use for the MFCC fingerprint', )
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      # default='1000,1000,1000,5000,2000',
      default='15000',
      # default='1000,8000,4000,1000',
      help='How many training loops to run', )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      # default=400,
      default=100,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.01',
      # default='0.03,0.01,0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once', )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='tmp/retrain_logs_lyc/twowords_dscnn_bnndatav2',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      # default='yes,no,up,down,left,right,on,off,stop,go',
      #default='five,happy,left,marvin,nine,seven,sheila,six,stop,zero',
      default = 'happy',
      help='Words to use (others will be added to an unknown label)', )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='tmp/speech_commands_train/twowords_dscnn_bnndatav2',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='dscnn_bnn',
      # default='dnn',
      # default='cnn_one_fpool3',
      # default='dnn_bn_bnn_3n',
      # default='single_fc',
      # default='low_latency_conv',
      # default='l_ds_conv',
      help='What model architecture to use')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--figure_dir',
      type=str,
      default='tmp/FAandFR',
      help='保存fafr图片的路径')
  parser.add_argument(
      '--beta_norm',
      type=float,
      default=0,
      help='正则化参数')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
