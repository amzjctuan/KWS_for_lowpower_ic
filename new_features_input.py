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

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile
import wave
import contextlib
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import h5py  #导入工具包

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
RANDOM_SEED = 59185

FILLER_WORD_LABEL = '_filler_'
FILLER_WORD_INDEX = 0

def prepare_words_list_my(wanted_words):
  """Prepends common tokens to the custom word list.

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
    List with the standard silence and unknown tokens added.
  """
  return [FILLER_WORD_LABEL] + wanted_words

def prepare_words_list(wanted_words):
  """Prepends common tokens to the custom word list.

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
    List with the standard silence and unknown tokens added.
  """
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)                    #筛选出人名
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_dir, silence_percentage, wanted_words, validation_percentage, testing_percentage):
    self.data_dir = data_dir
    self.prepare_data_index(silence_percentage,wanted_words,validation_percentage,testing_percentage)

  def prepare_data_index(self, silence_percentage,wanted_words, validation_percentage,testing_percentage):
    """Prepares a list of the samples organized by set and label.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 1
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    self.unknown_index =  {'validation': [], 'testing': [], 'training': []}
    self.final_index =  {'validation': [], 'testing': [], 'training': []}
    self.seven_to_one_set = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.data_dir, '*', '*.h5')
    for wav_path in gfile.Glob(search_path):
      _, word = os.path.split(os.path.dirname(wav_path))
      word = word.lower()
      # Treat the '_background_noise_' folder as a special case, since we expect
      # it to contain long audio samples we mix in to improve training.
      # if word == BACKGROUND_NOISE_DIR_NAME:
      #   continue
      #no need
      all_words[word] = True
      set_index = which_set(wav_path, validation_percentage, testing_percentage)
      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      if word in wanted_words_index:
        self.data_index[set_index].append({'label': word, 'file': wav_path})
      else:
        unknown_index[set_index].append({'label': word, 'file': wav_path})
        self.unknown_index[set_index] = unknown_index[set_index]
            #all_words 包含所有音频的关键词
    if not all_words:
      raise Exception('No .wavs found at ' + search_path)
    for index, wanted_word in enumerate(wanted_words):     #data set has no such word
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      # Pick some unknowns to add to each partition of the data set.
      random.shuffle(self.unknown_index[set_index])
      random.shuffle(unknown_index[set_index])
      # unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
      # self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    # Make sure the ordering is random.
    self.unknown_num = {'validation':[], 'testing':[], 'training':[]}
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
      ########################################
      self.final_index[set_index]  = self.data_index[set_index] + self.unknown_index[set_index] #new
      random.shuffle(self.final_index[set_index])

      set_size = len(self.data_index[set_index])
      unknown_size = int(math.ceil(set_size * 7))
      # for ii,clip in enumerate(self.unknown_index[set_index]):
      #   self.seven_to_one_set[set_index].append(clip)
      #   if ii == unknown_size:
      #     break
      self.seven_to_one_set[set_index].extend(self.unknown_index[set_index][:unknown_size])
      self.unknown_num[set_index] =  len(self.seven_to_one_set[set_index])
      self.seven_to_one_set[set_index].extend(self.data_index[set_index])
      print('seven_to_one_set:'+ set_index +" has  " + str(len(self.seven_to_one_set[set_index])) + ' data,with ' + str(self.unknown_num[set_index]) +' fillers')
      print('finalset:'+ set_index +" has  " + str(len(self.final_index[set_index])) + ' data,with ' + str(len(self.unknown_index[set_index])) +' fillers')

      random.shuffle(self.seven_to_one_set[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list_my(wanted_words)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]  #wanted_words_index[wanted_word] = index + 2
      else:
        self.word_to_index[word] = FILLER_WORD_INDEX    # UNKNOWN_WORD_INDEX=1
    self.word_to_index[SILENCE_LABEL] = FILLER_WORD_INDEX   #SILENCE_INDEX=0

  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])
  def get_data_my(self, how_many, offset, model_settings, mode):
    """Gather samples from the data set, applying transformations as needed.

    Returns:
      List of sample data for the transformed samples, and list of labels in
      one-hot form.
    """
    candidates = self.data_index[mode]
    if how_many == -1:
        #sample_count = len(candidates) + len(self.unknown_index[mode])
        sample_count = len(self.seven_to_one_set[mode])
    elif how_many == -2:
      sample_count = 100
    elif how_many == -3:
      sample_count = len(candidates) + len(self.unknown_index[mode])
    else:
      #sample_count = max(0, min(how_many, len(candidates) - offset))   #offset=0 实际sample_count=batchsize
      sample_count = how_many  #offset=0 实际sample_count=batchsize

    # Data and labels will be populated and returned.
    # data = np.zeros((sample_count, model_settings['fingerprint_size']))
    # labels = np.zeros((sample_count, model_settings['label_count']))
    data = np.zeros((sample_count,model_settings['fingerprint_size']))
    h5data = np.zeros((model_settings['dct_coefficient_count'],model_settings['spectrogram_length']))
    labels = np.zeros((sample_count, model_settings['label_count']))
    # offset 为样本数偏置
    wrong_num = 0
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      if how_many == -1:   #new   or pick_deterministically:
          sample_index = i                            #
          sample = self.seven_to_one_set[mode][sample_index]#
          #sample = self.seven_to_one_set[mode][sample_index]#
      elif how_many == -3:
          sample_index = i                            #
          sample = self.final_index[mode][sample_index]#
      else:
        #random_seed =random_seed +1
        rate = np.random.random()
        # print ("rate = " + str(rate))
        if rate < 0.5:#0.35 filler  0.39-->7:1rate(used with-->seven to one set)
          sample_index = np.random.randint(len(self.unknown_index[mode]))
          sample = self.unknown_index[mode][sample_index]
        else :
          sample_index = np.random.randint(len(candidates))
          sample = candidates[sample_index]
      # If we're time shifting, set up the offset for this sample.
      # Choose a section of background noise to mix in.
      with h5py.File(sample['file'], 'r') as f:
          for key in f.keys():
              # print(key)
              # print(f[key].name)
              # print(f[key].shape)
              if f[key].shape != (model_settings['dct_coefficient_count'],model_settings['spectrogram_length']):
                  wrong_num = wrong_num + 1
                  print('data wrong! ;' + sample['file'])
                  print(wrong_num)
                  print(f[key].shape)
                  #os.remove(sample['file'])
              else :
                h5data = f[key].value
      # If we want silence, mute out the main sample but leave the background.
          if sample['label'] == SILENCE_LABEL:
              h5data = h5data*0
          data[i - offset, :] = h5data.flatten()
          label_index = self.word_to_index[sample['label']]
          labels[i - offset, label_index] = 1
    return data, labels

