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
#data_dir = 'tmp/speech_dataset_raw/'
data_dir = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/data_over_300'

#data_dir = '/home/zhangs/zs_data_cut/tmp/data_agc/'
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
wanted_words ='on,off'
#wanted_words='five,happy,left,marvin,nine,seven,sheila,six,stop,zero'

model_architecture = 'fixed_point_twn_bnn'
clip_duration_ms = 1000
window_size_ms = 40
window_stride_ms = 20
dct_coefficient_count = 10
start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/speech_commands_train/fixed_point_twn_bnn/fixed_point_twn_bnn.ckpt-8900'
#start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/moxing/ds_conv_2_5000_0.09filler/l_ds_conv_2.ckpt-5000'
#start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech/tmp/moxing/ds_conv_2_5000_0.09filler/l_ds_conv_2.ckpt-5000'
pbs_path = ''

def np_round_and_clip(x):
    x = np.round(x)
    x = np.clip(x, -50, 50)/64
    return x



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
fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')
ground_truth_input = tf.placeholder(
    tf.float32, [None, label_count], name='groundtruth_input')

logits= models.create_model(
      fingerprint_input,
      model_settings,
      model_architecture,
      is_training=False)
softmax = tf.nn.softmax(logits, name='labels_softmax')

with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits))
predicted_indices = tf.argmax(logits, 1)
expected_indices = tf.argmax(ground_truth_input, 1)
correct_prediction = tf.equal(predicted_indices, expected_indices)
confusion_matrix = tf.confusion_matrix(
      expected_indices, predicted_indices, num_classes=label_count)
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

global_step = tf.train.get_or_create_global_step()
increment_global_step = tf.assign(global_step, global_step + 1)
tf.global_variables_initializer().run()
if start_checkpoint:
    models.load_variables_from_checkpoint(sess, start_checkpoint)
    start_step = global_step.eval(session=sess)
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
    -1, 0, model_settings, 0, 0, 0, 'testing', sess)
#test_fingerprints = np.round(test_fingerprints)
test_fingerprints = np_round_and_clip(test_fingerprints)

print("* fingerprint size          ； " + str(model_settings['fingerprint_size']))
print("* test set examples number  ； " + str(np.sum(np.sum(test_ground_truth, axis=0))))
print("* test set features size    ； " + str(test_fingerprints.shape))
print("* test set labels size      ； " + str(test_ground_truth.shape))
if model_settings['fingerprint_size']==test_fingerprints.shape[1] and \
        len(input_data_filler.prepare_words_list_my(wanted_words.split(','))) ==        	test_ground_truth.shape[1] :
    print("------------->  ALL CORRECT <--------------")
else:
    print("------------->  DATA WRONG! <--------------")
print(" ********************************************************" + '\n')
print(" ***************       import ckpt    *******************")
total_accuracy = 0
total_conf_matrix = None
test_accuracy, conf_matrix ,softmax ,correct_prediction,expected_indices,predicted_indices,logits= sess.run(
        [evaluation_step, confusion_matrix, softmax,correct_prediction,expected_indices,predicted_indices,logits],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
        })
total_accuracy += test_accuracy 
if total_conf_matrix is None:
  total_conf_matrix = conf_matrix
else:
  total_conf_matrix += conf_matrix
tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,testing_datas))
print(" ********************************************************" + '\n')
'''
print("softmax shape:  " + str(softmax.shape))
print("correct prediction:" + str(correct_prediction))
index = 0
for wrong_p in correct_prediction:
    if not wrong_p:
        print(str(softmax[index]))
        print(str(expected_indices[index]))
        print(str(predicted_indices[index]))
    index += 1
'''
print(" ***************      gate control    *******************")
def gate_control(gate,softmax,expected_indices):
    softmax_max = np.max(softmax,1)
    softmax_index = np.argmax(softmax,1)
    for i,max_num in enumerate(softmax_max):
        if max_num < gate or max_num == gate:
            softmax_index[i] = 0
    correct_prediction_gate = softmax_index==expected_indices
    accuracy_gate = sum(correct_prediction_gate)/correct_prediction_gate.shape[0]
    print("Accuracy after gate control: " + str(accuracy_gate*100) + '%')
    return accuracy_gate,softmax_index
#acc , softmax_index = gate_control(0.5,softmax,expected_indices)
print(" ********************************************************" + '\n')
print(" ***************       draw  ROC      *******************")

def FAFR_caculate(start_label,label_count,expected_indices,softmax_index):
    FRR = 0
    FAR = 0
    for i in xrange(start_label,label_count):
        label_list = (expected_indices == i).astype('int')
        predict_list =(softmax_index == i).astype('int')
        confuse_list = label_list - predict_list
        confuse_list2 = label_list + predict_list
        FP = np.sum(confuse_list==-1)
        #print("FN =  " + str(FN))
        FN = np.sum(confuse_list == 1)
        #print("FP =  " + str(FP))
        TP =  np.sum(confuse_list2 == 2)
        TN =  np.sum(confuse_list2 == 0)
        #print("T =  " + str(T))
        FRR += FN/(TP + FN)/(label_count-1)
        FAR += FP/(TN + FP)/(label_count-1)
        #FRR += FN/(T + FP)/label_count
        #FAR += FP/(T + FN)/label_count
    return FRR,FAR
#FRR,FAR = FAFR_caculate(label_count,expected_indices,softmax_index)
#print("False Alarm Rate =  " + str(FAR*100) + '%')
#print("False Reject Rate =  " + str(FRR*100) + '%')
def generate_xy(softmax,expected_indices):
    x=[]
    y=[]
    xy=[]
    acc_max = 0
    acc_max_gate = 0
    acc_max_FA = 0
    acc_max_FR = 0
    auc = 0
    prev_xx = 0
    tipical_fa = 0
    count = 0
    for i in np.arange(0,1.0001,0.0001):
        acc , softmax_index = gate_control(i,softmax,expected_indices)
        #print("this is a check (should be 0):" + str(sum(softmax_index)))
        FRR,FAR = FAFR_caculate(1,3,expected_indices,softmax_index)

        if acc > acc_max:
            acc_max = acc
            acc_max_gate = i
            acc_max_FA = FAR
            acc_max_FR = FRR
        x.append(FAR)
        y.append(FRR)
        xy.append([FAR,FRR])
    #cacluate AUC
    xy.sort()
    for xx,yy in xy:
        if xx != prev_xx:
            auc += (xx-prev_xx)*yy
            prev_xx = xx
        if xx>0.0019 and xx<0.0021:
            tipical_fa += yy
            count +=1
    tipical_fa =tipical_fa/(count+0.01)
    return x,y,xy,acc_max,acc_max_gate,acc_max_FA,acc_max_FR,auc,tipical_fa

x,y,xy,acc_max,acc_max_gate,acc_max_FA,acc_max_FR,auc,tipical_fa = generate_xy(softmax,expected_indices)
plt.figure(figsize=(8, 8))
#plt.scatter(x, y, label='FA  Vs. FR ', color='red', linewidth=0.01)  #san dian
plt.plot(x, y, label='FA  Vs. FR ',linewidth=1, color='red' )  #san dian
# plt.xlim(0,0.02)
#plt.ylim(0,0.5)
#plt.xlim(0, 0.02)

plt.xlabel('False Alarm rate')
plt.ylabel('False reject rate')
plt.title(' FA  Vs. FR picture')
#plt.legend()
plt.show()
print(" ***************    save fa fr data   *******************")
# data_array = np.array(xy)
# print(type(data_array))
# save_path = '/home/zhangs/tensorflow-master/tensorflow/examples/speech/tmp/fa_fr_data/'
# np.savetxt(save_path + model_architecture+"_fafr_data.csv",data_array,delimiter=',')
print(" ***************      max accuracy    *******************")
print('* AUC                        :  '+ str(auc))
print('* 0.2%FA point:              :  '+ str(tipical_fa*100) + '% FR')
print('* Max accuracy :                '+ str(acc_max*100) + ' %' + ' (%.1f%% raw_acc)'% (total_accuracy * 100))
print('* FA at max accuracy  :         '+ str(acc_max_FA*100) + ' %')
print('* FR at max accuracy  :         '+ str(acc_max_FR*100) + ' %')
print('* Gate value at max accuracy :  '+ str(acc_max_gate))

