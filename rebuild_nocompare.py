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
#from confuse_test import *
import matplotlib.pyplot as plt
from numpy import asfarray
import rebuild_model
#data_dir = 'tmp/speech_dataset_raw/'
data_dir = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/newfeature4/'
#data_dir ='/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/H5_diff_feature/'

silence_percentage = 3
validation_percentage = 0
testing_percentage = 10
sample_rate = 16000
batch_size = 100
#wanted_words ='yes,no,up,down,left,right,on,off,stop,go'
wanted_words ='happy'
#wanted_words='five,happy,left,marvin,nine,seven,sheila,six,stop,zero'

model_architecture = 'dscv1'
clip_duration_ms = 998
window_size_ms = 40
window_stride_ms = 20
dct_coefficient_count = 16
#start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/speech_commands_train/hd5_train_2words_8bit_mixup_16fea_bnnv2/fixed_point_bnn_v2/fixed_point_bnn_v2.ckpt-9800'
start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/speech_commands_train/h5_40_20ms16_1word_happy_dscv1_mix1Q6nobias/dscv1/dscv1.ckpt-5800'
pbs_path = ''

# We want to see all the logging messages for this tutorial.
tf.logging.set_verbosity(tf.logging.INFO)

# Start a new TensorFlow session.
sess = tf.InteractiveSession()

model_settings = models.prepare_model_settings(
  len(new_features_input.prepare_words_list_my(wanted_words.split(','))),
  sample_rate, clip_duration_ms, window_size_ms,
  window_stride_ms, dct_coefficient_count)
audio_processor = new_features_input.AudioProcessor(
    data_dir, silence_percentage,
    wanted_words.split(','), validation_percentage,
    testing_percentage)
fingerprint_size = model_settings['fingerprint_size']
label_count = model_settings['label_count']

fingerprint_input = tf.placeholder(
     # tf.float32, [None, fingerprint_size], name='fingerprint_input')    #
    tf.float32, [None, 20,7,64], name='fingerprint_input')    #
ground_truth_input = tf.placeholder(
    tf.float32, [None, label_count], name='groundtruth_input')
###########################
dscv1model = rebuild_model.paras(start_checkpoint)
logits,lconv1,lconv1bn,ldsc1,ldsc1bn,ldsc1pw,ldsc1pwbn,ldsc2,ldsc2bn,ldsc2pw,ldsc2pwbn,ldsc3,ldsc3bn,ldsc3pw,ldsc3pwbn,\
           ldsc4,ldsc4bn,ldsc4pw,ldsc4pwbn,lfc= dscv1model.build_model(
      fingerprint_input,
      model_settings)

##########################



softmax = tf.nn.softmax(logits, name='labels_softmax')
print(softmax)

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
test_fingerprints_raw, test_ground_truth= audio_processor.get_data_my(
   -1, 0, model_settings, 'testing')

#####################
# test_fingerprints 就是输入数据 维度不记得了，，反正是numpy格式的，你在这里写第一层，然后输出名字还叫这个test_fingerprints，我在model里改输入
dd1=1320
dd4=1
dd2=48
dd3=16
print(test_fingerprints_raw.shape)
input1=test_fingerprints_raw.reshape([1320,48,16])
print(input1)
#dd2,dd3=test_fingerprints.shape
print(dd2,dd3)
wd1,wd2,wd3,wd4=dscv1model.conv1_w.shape
print(wd1,wd2,wd3,wd4)
m=int((dd2-wd1)/2+1)
n=int((dd3-wd2)/2+1)
tmp=np.zeros((dd1,m,n,wd4))
for i in range(dd1):
    for j in range(wd4):
        for k in range(m):
            k1=k*2
            for h in range(n):
                h1=h*2
                temp=np.multiply(input1[i,k1:k1+wd1,h1:h1+wd2],dscv1model.conv1_w[:,:,0,j])
                tmp[i,k,h,j]=temp.sum()
test_fingerprints=tmp
#test_fingerprints=test_fingerprints.reshape(dd1,768)
print(test_fingerprints.shape)
##################
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
total_accuracy = 0
total_conf_matrix = None
test_accuracy, conf_matrix ,softmax ,correct_prediction,expected_indices,predicted_indices,logits= sess.run(
        [evaluation_step, confusion_matrix, softmax,correct_prediction,expected_indices,predicted_indices,logits],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth
        })


# num_of_inconfident =0
# print("softmax out:" + str(softmax))
# for ii in range(len(list(softmax))):
#     if np.max(softmax[ii])<0.8:
#         num_of_inconfident += 1
#         print("Inconfident sample: " + str(softmax[ii]))
# print("Inconfident number: " + str(num_of_inconfident))

total_accuracy += test_accuracy
if total_conf_matrix is None:
  total_conf_matrix = conf_matrix
else:
  total_conf_matrix += conf_matrix
tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,testing_datas))
print(" ********************************************************" + '\n')

print(" ***************      gate control    *******************")
def gate_control(gate,softmax,expected_indices):
    softmax_max = np.max(softmax,1)
    softmax_index = np.argmax(softmax,1)
    for i,max_num in enumerate(softmax_max):
        if max_num < gate or max_num == gate:
            softmax_index[i] = 0
    correct_prediction_gate = softmax_index==expected_indices
    accuracy_gate = sum(correct_prediction_gate)/correct_prediction_gate.shape[0]
   # print("Accuracy after gate control: " + str(accuracy_gate*100) + '%')
    return accuracy_gate,softmax_index
#acc , softmax_index = gate_control(0.5,softmax,expected_indices)
print(" ********************************************************" + '\n')
print(" ***************       draw  ROC      *******************")

def FAFR_caculate(start_label,expected_indices,softmax_index):
    label_list = (expected_indices == start_label).astype('int')
    predict_list =(softmax_index == start_label).astype('int')
    confuse_list = label_list - predict_list
    confuse_list2 = label_list + predict_list
    FP = np.sum(confuse_list==-1)
    #print("FN =  " + str(FN))
    FN = np.sum(confuse_list == 1)
    #print("FP =  " + str(FP))
    TP =  np.sum(confuse_list2 == 2)
    TN =  np.sum(confuse_list2 == 0)
    #print("T =  " + str(T))
    FRR = FN/(TP + FN)
    FAR = FP/(TN + FP)
    #FAFR.append([FAR,FRR])
    #FRR += FN/(T + FP)/label_count
    #FAR += FP/(T + FN)/label_count
    return FRR,FAR
#FRR,FAR = FAFR_caculate(label_count,expected_indices,softmax_index)
#print("False Alarm Rate =  " + str(FAR*100) + '%')
#print("False Reject Rate =  " + str(FRR*100) + '%')
def generate_xy(softmax,expected_indices):
    acc_max = 0
    acc_max_gate = 0
    x_max = 0.02
    d_x = 0.0001
    x_plot = np.arange(0, x_max, d_x)
    FAFRlist = []
    for n in range(1,label_count):
        FAFR = []
        frr =[]
        far =[]
        for i in np.arange(0,1,0.0005):
            acc , softmax_index = gate_control(i,softmax,expected_indices)
            #print("this is a check (should be 0):" + str(sum(softmax_index)))
            FRR, FAR = FAFR_caculate(n,expected_indices,softmax_index)
            FAFR.append([FAR, FRR])
            if acc > acc_max:
                     acc_max = acc
                     acc_max_gate = i
        FAFR.sort()
        #print("FAFR" + str(FAFR))
        out_fr = []
        fa_prev = 0
        fr_prev = 1
        FAFR.append([x_max,FAFR[-1][1]])
        # for m in xrange(len(FAFR)):
        #     far.append(FAFR[m][0])
        #     frr.append(FAFR[m][1])
        # plt.subplot(211)
        # plt.plot(far, frr, label='FA  Vs. FR ', linewidth=1, color='blue')  # san dian

        for ii in x_plot:
            for fa,fr in FAFR:
                if ii == 0:
                    my_fr =1
                    out_fr.append(my_fr)
                    break
                if (ii> fa_prev or ii ==fa_prev) and ii<fa:
                    my_fr = (fr_prev -fr)*(fa-ii)/(fa-fa_prev)+fr
                    out_fr.append(my_fr)
                    break
                fa_prev = fa
                fr_prev = fr
        #print("out_fr shape = " +str(len(out_fr)))
        #print("out_fr = " +str(out_fr))
        FAFRlist.append(out_fr)
    y_plot = np.sum(np.array(FAFRlist),axis=0)/(label_count-1)
    print("The ploted y label have dimantion of:" + str(np.array(y_plot).shape))

    tipical_fa =y_plot[int(0.005 / d_x) - 1]
    auc = sum(d_x*y_plot)

    return acc_max,acc_max_gate,auc,tipical_fa,[list(x_plot),list(y_plot)]

def plot_and_save(x_plot,y_plot,save_path):
    plt.plot(list(x_plot), y_plot, label='FA  Vs. FR ', linewidth=1, color='red')  # san dian
    plt.xlabel('False Alarm rate')
    plt.ylabel('False reject rate')
    plt.title(' FA  Vs. FR picture')
    # plt.ylim(0, 0.4)
    # plt.xlim(0, 0.02)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        plt.savefig(save_path + '/FAFR.png', format='png')
    else:
        plt.savefig(save_path + '/FAFR.png', format='png')
    plt.show()
print(" ***************    save fa fr data   *******************")
save_path = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/fa_fr_data/h5_features/' + model_architecture
acc_max,acc_max_gate,auc,tipical_fa,xy = generate_xy(softmax,expected_indices)
plot_and_save(xy[0],xy[1],save_path)
data_array = np.array(xy)
print(type(data_array))
np.savetxt(save_path + "/fafr_data.csv",data_array,delimiter=',')
print(" ***************      max accuracy    *******************")
print('* AUC                        :  '+ str(auc))
print('* Max accuracy :                '+ str(acc_max*100) + ' %' + ' (%.1f%% raw_acc)'% (total_accuracy * 100))
print('* Gate value at max accuracy :  '+ str(acc_max_gate))
print('* 0.5%FA point:              :  '+ str(tipical_fa*100) + '% FR')
print_line = '* AUC                        :  '+ str(auc) + '\n' + \
             '* Max accuracy               :  '+ str(acc_max*100) + ' %' + ' (%.1f%% raw_acc)'% (total_accuracy * 100) + '\n' + \
             '* Gate value at max accuracy :  '+ str(acc_max_gate) + '\n' + \
             '* 0.5%FA point:              :  '+ str(tipical_fa*100) + '% FR'
with open (save_path + '/details.txt','w') as f:
    f.write(print_line)

num_of_unconfident = 0
num_of_wrong = 0
print("softmax out:" + str(softmax))
_, softmax_index_final = gate_control(acc_max_gate, softmax, expected_indices)
confusion_matrix_final = tf.confusion_matrix(
      expected_indices, softmax_index_final, num_classes=label_count)
for ii in range(len(list(softmax))):
    if np.max(softmax[ii])<acc_max_gate:
        num_of_unconfident += 1
        print("Unconfident sample: " + str(softmax[ii]))
        #if np.max(softmax[ii]) !=  softmax[ii][0]:
        if expected_indices[ii] != 0:
            print("Wrong sample: " + str(softmax[ii]))
            num_of_wrong +=1
print("Unconfident number: " + str(num_of_unconfident))
print("Wrong number: " + str(num_of_wrong))
print("confusion matrix:" )
print(str(sess.run(confusion_matrix_final)))