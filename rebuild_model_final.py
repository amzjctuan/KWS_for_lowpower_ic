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
    #print(name + " :" + str(w))
    return w
def read_weight_v3(name,reader):  #not ternary
    w = np.sign(reader.get_tensor(name))
    print(name + " shape :" + str(w.shape))
    #print(name + " :" + str(w))
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
    bias = -1.0000 * moving_mean + beta/mul
    return mul,bias
def batchnorm(x,conv1_bn_mul,conv1_bn_bias):
    return  tf.multiply(tf.add(x, conv1_bn_bias), conv1_bn_mul)


class paras(object):
    def __init__(self, start_checkpoint):
        self.read_dscv1model(start_checkpoint)
    def read_dscv1model(self,start_checkpoint):
        reader = tf.train.NewCheckpointReader(start_checkpoint)
        self.all_variables = reader.get_variable_to_shape_map()
        #############weights##################
        self.conv1_w = read_weight_v2('CONV1_layer/CONV1/conv/weights',reader)

        self.DSC1_w = read_weight_v2('DSC1_layer/DSC1/conv/weights',reader)
        self.DSC1pw_w = read_weight_v2('DSC1_layer/DSC1_pointwise/conv/weights',reader)

        self.DSC2_w = read_weight_v2('DSC2_layer/DSC2/conv/weights',reader)
        self.DSC2pw_w = read_weight_v2('DSC2_layer/DSC2_pointwise/conv/weights',reader)

        self.DSC3_w = read_weight_v2('DSC3_layer/DSC3/conv/weights',reader)
        self.DSC3pw_w = read_weight_v2('DSC3_layer/DSC3_pointwise/conv/weights',reader)

        self.DSC4_w = read_weight_v2('DSC4_layer/DSC4/conv/weights',reader)
        self.DSC4pw_w = read_weight_v2('DSC4_layer/DSC4_pointwise/conv/weights',reader)

        self.FCout_w = read_weight_v2('FCout_layer/FCout/dense/weights',reader)
        ##############################################################
        #############bn parameters##################
        self.conv1_bn_mul,self.conv1_bn_bias = read_bn('CONV1_layer/BN1',reader)

        self.dsc1_bn_mul,self.dsc1_bn_bias = read_bn('DSC1_layer/BNdsc1',reader)
        self.dsc1pw_bn_mul,self.dsc1pw_bn_bias = read_bn('DSC1_layer/BNpw1',reader)

        self.dsc2_bn_mul,self.dsc2_bn_bias = read_bn('DSC2_layer/BNdsc2',reader)
        self.dsc2pw_bn_mul,self.dsc2pw_bn_bias = read_bn('DSC2_layer/BNps2',reader)

        self.dsc3_bn_mul,self.dsc3_bn_bias = read_bn('DSC3_layer/BNdsc3',reader)
        self.dsc3pw_bn_mul,self.dsc3pw_bn_bias = read_bn('DSC3_layer/BNps3',reader)

        self.dsc4_bn_mul,self.dsc4_bn_bias = read_bn('DSC4_layer/BNdsc4',reader)
        self.dsc4pw_bn_mul,self.dsc4pw_bn_bias = read_bn('DSC4_layer/BNps4',reader)

        self.FC_bn_mul,self.FC_bn_bias = read_bn('FCout_layer/BNfcout',reader)

    def build_model(self,input,model_settings):
        input_frequency_size = model_settings['dct_coefficient_count']
        input_time_size = model_settings['spectrogram_length']
        ##############################################
        input = tf.reshape(input, [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
        x = tf.nn.conv2d(input, self.conv1_w, [1, 2, 2, 1], padding='VALID')
        lconv1=x
        x = batchnorm(x, self.conv1_bn_mul,self.conv1_bn_bias)
        lconv1bn=x
        x = relu(x)
        print(x.shape)

        #dsc1
        x = depthwise_conv2d(name='DSC1', x=x, w=self.DSC1_w)
        ldsc1=x
        x = batchnorm(x, self.dsc1_bn_mul,self.dsc1_bn_bias)
        ldsc1bn=x
        x = relu(x)
        print(x.shape)
        x = tf.nn.conv2d(x, self.DSC1pw_w, [1, 1, 1, 1], padding='SAME')
        ldsc1pw=x
        #x = conv2d('DSC1_pointwise', x=x,w=self.DSC1pw_w,num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        x = batchnorm(x, self.dsc1pw_bn_mul,self.dsc1pw_bn_bias)
        ldsc1pwbn=x
        x = relu(x)
        print(x.shape)

        #dsc2
        x = depthwise_conv2d(name='DSC2', x=x, w=self.DSC2_w)
        ldsc2=x
        x = batchnorm(x, self.dsc2_bn_mul,self.dsc2_bn_bias)
        ldsc2bn=x
        x = relu(x)
        print(x.shape)
        x = tf.nn.conv2d(x, self.DSC2pw_w, [1, 1, 1, 1], padding='SAME')
        ldsc2pw=x
        #x = conv2d('DSC2_pointwise', x=x,w=self.DSC2pw_w,num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        x = batchnorm(x, self.dsc2pw_bn_mul,self.dsc2pw_bn_bias)
        ldsc2pwbn=x
        x = relu(x)
        print(x.shape)

        #dsc3
        x = depthwise_conv2d(name='DSC3', x=x, w=self.DSC3_w)
        ldsc3=x
        x = batchnorm(x, self.dsc3_bn_mul,self.dsc3_bn_bias)
        ldsc3bn=x
        x = relu(x)
        print(x.shape)
        x = tf.nn.conv2d(x, self.DSC3pw_w, [1, 1, 1, 1], padding='SAME')
        ldsc3pw=x
        #x = conv2d('DSC3_pointwise', x=x,w=self.DSC3pw_w,num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        x = batchnorm(x, self.dsc3pw_bn_mul,self.dsc3pw_bn_bias)
        ldsc3pwbn=x
        x = relu(x)
        print(x.shape)

        #dsc4
        x = depthwise_conv2d(name='DSC4', x=x, w=self.DSC4_w)
        ldsc4=x
        x = batchnorm(x, self.dsc4_bn_mul,self.dsc4_bn_bias)
        ldsc4bn=x
        x = relu(x)
        print(x.shape)
        x = tf.nn.conv2d(x, self.DSC4pw_w, [1, 1, 1, 1], padding='SAME')
        ldsc4pw=x
        #x = conv2d('DSC4_pointwise', x=x,w=self.DSC4pw_w,num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        x = batchnorm(x, self.dsc4pw_bn_mul,self.dsc4pw_bn_bias)
        ldsc4pwbn=x
        x = relu(x)
        print(x.shape)

        #fc
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = tf.matmul(x, self.FCout_w)
        lfc=x
        print("final out shape：" +str(x))
        x = batchnorm(x, self.FC_bn_mul,self.FC_bn_bias)

        return x,lconv1,lconv1bn,ldsc1,ldsc1bn,ldsc1pw,ldsc1pwbn,ldsc2,ldsc2bn,ldsc2pw,ldsc2pwbn,ldsc3,ldsc3bn,ldsc3pw,ldsc3pwbn,\
           ldsc4,ldsc4bn,ldsc4pw,ldsc4pwbn,lfc
class paras_bnn(object):
    def __init__(self, start_checkpoint):
        self.read_dscv1model(start_checkpoint)
    def read_dscv1model(self,start_checkpoint):
        reader = tf.train.NewCheckpointReader(start_checkpoint)
        self.all_variables = reader.get_variable_to_shape_map()
        #############weights##################
        self.conv1_w = read_weight_v3('CONV1_layer/CONV1/conv/weights',reader)

        self.DSC1_w = read_weight_v3('DSC1_layer/DSC1/conv/weights',reader)

        self.DSC1pw_w = read_weight_v3('DSC1_layer/DSC1_pointwise/conv/weights',reader)


        self.DSC2_w = read_weight_v3('DSC2_layer/DSC2/conv/weights',reader)

        self.DSC2pw_w = read_weight_v3('DSC2_layer/DSC2_pointwise/conv/weights',reader)

        self.DSC3_w = read_weight_v3('DSC3_layer/DSC3/conv/weights',reader)
        self.DSC3pw_w = read_weight_v3('DSC3_layer/DSC3_pointwise/conv/weights',reader)

        self.DSC4_w = read_weight_v3('DSC4_layer/DSC4/conv/weights',reader)
        self.DSC4pw_w = read_weight_v3('DSC4_layer/DSC4_pointwise/conv/weights',reader)

        self.FCout_w = read_weight_v3('FCout_layer/FCout/dense/weights',reader)
        ##############################################################
        #############bn parameters##################
        self.conv1_bn_mul,self.conv1_bn_bias = read_bn('CONV1_layer/BN1',reader)

        self.dsc1_bn_mul,self.dsc1_bn_bias = read_bn('DSC1_layer/BNdsc1',reader)
        self.dsc1pw_bn_mul,self.dsc1pw_bn_bias = read_bn('DSC1_layer/BNpw1',reader)

        self.dsc2_bn_mul,self.dsc2_bn_bias = read_bn('DSC2_layer/BNdsc2',reader)
        self.dsc2pw_bn_mul,self.dsc2pw_bn_bias = read_bn('DSC2_layer/BNps2',reader)

        self.dsc3_bn_mul,self.dsc3_bn_bias = read_bn('DSC3_layer/BNdsc3',reader)
        self.dsc3pw_bn_mul,self.dsc3pw_bn_bias = read_bn('DSC3_layer/BNps3',reader)

        self.dsc4_bn_mul,self.dsc4_bn_bias = read_bn('DSC4_layer/BNdsc4',reader)
        self.dsc4pw_bn_mul,self.dsc4pw_bn_bias = read_bn('DSC4_layer/BNps4',reader)

        self.FC_bn_mul,self.FC_bn_bias = read_bn('FCout_layer/BNfcout',reader)

    def build_model(self,input,model_settings):
        input_frequency_size = model_settings['dct_coefficient_count']
        input_time_size = model_settings['spectrogram_length']
        ##############################################
        input = tf.reshape(input, [-1, input_time_size, input_frequency_size, 1])  # N H W C mode #8 replace =binarize
        x = tf.nn.conv2d(input, self.conv1_w, [1, 2, 2, 1], padding='VALID')
        lconv1=x
        x = batchnorm(x, self.conv1_bn_mul,self.conv1_bn_bias)
        lconv1bn=x
        x = binarize_v2(x)
        print(x.shape)

        #dsc1
        x = depthwise_conv2d(name='DSC1', x=x, w=self.DSC1_w)
        ldsc1=x
        x = batchnorm(x, self.dsc1_bn_mul,self.dsc1_bn_bias)
        ldsc1bn=x
        x = binarize_v2(x)
        print(x.shape)
        x = tf.nn.conv2d(x, self.DSC1pw_w, [1, 1, 1, 1], padding='SAME')
        ldsc1pw=x
        #x = conv2d('DSC1_pointwise', x=x,w=self.DSC1pw_w,num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        x = batchnorm(x, self.dsc1pw_bn_mul,self.dsc1pw_bn_bias)
        ldsc1pwbn=x
        x = binarize_v2(x)
        print(x.shape)

        #dsc2
        x = depthwise_conv2d(name='DSC2', x=x, w=self.DSC2_w)
        ldsc2=x
        x = batchnorm(x, self.dsc2_bn_mul,self.dsc2_bn_bias)
        ldsc2bn=x
        x = binarize_v2(x)
        print(x.shape)
        x = tf.nn.conv2d(x, self.DSC2pw_w, [1, 1, 1, 1], padding='SAME')
        ldsc2pw=x
        #x = conv2d('DSC2_pointwise', x=x,w=self.DSC2pw_w,num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        x = batchnorm(x, self.dsc2pw_bn_mul,self.dsc2pw_bn_bias)
        ldsc2pwbn=x
        x = binarize_v2(x)
        print(x.shape)

        #dsc3
        x = depthwise_conv2d(name='DSC3', x=x, w=self.DSC3_w)
        ldsc3=x
        x = batchnorm(x, self.dsc3_bn_mul,self.dsc3_bn_bias)
        ldsc3bn=x
        x = binarize_v2(x)
        print(x.shape)
        x = tf.nn.conv2d(x, self.DSC3pw_w, [1, 1, 1, 1], padding='SAME')
        ldsc3pw=x
        #x = conv2d('DSC3_pointwise', x=x,w=self.DSC3pw_w,num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        x = batchnorm(x, self.dsc3pw_bn_mul,self.dsc3pw_bn_bias)
        ldsc3pwbn=x
        x = binarize_v2(x)
        print(x.shape)

        #dsc4
        x = depthwise_conv2d(name='DSC4', x=x, w=self.DSC4_w)
        ldsc4=x
        x = batchnorm(x, self.dsc4_bn_mul,self.dsc4_bn_bias)
        ldsc4bn=x
        x = binarize_v2(x)
        print(x.shape)
        x = tf.nn.conv2d(x, self.DSC4pw_w, [1, 1, 1, 1], padding='SAME')
        ldsc4pw=x
        #x = conv2d('DSC4_pointwise', x=x,w=self.DSC4pw_w,num_filters=64, kernel_size=(1, 1), padding='SAME', stride=(1, 1))
        x = batchnorm(x, self.dsc4pw_bn_mul,self.dsc4pw_bn_bias)
        ldsc4pwbn=x
        x = binarize_v2(x)
        print(x.shape)

        #fc
        x = avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID')
        x = flatten(x)
        x = tf.matmul(x, self.FCout_w)
        lfc=x
        print("final out shape：" +str(x))
        x = batchnorm(x, self.FC_bn_mul,self.FC_bn_bias)

        return x,lconv1,lconv1bn,ldsc1,ldsc1bn,ldsc1pw,ldsc1pwbn,ldsc2,ldsc2bn,ldsc2pw,ldsc2pwbn,ldsc3,ldsc3bn,ldsc3pw,ldsc3pwbn,\
           ldsc4,ldsc4bn,ldsc4pw,ldsc4pwbn,lfc

def plot_hist(name,x,xx,i):
    plt.subplot(2,10,i)
    plt.title(name + 'mul')
    plt.hist(x, bins=60, color='steelblue', normed=True)
    plt.subplot(2,10,i+10)
    plt.title(name + 'bias')
    plt.hist(xx, bins=60, color='steelblue', normed=True)


if __name__ == '__main__':
    start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/tmp/speech_commands_train/h5_40_20ms16_1word_happy_bnn_mix1Q6nobiaswvj/fixed_point_bnn/fixed_point_bnn.ckpt-5700'  #bnn

    #start_checkpoint = '/home/zhangs/tensorflow-master/tensorflow/examples/speech_lyc/ \
    #tmp/speech_commands_train/h5_40_20ms16_1word_happy_dscv1_mix1Q6nobias/dscv1/dscv1.ckpt-5800' #all point

    #par = paras(start_checkpoint)   #all point
    par = paras_bnn(start_checkpoint)   #bnn
    # plot_hist('CONV1_layer',par.conv1_bn_mul,par.conv1_bn_bias,1)
    # plot_hist('dsc1',par.dsc1_bn_mul,par.dsc1_bn_bias,2)
    # plot_hist('dsc1pw',par.dsc1pw_bn_mul,par.dsc1pw_bn_bias,3)
    # plot_hist('dsc2',par.dsc2_bn_mul,par.dsc2_bn_bias,4)
    # plot_hist('dsc2pw',par.dsc2pw_bn_mul,par.dsc2pw_bn_bias,5)
    # plot_hist('dsc3',par.dsc3_bn_mul,par.dsc3_bn_bias,6)
    # plot_hist('dsc3pw',par.dsc3pw_bn_mul,par.dsc3pw_bn_bias,7)
    # plot_hist('dsc4',par.dsc4_bn_mul,par.dsc4_bn_bias,8)
    # plot_hist('dsc4pw',par.dsc4pw_bn_mul,par.dsc4pw_bn_bias,9)
    # plot_hist('fc',par.FC_bn_mul,par.FC_bn_bias,10)
    # plt.show()
    #conv1 = np.reshape(par.conv1_w,[10,4,1,64])
    #np.save("conv1.npy", conv1)
    #a = np.load("conv1.npy")
#    print(a.shape)
    np.save('/home/zhangs/lyc/hardware_data/' + 'conv1_w', par.conv1_w)
    np.save('/home/zhangs/lyc/hardware_data/' + 'DSC1_w', par.DSC1_w)
    np.save('/home/zhangs/lyc/hardware_data/' + 'DSC1pw_w', par.DSC1pw_w)
    np.save('/home/zhangs/lyc/hardware_data/' + 'DSC2_w', par.DSC2_w)
    np.save('/home/zhangs/lyc/hardware_data/' + 'DSC2pw_w', par.DSC2pw_w)
    np.save('/home/zhangs/lyc/hardware_data/' + 'DSC3_w', par.DSC3_w)
    np.save('/home/zhangs/lyc/hardware_data/' + 'DSC3pw_w', par.DSC3pw_w)
    np.save('/home/zhangs/lyc/hardware_data/' + 'DSC4_w', par.DSC4_w)
    np.save('/home/zhangs/lyc/hardware_data/' + 'DSC4pw_w', par.DSC4pw_w)
    np.save('/home/zhangs/lyc/hardware_data/' + 'FCout_w', par.FCout_w)

    np.save('/home/zhangs/lyc/hardware_data/' + 'conv1_bn_mul', par.conv1_bn_mul)
    np.save('/home/zhangs/lyc/hardware_data/' + 'conv1_bn_bias', par.conv1_bn_bias)

    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc1_bn_mul', par.dsc1_bn_mul)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc1_bn_bias', par.dsc1_bn_bias)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc1pw_bn_mul', par.dsc1pw_bn_mul)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc1pw_bn_bias', par.dsc1pw_bn_bias)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc2_bn_mul', par.dsc2_bn_mul)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc2_bn_bias', par.dsc2_bn_bias)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc2pw_bn_mul', par.dsc2pw_bn_mul)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc2pw_bn_bias', par.dsc2pw_bn_bias)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc3_bn_mul', par.dsc3_bn_mul)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc3_bn_bias', par.dsc3_bn_bias)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc3pw_bn_mul', par.dsc3pw_bn_mul)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc3pw_bn_bias', par.dsc3pw_bn_bias)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc4_bn_mul', par.dsc4_bn_mul)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc4_bn_bias', par.dsc4_bn_bias)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc4pw_bn_mul', par.dsc4pw_bn_mul)
    np.save('/home/zhangs/lyc/hardware_data/' + 'dsc4pw_bn_bias', par.dsc4pw_bn_bias)

    np.save('/home/zhangs/lyc/hardware_data/' + 'FC_bn_mul', par.FC_bn_mul)
    np.save('/home/zhangs/lyc/hardware_data/' + 'FC_bn_bias', par.FC_bn_bias)