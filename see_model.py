import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

'''''
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model.ckpt.meta') #load graph
    for var in tf.trainable_variables(): #get the param names
        print (var.name) #print parameters' names
        #new_saver.restore(sess, tf.train.latest_checkpoint('./')) #find the newest training result
        all_vars = tf.trainable_variables()
        #for v in all_vars:
        #   v_4d = np.array(sess.run(v)) #get the real parameters
'''''
def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()  # 获取当前默认计算图
    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}): #覆盖当前图中的梯度算法
            x=tf.clip_by_value(x,-1,1)                      #限制x范围在-1和1之间
            return tf.sign(x)
#with tf.Session() as sess:
   # bin_w = binarize(w1)
    #sess.run(bin_w)
   # print(type(bin_w))
   # print(bin_w.shape)
   # print(bin_w.eval())
def draw_dnn():
    reader = tf.train.NewCheckpointReader("/home/zhangs/tensorflow-master/tensorflow/examples/speech/tmp/dnn128_650ms_50_20/dnn.ckpt-16000")
    all_variables = reader.get_variable_to_shape_map()
    w1 = reader.get_tensor("L1_weights")
    w2 = reader.get_tensor("L2_weights")
    w3 = reader.get_tensor("L3_weights")
    w4 = reader.get_tensor("Lout_weights")

    b1 = reader.get_tensor("L1_bias")
    b2 = reader.get_tensor("L2_bias")
    b3 = reader.get_tensor("L3_bias")
    b4 = reader.get_tensor("Lout_bias")
    # print(w1)
    # print(w1.shape)
    w1 = w1.flatten()
    w2 = w2.flatten()
    w3 = w3.flatten()
    w4 = w4.flatten()

    b1 = b1.flatten()
    b2 = b2.flatten()
    b3 = b3.flatten()
    b4 = b4.flatten()

    plt.subplot(241)
    plt.xlabel('L1_weights')
    plt.ylabel('probability')
    plt.title('Dnn128*3 Layer_1')
    plt.hist(w1, bins=60, color='steelblue', normed=True)

    plt.subplot(242)
    plt.xlabel('L2_weights')
    plt.ylabel('probability')
    plt.title('Dnn128*3 Layer_2')
    plt.hist(w2, bins=60, color='steelblue', normed=True)

    plt.subplot(243)
    plt.xlabel('L3_weights')
    plt.ylabel('probability')
    plt.title('Dnn128*3 Layer_3')
    plt.hist(w3, bins=60, color='steelblue', normed=True)

    plt.subplot(244)
    plt.xlabel('L4_weights')
    plt.ylabel('probability')
    plt.title('Dnn128*3 Layer_4')
    plt.hist(w4, bins=60, color='steelblue', normed=True)

    plt.subplot(245)
    plt.xlabel('L1_bias')
    plt.ylabel('probability')
    # plt.title('Dnn128*3 L1_bias')
    plt.hist(b1, bins=40, color='steelblue', normed=True)

    plt.subplot(246)
    plt.xlabel('L2_bias')
    plt.ylabel('probability')
    # plt.title('Dnn128*3 L2_bias')
    plt.hist(b2, bins=40, color='steelblue', normed=True)

    plt.subplot(247)
    plt.xlabel('L3_bias')
    plt.ylabel('probability')
    # plt.title('Dnn128*3 L3_bias')
    plt.hist(b3, bins=40, color='steelblue', normed=True)

    plt.subplot(248)
    plt.xlabel('L4_bias')
    plt.ylabel('probability')
    # plt.title('Dnn128*3 L3_bias')
    plt.hist(b4, bins=40, color='steelblue', normed=True)


def draw_dnn_bnn():
    reader = tf.train.NewCheckpointReader("/home/zhangs/tensorflow-master/tensorflow/examples/speech/tmp/speech_commands_train/dnn_bnn_retrain.ckpt-11300")
    all_variables = reader.get_variable_to_shape_map()
    w1 = reader.get_tensor("L1_weights")
    w2 = reader.get_tensor("L2_weights")
    w3 = reader.get_tensor("L3_weights")
    w4 = reader.get_tensor("Lout_weights")

    w1 = w1.flatten()
    w2 = w2.flatten()
    w3 = w3.flatten()
    w4 = w4.flatten()
    plt.subplot(141)
    plt.xlabel('L1_weights')
    plt.ylabel('probability')
    plt.title('Dnn_bnn_128*3 Layer_1')
    plt.hist(w1, bins=60, color='steelblue', normed=True)

    plt.subplot(142)
    plt.xlabel('L2_weights')
    plt.ylabel('probability')
    plt.title('Dnn_bnn_128*3 Layer_2')
    plt.hist(w2, bins=60, color='steelblue', normed=True)

    plt.subplot(143)
    plt.xlabel('L3_weights')
    plt.ylabel('probability')
    plt.title('Dnn_bnn_128*3 Layer_3')
    plt.hist(w3, bins=60, color='steelblue', normed=True)

    plt.subplot(144)
    plt.xlabel('L4_weights')
    plt.ylabel('probability')
    plt.title('Dnn_bnn_128*3 Layer_4')
    plt.hist(w4, bins=40, color='steelblue', normed=True)
plt.figure(0)
draw_dnn()
plt.figure(1)
draw_dnn_bnn()
plt.show()
reader = tf.train.NewCheckpointReader("/home/zhangs/tensorflow-master/tensorflow/examples/speech/tmp/dnn128_650ms_50_20/dnn.ckpt-16000")
#all_variables = reader.get_variable_to_shape_map()
w1 = reader.get_tensor("L1_weights")
w2 = reader.get_tensor("L2_weights")
w3 = reader.get_tensor("L3_weights")
w4 = reader.get_tensor("Lout_weights")
#
# hidden_layer1 = 128
# hidden_layer2 = 128
# hidden_layer3 = 128
# fingerprint_size = 450
# label_count = 4
#
# layer1_weights = tf.Variable(
#     w1, name="L1_weights")
#
# layer2_weights = tf.Variable(
#     w2, name="L2_weights")
#
# layer3_weights = tf.Variable(
#     w3, name="L3_weights")
#
# output_weights = tf.Variable(
#     w4, name="Lout_weights")
#
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(layer1_weights.eval()==w1)
#     print(layer2_weights.eval()==w2)
#     print(layer3_weights.eval()==w3)
#     print(output_weights.eval()==w4)
w1_variance = np.sqrt(np.var(w1))
w2_variance = np.sqrt(np.var(w2))
w3_variance = np.sqrt(np.var(w3))
w4_variance = np.sqrt(np.var(w4))
w1_mean = np.mean(w1)
w2_mean = np.mean(w2)
w3_mean = np.mean(w3)
w4_mean = np.mean(w4)
print("w1_mean = "+str(w1_mean))
print("w2_mean = "+str(w2_mean))
print("w3_mean = "+str(w3_mean))
print("w4_mean = "+str(w4_mean))
print("w1_variance = "+str(w1_variance))
print("w2_variance = "+str(w2_variance))
print("w3_variance = "+str(w3_variance))
print("w4_variance = "+str(w4_variance))

print("w1_mean - 3 * sigma1="+str(w1_mean - 3 * w1_variance))
print("w1_mean + 3 * sigma1="+str( w1_mean + 3 * w1_variance))
print("w2_mean - 3 * sigma2="+str(w2_mean - 3 * w2_variance))
print("w2_mean + 3 * sigma2="+str( w2_mean + 3 * w2_variance))
print("w3_mean - 3 * sigma3="+str(w3_mean - 3 * w3_variance))
print("w3_mean + 3 * sigma3="+str( w3_mean + 3 * w3_variance))
print("w4_mean - 3 * sigma4="+str(w4_mean - 3 * w4_variance))
print("w4_mean + 3 * sigma4="+str( w4_mean + 3 * w4_variance))
print('\n')
print("w1_mean - 2 * sigma1="+str(w1_mean - 2 * w1_variance))
print("w1_mean + 2 * sigma1="+str( w1_mean + 2 * w1_variance))
print("w2_mean - 2 * sigma2="+str(w2_mean - 2 * w2_variance))
print("w2_mean + 2 * sigma2="+str( w2_mean + 2 * w2_variance))
print("w3_mean - 2 * sigma3="+str(w3_mean - 2 * w3_variance))
print("w3_mean + 2 * sigma3="+str( w3_mean + 2 * w3_variance))
print("w4_mean - 2 * sigma4="+str(w4_mean - 2 * w4_variance))
print("w4_mean + 2 * sigma4="+str( w4_mean + 2 * w4_variance))
print('\n')
print("w1_mean - 1 * sigma1="+str(w1_mean - 1 * w1_variance))
print("w1_mean + 1 * sigma1="+str( w1_mean + 1 * w1_variance))
print("w2_mean - 1 * sigma2="+str(w2_mean - 1 * w2_variance))
print("w2_mean + 1 * sigma2="+str( w2_mean + 1 * w2_variance))
print("w3_mean - 1 * sigma3="+str(w3_mean - 1 * w3_variance))
print("w3_mean + 1 * sigma3="+str( w3_mean + 1 * w3_variance))
print("w4_mean - 1 * sigma4="+str(w4_mean - 1 * w4_variance))
print("w4_mean + 1 * sigma4="+str( w4_mean + 1 * w4_variance))
x = tf.clip_by_value(w1, w1_mean - 3 * w1_variance, w1_mean + 3 * w1_variance)  # 限制x范围在u-3sigma和u+3sigma之间
xx = w1_mean + 3 * w1_variance * tf.sign(x - w1_mean)
with tf.Session() as sess:
    sess.run(xx)
    print(xx.eval())
    #print()