import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops

def mixup_data(x,y,alpha = 1):
    if alpha >0:
        lam = np.random.beta(alpha,alpha)
    else:
        lam = 1
    #print("lam"+str(lam))
    batch_size = len(list(np.array(x,dtype=np.float)))
    #print(batch_size)
    index = list(np.arange(batch_size))
    #print(index)
    np.random.shuffle(index)
    #print(index)
    mixed_x = lam*x + (1-lam)*x[index,:]
    mixed_y = lam*y + (1-lam)*y[index,:]
    return mixed_x,mixed_y

def np_round_and_clip(x):
    x = np.round(x)
    x = np.clip(x, -100, 100)
    return x
def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()  # 获取当前默认计算图
    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}): #覆盖当前图中的梯度算法
            #x=tf.clip_by_value(x,-1,1)                      #限制x范围在-1和1之间
            return tf.sign(x)
def binarize_v2(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()  # 获取当前默认计算图
    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}): #覆盖当前图中的梯度算法
            x=tf.clip_by_value(x,-1,1)                      #限制x范围在-1和1之间
            return tf.sign(x)

def ternary(x,thre=0.05):

    g = tf.get_default_graph()  # 获取当前默认计算图
    shape = x.get_shape()
    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)
    #shape = [100,shape_x[1],shape_x[2],shape_x[3]]
    # w_p = tf.get_variable('Wp',trainable=False,collections=[tf.GraphKeys.VARIABLES, 'positives'], initializer=1.0)
    # w_n = tf.get_variable('Wn',trainable=False,collections=[tf.GraphKeys.VARIABLES, 'negatives'], initializer=1.0)

    #tf.scalar_summary(w_p.name, w_p)
    #tf.scalar_summary(w_n.name, w_n)
    w_p = 1
    w_n = 1
    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    with g.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask_z)

    w = w * mask_np

    #tf.histogram_summary(w.name, w)
    return w
def ternary_v2(x,batch):
    thre = 0.05
    g = tf.get_default_graph()  # 获取当前默认计算图
    shape_x= x.get_shape()
    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)
    shape =  tf.stop_gradient([batch,shape_x[1],shape_x[2],shape_x[3]])
    # w_p = tf.get_variable('Wp',trainable=False,collections=[tf.GraphKeys.VARIABLES, 'positives'], initializer=1.0)
    # w_n = tf.get_variable('Wn',trainable=False,collections=[tf.GraphKeys.VARIABLES, 'negatives'], initializer=1.0)

    #tf.scalar_summary(w_p.name, w_p)
    #tf.scalar_summary(w_n.name, w_n)
    w_p = 1
    w_n = 1
    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    with g.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask_z)

    w = w * mask_np

    #tf.histogram_summary(w.name, w)
    return w
def ternary_v3(x):
    x = round_through(x)
    x = tf.clip_by_value(x,-1,1)
    return x

def fourvalue(x,thre=0.05,thre1=0.1):

    g = tf.get_default_graph()  # 获取当前默认计算图
    shape = x.get_shape()
    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)
    thre_x1 = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre1)
    #shape = [100,shape_x[1],shape_x[2],shape_x[3]]
    # w_p = tf.get_variable('Wp',trainable=False,collections=[tf.GraphKeys.VARIABLES, 'positives'], initializer=1.0)
    # w_n = tf.get_variable('Wn',trainable=False,collections=[tf.GraphKeys.VARIABLES, 'negatives'], initializer=1.0)

    #tf.scalar_summary(w_p.name, w_p)
    #tf.scalar_summary(w_n.name, w_n)
    w_p = 0.5
    w_n = 0.5
    w_nn = 1
    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where((x < -thre_x) & (x> -thre_x1), tf.ones(shape) * w_n, mask_p)
    mask_nnp = tf.where(x < -thre_x1, tf.ones(shape) * w_nn, mask_np)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    with g.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask_z)

    w = w * mask_nnp

    #tf.histogram_summary(w.name, w)
    return w
def tw_binarize(x):

    shape = x.get_shape()
    mean,var = tf.nn.moments(x,axes=None)
    sigma = tf.sqrt(var)
    #thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)
    w_p = sigma*1
    w_n = sigma*1
    # w_p = tf.get_variable('Wp', initializer=sigma)
    # w_n = tf.get_variable('Wn', initializer=sigma)
    tf.add_to_collection('positives', w_p)
    tf.add_to_collection('negatives', w_n)
    # tf.summary.scalar(w_p.name, w_p)
    # tf.summary.scalar(w_n.name, w_n)

    mask = tf.where(x > 0, tf.ones(shape) * w_p, tf.ones(shape) * w_n)
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
        x = tf.sign(x)
    x = x*mask + mean

   # tf.histogram_summary(w.name, w)
    return x
def hard_tanh(x):
    return tf.clip_by_value(x,-1,1)
def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = tf.round(x)
    rounded_through = x + tf.stop_gradient(rounded - x)
    return rounded_through
def clip_through(x, min, max):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = tf.clip_by_value(x,min,max)
    return x + tf.stop_gradient(clipped - x)
def quantize(W, nb = 16):

    '''The weights' binarization function,
    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    wb = tf.clip_by_value(round_through(W*m),-m,m-1)/m
    #wb = clip_through(round_through(W*m),-m,m-1)/m
    #wb = tf.Print(wb, [wb], summarize=20)
    return wb
def quantize_noclip(W, nb = 16):

    '''The weights' binarization function,
    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    wb = round_through(W*m)/m
    #wb = clip_through(round_through(W*m),-m,m-1)/m
    #wb = tf.Print(wb, [wb], summarize=20)
    return wb
############################################################################################################
# Convolution layer Methods
def __conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            # out = tf.nn.bias_add(conv, bias)
            out = conv
    return out
def __conv2d_quantize_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,wei=16):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            w_quantize = quantize(w,wei)
            __variable_summaries(w)
        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w_quantize, stride, padding)
            # out = tf.nn.bias_add(conv, bias)
            out = conv
    return out

def __conv2d_bwn_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,bias=0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            w_binarized = binarize(w)
            __variable_summaries(w_binarized)
        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w_binarized, stride, padding)
            #conv = tf.nn.bias_add(conv, bias)

    return conv
def __conv2d_twbwn_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)

            w_binarized = tw_binarize(w)
            __variable_summaries(w_binarized)

        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w_binarized, stride, padding)
            # out = tf.nn.bias_add(conv, bias)

    return conv
def __conv2d_twn_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,the=0.05):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
           # __variable_summaries(w)
            w_ternaried = ternary(w,the)
            __variable_summaries(w_ternaried)
        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w_ternaried, stride, padding)
            # out = tf.nn.bias_add(conv, bias)

    return conv
def __conv2d_fwn_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            w_fourvalue = fourvalue(w)
            __variable_summaries(w_fourvalue)

        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w_fourvalue, stride, padding)
            # out = tf.nn.bias_add(conv, bias)

    return conv

def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True,epsilon = 1e-3):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_p('conv', x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)
        print("After " + name + " output a shape of :" + str(conv_o.get_shape()))
    return conv_o
def conv2d_bwn(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True,epsilon = 1e-3,bias=0):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_bwn_p('conv', x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength,bias=bias)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)
        print("After " + name + " output a shape of :" + str(conv_o.get_shape()))
    return conv_o
def conv2d_quantize(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True,epsilon = 1e-3,bias=0,wei=16):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_quantize_p('conv', x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength,bias=bias,wei=wei)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)
        print("After " + name + " output a shape of :" + str(conv_o.get_shape()))
    return conv_o
def conv2d_twbwn(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True,epsilon = 1e-3):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_twbwn_p('conv', x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)
        print("After " + name + " output a shape of :" + str(conv_o.get_shape()))
    return conv_o
def conv2d_twn(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True,epsilon = 1e-3,the=0.05):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_twn_p('conv', x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength,the=the)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)
        print("After " + name + " output a shape of :" + str(conv_o.get_shape()))
    return conv_o
def conv2d_fwn(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True,epsilon = 1e-3):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_fwn_p('conv', x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)
        print("After " + name + " output a shape of :" + str(conv_o.get_shape()))
    return conv_o
def grouped_conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                   initializer=tf.contrib.layers.xavier_initializer(), num_groups=1, l2_strength=0.0, bias=0.0,
                   activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
                   is_training=True):
    with tf.variable_scope(name) as scope:
        sz = x.get_shape()[3].value // num_groups
        conv_side_layers = [
            conv2d(name + "_" + str(i), x[:, :, :, i * sz:i * sz + sz], w, num_filters // num_groups, kernel_size,
                   padding,
                   stride,
                   initializer,
                   l2_strength, bias, activation=None,
                   batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=dropout_keep_prob,
                   is_training=is_training) for i in
            range(num_groups)]
        conv_g = tf.concat(conv_side_layers, axis=-1)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_g, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_g
            else:
                conv_a = activation(conv_g)

        return conv_a
def grouped_conv2d_bwn(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                   initializer=tf.contrib.layers.xavier_initializer(), num_groups=1, l2_strength=0.0,
                   activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
                   is_training=True,epsilon = 1e-3):
    with tf.variable_scope(name) as scope:
        sz = x.get_shape()[3].value // num_groups
        conv_side_layers = [
            conv2d_bwn(name + "_" + str(i), x[:, :, :, i * sz:i * sz + sz], w, num_filters // num_groups, kernel_size,
                   padding,
                   stride,
                   initializer,
                   l2_strength, activation=None,
                   batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=dropout_keep_prob,
                   is_training=is_training,epsilon = 1e-3) for i in range(num_groups)]
        conv_g = tf.concat(conv_side_layers, axis=-1)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_g, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_g
            else:
                conv_a = activation(conv_g)

        return conv_a
def grouped_conv2d_twn(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                   initializer=tf.contrib.layers.xavier_initializer(), num_groups=1, l2_strength=0.0,
                   activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
                   is_training=True,epsilon = 1e-3):
    with tf.variable_scope(name) as scope:
        sz = x.get_shape()[3].value // num_groups
        conv_side_layers = [
            conv2d_twn(name + "_" + str(i), x[:, :, :, i * sz:i * sz + sz], w, num_filters // num_groups, kernel_size,
                   padding,
                   stride,
                   initializer,
                   l2_strength, activation=None,
                   batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=dropout_keep_prob,
                   is_training=is_training,epsilon = 1e-3) for i in range(num_groups)]
        conv_g = tf.concat(conv_side_layers, axis=-1)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_g, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_g
            else:
                conv_a = activation(conv_g)

        return conv_a
def __depthwise_conv2d_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w, stride, padding)
            # out = tf.nn.bias_add(conv, bias)
            out = conv
    return out
def __depthwise_conv2d_bwn_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            w_binarized = binarize(w)
            __variable_summaries(w_binarized)

        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w_binarized, stride, padding)
            # out = tf.nn.bias_add(conv, bias)

    return conv
def __depthwise_conv2d_quantize_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,wei=16):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            w_quantize = quantize(w,wei)
            __variable_summaries(w_quantize)

        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w_quantize, stride, padding)
            # out = tf.nn.bias_add(conv, bias)

    return conv
def __depthwise_conv2d_twn_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,the=0.05):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            w_ternaried = ternary(w,the)
            __variable_summaries(w_ternaried)
        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w_ternaried, stride, padding)
            # out = tf.nn.bias_add(conv, bias)

    return conv
def __depthwise_conv2d_fwn_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            w_fourvalue = fourvalue(w)
            __variable_summaries(w_fourvalue)

        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w_fourvalue, stride, padding)
            # out = tf.nn.bias_add(conv, bias)

    return conv
def __depthwise_conv2d_twbwn_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)

            w_binarized = tw_binarize(w)
            __variable_summaries(w_binarized)

        # with tf.name_scope('layer_biases'):
        #     if isinstance(bias, float):
        #         bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
        #     __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w_binarized, stride, padding)
            # out = tf.nn.bias_add(conv, bias)

    return conv
def depthwise_conv2d(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True,epsilon = 1e-3):
    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_p(name='conv', x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)

            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    print("After " + name + " output a shape of :" + str(conv_a.get_shape()))

    return conv_a
def depthwise_conv2d_bwn(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True,epsilon = 1e-3):
    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_bwn_p(name='conv', x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)

            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    print("After " + name + " output a shape of :" + str(conv_a.get_shape()))

    return conv_a
def depthwise_conv2d_quantize(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True,epsilon = 1e-3,wei=16):
    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_quantize_p(name='conv', x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength,wei=wei)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)

            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    print("After " + name + " output a shape of :" + str(conv_a.get_shape()))

    return conv_a
def depthwise_conv2d_twbwn(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True,epsilon = 1e-3):
    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_twbwn_p(name='conv', x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)

            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    print("After " + name + " output a shape of :" + str(conv_a.get_shape()))

    return conv_a
def depthwise_conv2d_twn(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True,epsilon = 1e-3,the=0.05):
    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_twn_p(name='conv', x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength,the=the)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)

            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    print("After " + name + " output a shape of :" + str(conv_a.get_shape()))

    return conv_a
def depthwise_conv2d_fwn(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True,epsilon = 1e-3):
    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_fwn_p(name='conv', x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength)

        if batchnorm_enabled:
            #conv_o_bn = batch_normalization_layer(conv_o_b, isTrain=is_training)

            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    print("After " + name + " output a shape of :" + str(conv_a.get_shape()))

    return conv_a
############################################################################################################
# ShuffleNet unit methods

def shufflenet_unit(name, x, w=None, num_groups=1, group_conv_bottleneck=True, num_filters=16, stride=(1, 1),
                    l2_strength=0.0, bias=0.0, batchnorm_enabled=True, is_training=True, fusion='add'):
    # Paper parameters. If you want to change them feel free to pass them as method parameters.
    activation = tf.nn.relu

    with tf.variable_scope(name) as scope:
        residual = x
        bottleneck_filters = (num_filters // 4) if fusion == 'add' else (num_filters - residual.get_shape()[
            3].value) // 4

        if group_conv_bottleneck:
            bottleneck = grouped_conv2d('Gbottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                        padding='VALID',
                                        num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                        activation=activation,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            shuffled = channel_shuffle('channel_shuffle', bottleneck, num_groups)
        else:
            bottleneck = conv2d('bottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                padding='VALID', l2_strength=l2_strength, bias=bias, activation=activation,
                                batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            shuffled = bottleneck
        padded = tf.pad(shuffled, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        depthwise = depthwise_conv2d('depthwise', x=padded, w=None, stride=stride, l2_strength=l2_strength,
                                     padding='VALID', bias=bias,
                                     activation=None, batchnorm_enabled=batchnorm_enabled, is_training=is_training)
        if stride == (2, 2):
            residual_pooled = avg_pool_2d(residual, size=(3, 3), stride=stride, padding='SAME')
        else:
            residual_pooled = residual

        if fusion == 'concat':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters - residual.get_shape()[3].value,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(tf.concat([residual_pooled, group_conv1x1], axis=-1))
        elif fusion == 'add':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            residual_match = residual_pooled
            # This is used if the number of filters of the residual block is different from that
            # of the group convolution.
            if num_filters != residual_pooled.get_shape()[3].value:
                residual_match = conv2d('residual_match', x=residual_pooled, w=None, num_filters=num_filters,
                                        kernel_size=(1, 1),
                                        padding='VALID', l2_strength=l2_strength, bias=bias, activation=None,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(group_conv1x1 + residual_match)
        else:
            raise ValueError("Specify whether the fusion is \'concat\' or \'add\'")


def channel_shuffle(name, x, num_groups):
    with tf.variable_scope(name) as scope:
        n, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
        return output


############################################################################################################
# Fully Connected layer Methods

def __dense_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
              bias=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        __variable_summaries(w)
        # if isinstance(bias, float):
        #     bias = tf.get_variable("layer_biases", [output_dim], tf.float32, tf.constant_initializer(bias))
        # __variable_summaries(bias)
        # output = tf.nn.bias_add(tf.matmul(x, w), bias)
        output = tf.matmul(x, w)

        return output
def __dense_bwn_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        w_binarized = binarize(w)
        __variable_summaries(w_binarized)

        output = tf.matmul(x, w_binarized)
        return output
def __dense_quantize_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,wei=16):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        w_quanzize = quantize(w,wei)
        __variable_summaries(w_quanzize)

        output = tf.matmul(x, w_quanzize)
        return output

def __dense_twbwn_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        __variable_summaries(w)

        w_binarized = tw_binarize(w)
        __variable_summaries(w_binarized)

        output = tf.matmul(x, w_binarized)
        return output

def __dense_twn_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,the=0.05):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        w_ternaried = ternary(w,the)
        __variable_summaries(w_ternaried)
        output = tf.matmul(x, w_ternaried)
        return output
def __dense_fwn_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        w_fourvalue = fourvalue(w)
        __variable_summaries(w_fourvalue)
        output = tf.matmul(x, w_fourvalue)
        return output

def dense(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True,epsilon = 1e-3
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = __dense_p(name='dense', x=x, w=w, output_dim=output_dim, initializer=initializer,
                              l2_strength=l2_strength,
                              bias=bias)

        if batchnorm_enabled:
            #dense_o_bn = batch_normalization_layer(dense_o_b, isTrain=is_training)

            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr

        print("After " + name + " output a shape of :" + str(dense_o.get_shape()))

    return dense_o
def dense_quantize(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True,epsilon = 1e-3,wei=16
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = __dense_quantize_p(name='dense', x=x, w=w, output_dim=output_dim, initializer=initializer,
                                l2_strength=l2_strength,wei=wei)

        if batchnorm_enabled:
            #dense_o_bn = batch_normalization_layer(dense_o_b, isTrain=is_training)

            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr

        print("After " + name + " output a shape of :" + str(dense_o.get_shape()))

    return dense_o
def dense_bwn(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True,epsilon = 1e-3
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = __dense_bwn_p(name='dense', x=x, w=w, output_dim=output_dim, initializer=initializer,
                              l2_strength=l2_strength)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr

        print("After " + name + " output a shape of :" + str(dense_o.get_shape()))

    return dense_o

def dense_twbwn(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True,epsilon = 1e-3
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = __dense_twbwn_p(name='dense', x=x, w=w, output_dim=output_dim, initializer=initializer,
                              l2_strength=l2_strength)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr

        print("After " + name + " output a shape of :" + str(dense_o.get_shape()))

    return dense_o
def dense_twn(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True,epsilon = 1e-3,the=0.05
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = __dense_twn_p(name='dense', x=x, w=w, output_dim=output_dim, initializer=initializer,
                              l2_strength=l2_strength,the=the)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr

        print("After " + name + " output a shape of :" + str(dense_o.get_shape()))

    return dense_o
def dense_fwn(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True,epsilon = 1e-3
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = __dense_fwn_p(name='dense', x=x, w=w, output_dim=output_dim, initializer=initializer,
                              l2_strength=l2_strength)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=epsilon)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr

        print("After " + name + " output a shape of :" + str(dense_o.get_shape()))

    return dense_o
def flatten(x):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first])
    print("After " + 'Flatten' + " output a shape of :" + str(o.get_shape()))

    return o


############################################################################################################
# Pooling Methods

def max_pool_2d(x, size=(2, 2), stride=(2, 2), name='max_pooling',padding='VALID'):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :param name: (string) Scope name.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding=padding,
                          name=name)


def avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID'):
    """
        Average pooling 2D Wrapper
        :param x: (tf.tensor) The input to the layer (N,H,W,C).
        :param size: (tuple) This specifies the size of the filter as well as the stride.
        :param name: (string) Scope name.
        :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    o = tf.nn.avg_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding=padding,
                          name=name)
    print("After " + 'avgpool' + " output a shape of :" + str(o.get_shape()))

    return o

def batchnormalization(x,is_training=True,epsilon=1e-3):
    return  tf.layers.batch_normalization(x, training=is_training, epsilon=epsilon)

def batchnormalization_my(inputs,is_training=True,name='batchnorm_my'):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    with tf.variable_scope(name):
        axis = list(range(len(inputs.get_shape()) - 1))

        mean, variance = tf.nn.moments(inputs, axis)

        beta = tf.get_variable('beta', initializer=tf.zeros_initializer, shape=inputs.get_shape()[-1], dtype=tf.float32)
        gamma = tf.get_variable('gamma', initializer=tf.ones_initializer, shape=inputs.get_shape()[-1],dtype=tf.float32)

        moving_mean = tf.get_variable(name='moving_mean', initializer=tf.zeros_initializer,
                                      shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)
        moving_variance = tf.get_variable(name='moving_variance', initializer=tf.zeros_initializer,
                                          shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)
        #update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.999,zero_debias=True) ##
        #update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.999,zero_debias=True)##
        mul = tf.divide(gamma,tf.sqrt(tf.add(moving_variance,0.001)))
        bias = tf.add(tf.multiply(tf.multiply(mul,-1),moving_mean),beta)

        mul_train = tf.divide(gamma,tf.sqrt(tf.add(variance,0.001)))
        bias_train = tf.add(tf.multiply(tf.multiply(mul_train,-1),mean),beta)
        with tf.name_scope('summaries'):
            # mean = tf.reduce_mean(var)
            tf.summary.histogram(name + '_global_mul', mul)
            tf.summary.histogram(name + '_global_bias', bias)
            tf.summary.histogram(name + '_batch_mul', mul_train)
            tf.summary.histogram(name + '_batch_bias', bias_train)
        def batch_norm_training():
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.999,zero_debias=True) ##
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.999,zero_debias=True)##
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):  ####
                tf.add_to_collection('mean', update_moving_mean)
                tf.add_to_collection('variance', update_moving_variance)
                return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)
        def batch_norm_inference():
            mean = moving_mean  ## update_moving_mean
            variance = moving_variance  ##
            return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)

        out = tf.cond(tf.cast(is_training,tf.bool), batch_norm_training, batch_norm_inference)

        return out


def batchnormalization_quantize(inputs, is_training=True, name='batchnorm_my'):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    with tf.variable_scope(name):
        axis = list(range(len(inputs.get_shape()) - 1))

        mean, variance = tf.nn.moments(inputs, axis)

        beta = tf.get_variable('beta', initializer=tf.zeros_initializer, shape=inputs.get_shape()[-1], dtype=tf.float32)
        gamma = tf.get_variable('gamma', initializer=tf.ones_initializer, shape=inputs.get_shape()[-1],
                                dtype=tf.float32)

        moving_mean = tf.get_variable(name='moving_mean', initializer=tf.zeros_initializer,
                                      shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)
        moving_variance = tf.get_variable(name='moving_variance', initializer=tf.zeros_initializer,
                                          shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)


        mul = tf.divide(gamma,tf.sqrt(tf.add(moving_variance,0.001)))
        bias = tf.add(tf.multiply(tf.multiply(mul,-1.0),moving_mean),beta)

        mul_train = tf.divide(gamma,tf.sqrt(tf.add(variance,0.001)))
        bias_train = tf.add(tf.multiply(tf.multiply(mul_train,-1.0),mean),beta)

        with tf.name_scope('summaries'):
            # mean = tf.reduce_mean(var)
            tf.summary.histogram(name + '_global_mul', mul)
            tf.summary.histogram(name + '_global_bias', bias)
            tf.summary.histogram(name + '_batch_mul', mul_train)
            tf.summary.histogram(name + '_batch_bias', bias_train)

        def batch_norm_training():
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.999, zero_debias=True)  ##
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.999,zero_debias=True)  ##
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):  ####
                tf.add_to_collection('mean', update_moving_mean)
                tf.add_to_collection('variance', update_moving_variance)
                return tf.add(tf.multiply(inputs,mul_train),bias_train)

        def batch_norm_inference():
            #return tf.add(tf.multiply(inputs,quantize(mul,3)),quantize(bias,3))
            return tf.add(tf.multiply(inputs,mul),bias)


        out = tf.cond(tf.cast(is_training, tf.bool), batch_norm_training, batch_norm_inference)

        return out

def batchnormalization_hardware(inputs, is_training=True, name='batchnorm_my'):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    with tf.variable_scope(name):
        axis = list(range(len(inputs.get_shape()) - 1))

        mean, variance = tf.nn.moments(inputs, axis)

        beta = tf.get_variable('beta', initializer=tf.zeros_initializer, shape=inputs.get_shape()[-1], dtype=tf.float32)
        gamma = tf.get_variable('gamma', initializer=tf.ones_initializer, shape=inputs.get_shape()[-1],
                                dtype=tf.float32)

        moving_mean = tf.get_variable(name='moving_mean', initializer=tf.zeros_initializer,
                                      shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)
        moving_variance = tf.get_variable(name='moving_variance', initializer=tf.zeros_initializer,
                                          shape=inputs.get_shape()[-1], dtype=tf.float32, trainable=False)

        divider = tf.sqrt(tf.add(moving_variance, 0.001))
        mul = tf.divide(gamma,divider)
        bias = tf.add(tf.multiply(-1.0,moving_mean),tf.divide(beta,mul))

        divider_train = tf.sqrt(tf.add(variance, 0.001))
        mul_train = tf.divide(gamma,divider_train)
        bias_train = tf.add(tf.multiply(-1.0,mean),tf.divide(beta,mul_train))


        #mul = mul + quantize(tf.stop_gradient(tf.round(mul)-mul),8) #小数部分限定精度

        with tf.name_scope('summaries'):
            # mean = tf.reduce_mean(var)
            tf.summary.histogram(name + '_global_mul', mul)
            tf.summary.histogram(name + '_global_bias', bias)
            tf.summary.histogram(name + '_batch_mul', mul_train)
            tf.summary.histogram(name + '_batch_bias', bias_train)

            # tf.summary.scalar(name + '_global_mul_min', tf.reduce_min(mul))
            # tf.summary.scalar(name + '_global_mul_max', tf.reduce_max(mul))
            # tf.summary.scalar(name + '_global_bias_min', tf.reduce_min(bias))
            # tf.summary.scalar(name + '_global_bias_max', tf.reduce_max(bias))

        def batch_norm_training():
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.999, zero_debias=True)  ##
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.999,zero_debias=True)  ##
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):  ####
                tf.add_to_collection('mean', update_moving_mean)
                tf.add_to_collection('variance', update_moving_variance)
                return tf.multiply(tf.add(inputs,bias_train),mul_train)

        def batch_norm_inference():
            return tf.multiply(tf.add(inputs,bias),mul)


        out = tf.cond(tf.cast(is_training, tf.bool), batch_norm_training, batch_norm_inference)

        return out
def relu(x):
    return tf.nn.relu(x,'relu')
def sigmoid(x):
    return tf.nn.sigmoid(x,'sigmoid')
############################################################################################################
# Utilities for layers

def __variable_with_weight_decay(kernel_shape, initializer, wd):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    print(w.name + " have a shape of : " + str(w.get_shape()))
    return w


# Summaries for variables
def __variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    with tf.name_scope('summaries'):
        #mean = tf.reduce_mean(var)
        #tf.summary.scalar('mean', mean)
        #with tf.name_scope('stddev'):
        #    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        #tf.summary.scalar('stddev', stddev)
        #tf.summary.scalar('max', tf.reduce_max(var))
        #tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
def __variable_summaries_scale(var,name):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    with tf.name_scope('summaries'):
        #mean = tf.reduce_mean(var)
        tf.summary.scalar(name, var)
 

