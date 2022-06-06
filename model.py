import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers


def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)


def RCAB_withBN(input, reduction, name):
    """
    @Image super-resolution using very deep residual channel attention networks
    Residual Channel Attention Block
    """
    batch, height, width, channel = input.get_shape()  # (B, W, H, C)
    feature_maps1 = tf.layers.conv2d(input, channel, 3, padding='same',  name=name + 'con1')  # (B, W, H, C)
    bn1 = tf.layers.batch_normalization(feature_maps1, axis=-1, training=True, name=name + 'bn1')
    relu1 = tf.nn.relu(bn1, name=name + 're1')
    feature_maps2 = tf.layers.conv2d(relu1, channel, 3, padding='same', name=name + 'con2')  # (B, W, H, C)
    bn2 = tf.layers.batch_normalization(feature_maps2, axis=-1, training=True, name=name + 'bn2')

    global_pooling_weights = tf.reduce_mean(bn2, axis=(1, 2), keepdims=True, name=name + 'gp')  # (B, 1, 1, C)
    shrinking_weights = tf.layers.conv2d(global_pooling_weights, channel // reduction, 1, activation=tf.nn.relu,
                                         name=name + 'sw')  # (B, 1, 1, C // r)
    extending_weights = tf.layers.conv2d(shrinking_weights, channel, 1, activation=tf.nn.sigmoid,
                                         name=name + 'ew')  # (B, 1, 1, C)
    channel_attention_maps = tf.multiply(feature_maps2, extending_weights, name=name + 'cam')  # (B, W, H, C)
    channel_attention_output = tf.add(input, channel_attention_maps, name=name + 'cao')
    return channel_attention_output


def RCAB(input, reduction, name):
    """
    @Image super-resolution using very deep residual channel attention networks
    Residual Channel Attention Block
    """
    batch, height, width, channel = input.get_shape()  # (B, W, H, C)
    feature_maps1 = tf.layers.conv2d(input, channel, 3, padding='same', activation=tf.nn.relu, name=name+'con1')  # (B, W, H, C)
    feature_maps2 = tf.layers.conv2d(feature_maps1, channel, 3, padding='same', name=name+'con2')  # (B, W, H, C)

    global_pooling_weights = tf.reduce_mean(feature_maps2, axis=(1, 2), keepdims=True, name=name+'gp')  # (B, 1, 1, C)
    shrinking_weights = tf.layers.conv2d(global_pooling_weights, channel // reduction, 1, activation=tf.nn.relu, name=name+'sw')  # (B, 1, 1, C // r)
    extending_weights = tf.layers.conv2d(shrinking_weights, channel, 1, activation=tf.nn.sigmoid, name=name+'ew')  # (B, 1, 1, C)
    channel_attention_maps = tf.multiply(feature_maps2, extending_weights, name=name+'cam')  # (B, W, H, C)
    channel_attention_output = tf.add(input, channel_attention_maps, name=name+'cao')
    return channel_attention_output

def Residual_Group(input,training, name):
    RCAB_1 = RCAB(input, 16, name=name+'RCAB_1')
    RCAB_2 = RCAB(RCAB_1, 16, name=name+'RCAB_2')
    RCAB_3 = RCAB(RCAB_2, 16, name=name+'RCAB_3' )
    RCAB_4 = RCAB(RCAB_3, 16, name=name+'RCAB_4')
    RCAB_5 = RCAB(RCAB_4, 16, name=name + 'RCAB_5')
    #RCAB_6 = RCAB(RCAB_5, 16, name=name + 'RCAB_6')

    Residual_output = tf.layers.conv2d(RCAB_5, 64, 3, padding='same')

    RG_output = tf.add(input, Residual_output)

    return RG_output


def Channel_attention_net(input, training):
    with tf.variable_scope('I_enhance_Net', reuse=tf.AUTO_REUSE):
        input_feature_maps = tf.layers.conv2d(input, 64, 3, padding='same')
        #pool_maps = slim.max_pool2d(input_feature_maps, [2, 2], [2, 2], padding='SAME')
        RG_E1 = Residual_Group(input_feature_maps, training=training, name='RG_E1')
        RG_E2 = Residual_Group(RG_E1, training=training, name='RG_E2')
        RG_E3 = Residual_Group(RG_E2, training=training, name='RG_E3')

        RG_E2D = Residual_Group(RG_E3, training=training, name='RG_E2D')
        D_conv3 = tf.concat([RG_E2D, RG_E3], 3)
        D_conv3 = tf.layers.conv2d(D_conv3, 64, 3, padding='same', name='D_con3')
        RG_D3 = Residual_Group(D_conv3, training=training, name='RG_D3')
        D_conv2 = tf.concat([RG_D3, RG_E2], 3)
        D_conv2 = tf.layers.conv2d(D_conv2, 64, 3, padding='same', name='D_con2')
        RG_D2 = Residual_Group(D_conv2, training=training, name='RG_D2')
        D_conv1 = tf.concat([RG_D2, RG_E1], 3)
        D_conv1 = tf.layers.conv2d(D_conv1, 64, 3, padding='same', name='D_con1')
        RG_D1 = Residual_Group(D_conv1, training=training, name='RG_D1')  # 得到去掉了低频的云层，增强了高频的细节的特征图

        output = tf.add(input_feature_maps, RG_D1)
        output = tf.layers.conv2d(output, 3, 3, padding='same')
        out = tf.sigmoid(output)
    return out


### Whether the channel attention is valiad?.....................

def RCAB_noCA(input, reduction,training):
    batch, height, width, channel = input.get_shape()  # (B, W, H, C)
    feature_maps = tf.layers.conv2d(input, channel, 3, padding='same', activation=tf.nn.relu)  # (B, W, H, C)
    feature_maps = tf.layers.conv2d(feature_maps, channel, 3, padding='same')  # (B, W, H, C)

    channel_attention_output = tf.add(input, feature_maps)

    return channel_attention_output

def Residual_Group_noCA(input,training):
    RCAB_1 = RCAB_noCA(input, 16, training=training)
    RCAB_2 = RCAB_noCA(RCAB_1, 16, training=training)
    RCAB_3 = RCAB_noCA(RCAB_2, 16, training=training)
    RCAB_4 = RCAB_noCA(RCAB_3, 16, training=training)
    RCAB_5 = RCAB(RCAB_4, 16, training=training)
    # RCAB_6 = RCAB(RCAB_5, 16, training=training)
    # RCAB_7 = RCAB(RCAB_6, 16, training=training)
    # RCAB_8 = RCAB(RCAB_7, 16, training=training)
    # RCAB_9 = RCAB(RCAB_8, 16, training=training)
    # RCAB_10 = RCAB(RCAB_9, 16, training=training)
    Residual_output = tf.layers.conv2d(RCAB_5, 64, 3, padding='same')

    RG_output = tf.add(input, Residual_output)

    return RG_output

def Channel_attention_net_noCA(input, training):
    with tf.variable_scope('I_enhance_Net', reuse=tf.AUTO_REUSE):
        input_feature_maps = tf.layers.conv2d(input, 64, 3, padding='same')
        #pool_maps = slim.max_pool2d(input_feature_maps, [2, 2], [2, 2], padding='SAME')
        RG_E1 = Residual_Group_noCA(input_feature_maps, training=training)
        RG_E2 = Residual_Group_noCA(RG_E1, training=training)
        RG_E3 = Residual_Group_noCA(RG_E2, training=training)

        RG_D3 = Residual_Group_noCA(RG_E3, training=training)
        D_conv3 = tf.concat([RG_D3, RG_E3], 3)
        D_conv3 = tf.layers.conv2d(D_conv3, 64, 3, padding='same')
        RG_D2 = Residual_Group_noCA(D_conv3, training=training)
        D_conv2 = tf.concat([RG_D2, RG_E2], 3)
        D_conv2 = tf.layers.conv2d(D_conv2, 64, 3, padding='same')
        RG_D1 = Residual_Group_noCA(D_conv2, training=training)
        D_conv1 = tf.concat([RG_D1, RG_E1], 3)
        D_conv1 = tf.layers.conv2d(D_conv1, 64, 3, padding='same')
        output_feature_maps =  Residual_Group_noCA(D_conv1, training=training)

        output = tf.add(input_feature_maps, output_feature_maps)
        output = tf.layers.conv2d(output, 3, 3, padding='same')
        out = tf.sigmoid(output)

    return out


