# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
from skimage import color,filters
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results//CA_cloud2label_cat', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./dataset/thin_cloudy_dataset//test_A/', help='directory for testing inputs')
parser.add_argument('--adjustment', dest='adjustment', default=False, help='whether to adjust illumination')
parser.add_argument('--ratio', dest='ratio', default=5.0, help='ratio for illumination adjustment')
checkpoint_dir_restoration = './checkpoint/CA_cloud2label_cat/' # 训练模型的保存路径


args = parser.parse_args()

sess = tf.Session()
training = tf.placeholder_with_default(False, shape=(), name='training')
input_low_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_r')


#output_r = Restoration_Unet(input_low_r)
output_r = Channel_attention_net(input_low_r, training=False)

# load pretrained model
var_adjust = [var for var in tf.trainable_variables() if 'I_enhance_Net' in var.name]
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_adjust += bn_moving_vars

saver_restoration = tf.train.Saver(var_list=var_adjust)

ckpt=tf.train.get_checkpoint_state(checkpoint_dir_restoration)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restoration.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No restoration pre model!")

###load eval data
eval_low_data = []
eval_img_name =[]
eval_low_data_name = glob(args.test_dir+'*')                  # 加载测试图像
eval_low_data_name.sort()
for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name.append(name)
    eval_low_im = load_images(eval_low_data_name[idx])
    print(eval_low_im.shape)
    h,w,c = eval_low_im.shape
# the size of test image H and W need to be multiple of 4, if it is not a multiple of 4, we will discard some border pixels.  
    h_tmp = h%4
    w_tmp = w%4
    eval_low_im_resize = eval_low_im[0:h-h_tmp, 0:w-w_tmp, :]
    print(eval_low_im_resize.shape)
    eval_low_data.append(eval_low_im_resize)

sample_dir = args.save_dir 
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

print("Start evalating!")
start_time = time.time()
for idx in range(len(eval_low_data)):
    print(idx)
    name = eval_img_name[idx]
    input_low = eval_low_data[idx]
    input_low_eval = np.expand_dims(input_low, axis=0)
    restoration_r = sess.run(output_r, feed_dict={input_low_r: input_low_eval, training: False})  # 测试阶段training: False


#The restoration result can find more details from very dark regions, however, it will restore the very dark regions
#with gray colors, we use the following operator to alleviate this weakness.  

    save_images(os.path.join(sample_dir, '%s.png' % (name)), restoration_r)

    #save_images(os.path.join(sample_dir, '%s_KinD_plus.png' % (name)), restoration_r)
