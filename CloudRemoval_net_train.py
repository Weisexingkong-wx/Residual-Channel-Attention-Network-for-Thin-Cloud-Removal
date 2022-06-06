# coding: utf-8
from __future__ import print_function
import os
import time
import random
# from skimage import color

from utils import *
from model import *
from glob import glob
import time
from datetime import timedelta

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
training = tf.placeholder_with_default(False, shape=(), name='training')
batch_size = 8
patch_size = 64
eval_sample_nums = 70           # 从训练集中抽取验证集的数目
learning_rate = 0.0001
epoch = 900
eval_every_epoch = 20        # 每验证一次经历的迭代次数
h, w = 512, 512

checkpoint_dir = './checkpoint/CA_cloud2label_cat/'  # 训练模型的保存路径
sample_dir = './eval_samples/eval_IlluNet_samples/CA_cloud2label_cat//' # 训练中间过程输出的去云图像
trainLoss_txt = "txt文档的绝对路径"   # 训练中间过程生成的记录验证集损失函数的文档

#the input of illumination adjustment ne   t
input_cloud_i = tf.placeholder(tf.float32, [None, None, None, 3], name='input_cloud_i')
input_label_i = tf.placeholder(tf.float32, [None, None, None, 3], name='input_label_i')

output_H = Channel_attention_net(input_cloud_i, training)

#define loss
output_r_1 = output_H[:, :, :, 0:1]
input_high_1 = input_label_i[:, :, :, 0:1]
ssim_r_1 = tf_ssim(output_r_1, input_high_1)
output_r_2 = output_H[:, :, :, 1:2]
input_high_2 = input_label_i[:, :, :, 1:2]
ssim_r_2= tf_ssim(output_r_2, input_high_2)
output_r_3 = output_H[:, :, :, 2:3]
input_high_3 = input_label_i[:, :, :, 2:3]
ssim_r_3 = tf_ssim(output_r_3, input_high_3)
ssim_r = (ssim_r_1 + ssim_r_2 + ssim_r_3)/3.0
loss_ssim = 1-ssim_r

def grad_loss(input_i_cloud, input_i_label):
    input_i_cloud_r = input_i_cloud[:,:,:,0:1]
    input_i_label_r = input_i_label[:, :, :,0:1]
    x_loss_r = tf.square(gradient_no_abs(input_i_cloud_r, 'x') - gradient_no_abs(input_i_label_r, 'x'))
    y_loss_r = tf.square(gradient_no_abs(input_i_cloud_r, 'y') - gradient_no_abs(input_i_label_r, 'y'))

    input_i_cloud_g = input_i_cloud[:, :, :, 1:2]
    input_i_label_g = input_i_label[:, :, :, 1:2]
    x_loss_g = tf.square(gradient_no_abs(input_i_cloud_g, 'x') - gradient_no_abs(input_i_label_g, 'x'))
    y_loss_g = tf.square(gradient_no_abs(input_i_cloud_g, 'y') - gradient_no_abs(input_i_label_g, 'y'))

    input_i_cloud_b = input_i_cloud[:, :, :, 2:3]
    input_i_label_b = input_i_label[:, :, :, 2:3]
    x_loss_b = tf.square(gradient_no_abs(input_i_cloud_b, 'x') - gradient_no_abs(input_i_label_b, 'x'))
    y_loss_b = tf.square(gradient_no_abs(input_i_cloud_b, 'y') - gradient_no_abs(input_i_label_b, 'y'))

    x_loss = (x_loss_r + x_loss_g + x_loss_b) / 3.0
    y_loss = (y_loss_r + y_loss_g + y_loss_b) / 3.0
    grad_loss_all = tf.reduce_mean(x_loss + y_loss)
    return grad_loss_all

# 3) color loss
enhanced_blur = blur(output_H)
dslr_blur = blur(input_label_i)
color_loss = tf.reduce_mean(tf.square(dslr_blur - enhanced_blur))

loss_grad = grad_loss(output_H, input_label_i)
L1_loss = tf.reduce_mean(tf.abs(output_H - input_label_i))

loss_adjust = L1_loss + loss_grad + loss_ssim + color_loss
# loss_adjust = L1_loss + loss_grad + loss_ssim + color_loss

lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
def vars_getname(scope_name_var):
    return [var for var in tf.global_variables() if scope_name_var in var.name]

sess = tf.Session()
## var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_adjust = [var for var in tf.trainable_variables() if 'I_enhance_Net' in var.name]
saver_adjust = tf.train.Saver(var_list=var_adjust)
## saver_Decom = tf.train.Saver(var_list = var_Decom)
train_op_adjust = optimizer.minimize(loss_adjust, var_list = var_adjust)
sess.run(tf.global_variables_initializer())
print("[*] Initialize model successfully...")
summary_writer = tf.summary.FileWriter('temp', sess.graph)

### load data
###train_data

train_cloud_data = []
train_label_data = []
train_cloud_data_names = glob('./dataset/thin_cloudy_dataset//train_A/*.png')  # 薄云图像目录
train_cloud_data_names.sort()
train_label_data_names = glob('./dataset/thin_cloudy_dataset//train_B/*.png')  # 参考无云图像目录
train_label_data_names.sort()
assert len(train_cloud_data_names) == len(train_label_data_names)
print('[*] Number of all training data: %d' % len(train_cloud_data_names))

for idx in range(len(train_cloud_data_names)):
    cloud_im = load_images(train_cloud_data_names[idx])
    train_cloud_data.append(cloud_im)
    label_im = load_images(train_label_data_names[idx])
    train_label_data.append(label_im)

# 随机选择验证集
eval_adjust_cloud_i_data = []
eval_adjust_label_i_data = []
for i in range(eval_sample_nums):
    randIndex = int(np.random.uniform(0, len(train_cloud_data)))
    eval_adjust_cloud_i_data.append(train_cloud_data[randIndex])
    eval_adjust_label_i_data.append(train_label_data[randIndex])
    del(train_cloud_data[randIndex])
    del(train_label_data[randIndex])

train_sample_nums = len(train_cloud_data)
train_adjust_cloud_i_data = train_cloud_data[0:train_sample_nums]
train_adjust_label_i_data = train_label_data[0:train_sample_nums]

print('[*] Number of training data: %d' % len(train_adjust_label_i_data))

train_phase = 'adjustment'
numBatch = len(train_adjust_cloud_i_data) // int(batch_size)
train_op = train_op_adjust
train_loss = loss_adjust
saver = saver_adjust

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    print('loading finished!')
else:
    print("No adjustment net pre model, start training from scratch!")

start_step = 0
start_epoch = 0
iter_num = 0
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

# start_time = time.time()
# start_time = datetime.now()
start_time = time.monotonic()
image_id = 0

for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_cloud_i = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_label_i = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        input_cloud_i_rand = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        input_label_i_rand = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")

        for patch_id in range(batch_size):
            i_low_data = train_adjust_cloud_i_data[image_id]
            i_high_data = train_adjust_label_i_data[image_id]

            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            i_low_data_crop = i_low_data[x : x+patch_size, y : y+patch_size, :]
            i_high_data_crop = i_high_data[x : x+patch_size, y : y+patch_size, :]

            rand_mode = np.random.randint(0, 7)
            batch_input_cloud_i[patch_id, :, :, :] = data_augmentation(i_low_data_crop, rand_mode)
            batch_input_label_i[patch_id, :, :, :] = data_augmentation(i_high_data_crop, rand_mode)

            rand_mode = np.random.randint(0, 2)
            if rand_mode == 1:
                input_cloud_i_rand[patch_id, :, :, :] = batch_input_cloud_i[patch_id, :, :, :]
                input_label_i_rand[patch_id, :, :, :] = batch_input_label_i[patch_id, :, :, :]

            else:
                input_cloud_i_rand[patch_id, :, :, :] = batch_input_label_i[patch_id, :, :, :]
                input_label_i_rand[patch_id, :, :, :] = batch_input_cloud_i[patch_id, :, :, :]

            image_id = (image_id + 1) % len(train_adjust_cloud_i_data)
            if image_id == 0:
                tmp = list(zip(train_adjust_cloud_i_data, train_adjust_label_i_data))
                random.shuffle(list(tmp))
                train_adjust_cloud_i_data, train_adjust_label_i_data = zip(*tmp)

        _, training_loss = sess.run([train_op, train_loss], feed_dict={input_cloud_i: input_cloud_i_rand, \
                                                              input_label_i: input_label_i_rand, \
                                                          training: True, lr: learning_rate})
         # summary_writer.add_summary(summary, epoch)

        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, training_loss))
        iter_num += 1

    if (epoch + 1) % eval_every_epoch == 0:
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
        numEval = len(eval_adjust_cloud_i_data)
        iteration1 = []
        Loss1 = []
        all_eval_loss = 0.0

        for idx in range(numEval):
            rand_idx = idx # np.random.randint(26)
            input_uu_i = eval_adjust_cloud_i_data[rand_idx]
            input_cloud_eval_i = np.expand_dims(input_uu_i, axis=0)
            # input_low_eval_ii = np.expand_dims(input_low_eval_i, axis=2)
            input_uu_label_i = eval_adjust_label_i_data[rand_idx]

            result_1 = sess.run(output_H, feed_dict={input_cloud_i: input_cloud_eval_i, training: False})
            eval_loss = tf.reduce_mean(tf.square(input_uu_label_i - result_1), keep_dims=False)
            eval_loss = sess.run(eval_loss)         # ====sess.run保证张量tensor转换为numpy数据======
            all_eval_loss =  all_eval_loss + eval_loss
            save_images(os.path.join(sample_dir, 'eval_%d_%d.png' % ( epoch + 1 , rand_idx + 1)), input_uu_i, input_uu_label_i, result_1 )

        # output = "Epoch: [%2d], loss: %.6f" % (epoch + 1, eval_loss)
        with open(trainLoss_txt, 'a+') as file:  # 打开文件
            iteration1.append(epoch)  # 保存在数组中
            Loss1.append(all_eval_loss)
            file.write("Epoch:%2d loss:%.3f" % (epoch + 1, all_eval_loss) +'\n')

        global_step = epoch
        saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=global_step)

print("[*] Finish training for phase %s." % train_phase)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

