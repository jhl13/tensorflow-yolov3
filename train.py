#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 17:50:26
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import trange
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from core.config import cfg
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TRAIN.GPU

class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./data/log/train"
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.gpus = utils.get_available_gpus(cfg.TRAIN.GPU_NUM)
        self.steps_per_period    = len(self.trainset)
        self.steps_test          = len(self.testset)
        self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.batch_size_per_gpu = cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.GPU_NUM
        self.clone_scopes = ['clone_%d'%(idx) for idx in range(len(self.gpus))]

        # warmup_steps作用：   
        # 神经网络在刚开始训练的过程中容易出现loss=NaN的情况，为了尽量避免这个情况，因此初始的学习率设置得很低
        # 但是这又使得训练速度变慢了。因此，采用逐渐增大的学习率，从而达到既可以尽量避免出现nan，又可以等训练过程稳定了再增大训练速度的目的。
        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            # 判断语句，在tensorflow中为了方便写成了函数
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            self.global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope('define_input'):
            self.trainable     = tf.placeholder(dtype=tf.bool, name='training')

        # 只训练指定的层，不会一团糟吗？
        with tf.name_scope("define_first_stage_train"):
            self.first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate)#.minimize(loss,
                                                      #var_list=self.first_stage_trainable_var_list)
            
        with tf.name_scope("define_second_stage_train"):
            self.second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate)#.minimize(loss,
                                                      #var_list=second_stage_trainable_var_list)
        
        with tf.device('/cpu:0'):
            train_dataset = tf.data.Dataset.from_generator(lambda: self.trainset, \
                output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
            train_dataset = train_dataset.repeat()
            train_dataset = train_dataset.prefetch(buffer_size=50)
            train_dataset_iter = train_dataset.make_one_shot_iterator()
            
            test_dataset = tf.data.Dataset.from_generator(lambda: self.testset, \
                output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
            test_dataset = test_dataset.repeat()
            test_dataset = test_dataset.prefetch(2)
            test_dataset_iter = test_dataset.make_one_shot_iterator()

            input_data, label_sbbox, label_mbbox, label_lbbox, \
                                    true_sbboxes, true_mbboxes, true_lbboxes, batch_bboxes_gt = \
                                        tf.cond(self.trainable, lambda: train_dataset_iter.get_next(), lambda: test_dataset_iter.get_next())
        
        self.total_loss = 0; # for summary only
        self.giou_loss = 0;
        self.conf_loss = 0;
        self.prob_loss = 0;
        first_stage_gradients = []
        second_stage_gradients = []

        for clone_idx, gpu in enumerate(self.gpus):
            reuse = clone_idx > 0
            with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
                with tf.name_scope(self.clone_scopes[clone_idx]) as clone_scope:
                    with tf.device(gpu) as clone_device:
                            model = YOLOV3(input_data[clone_idx*self.batch_size_per_gpu:(clone_idx+1)*self.batch_size_per_gpu, :, :, :], self.trainable)
                            # self.net_var = tf.global_variables()
                            label_sbbox_per_gpu = label_sbbox[clone_idx*self.batch_size_per_gpu:(clone_idx+1)*self.batch_size_per_gpu, :, :, :, :]
                            label_mbbox_per_gpu = label_mbbox[clone_idx*self.batch_size_per_gpu:(clone_idx+1)*self.batch_size_per_gpu, :, :, :, :]
                            label_lbbox_per_gpu = label_lbbox[clone_idx*self.batch_size_per_gpu:(clone_idx+1)*self.batch_size_per_gpu, :, :, :, :]
                            true_sbboxes_per_gpu = true_sbboxes[clone_idx*self.batch_size_per_gpu:(clone_idx+1)*self.batch_size_per_gpu, :, :]
                            true_mbboxes_per_gpu = true_mbboxes[clone_idx*self.batch_size_per_gpu:(clone_idx+1)*self.batch_size_per_gpu, :, :]
                            true_lbboxes_per_gpu = true_lbboxes[clone_idx*self.batch_size_per_gpu:(clone_idx+1)*self.batch_size_per_gpu, :, :]
                            
                            giou_loss, conf_loss, prob_loss = model.compute_loss(
                                                                    label_sbbox_per_gpu, label_mbbox_per_gpu, label_lbbox_per_gpu,
                                                                    true_sbboxes_per_gpu, true_mbboxes_per_gpu, true_lbboxes_per_gpu)
                            loss = giou_loss + conf_loss + prob_loss
                            self.total_loss += loss
                            self.giou_loss += giou_loss
                            self.conf_loss += conf_loss
                            self.prob_loss += prob_loss
                            conv_lbbox_p = model.pred_conf_l
                            conv_mbbox_p = model.pred_conf_m
                            conv_sbbox_p = model.pred_conf_s

                            batch_image_gt = input_data[clone_idx*self.batch_size_per_gpu:(clone_idx+1)*self.batch_size_per_gpu, :, :, :]
                            batch_image_gt = tf.image.draw_bounding_boxes(batch_image_gt, batch_bboxes_gt[clone_idx*self.batch_size_per_gpu:(clone_idx+1)*self.batch_size_per_gpu, :, :])
                            tf.summary.image("batch_image_gt", batch_image_gt, 3)
                            tf.summary.image("conv_lbbox_p", tf.reshape(conv_lbbox_p[:, :, :, tf.cast(self.global_step % self.anchor_per_scale, dtype=tf.int32), :], \
                                (tf.shape(conv_lbbox_p)[0] , tf.shape(conv_lbbox_p)[1], tf.shape(conv_lbbox_p)[1], 1)), 3)
                            tf.summary.image("conv_mbbox_p", tf.reshape(conv_mbbox_p[:, :, :, tf.cast(self.global_step % self.anchor_per_scale, dtype=tf.int32), :], \
                                (tf.shape(conv_mbbox_p)[0] , tf.shape(conv_mbbox_p)[1], tf.shape(conv_mbbox_p)[1], 1)), 3)
                            tf.summary.image("conv_sbbox_p", tf.reshape(conv_sbbox_p[:, :, :, tf.cast(self.global_step % self.anchor_per_scale, dtype=tf.int32), :], \
                                (tf.shape(conv_sbbox_p)[0] , tf.shape(conv_sbbox_p)[1], tf.shape(conv_sbbox_p)[1], 1)), 3)

                            # compute clone gradients
                            self.first_stage_trainable_var_list = []
                            for var in tf.trainable_variables():
                                var_name = var.op.name
                                var_name_mess = str(var_name).split('/')
                                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                                    self.first_stage_trainable_var_list.append(var)
                            first_clone_gradients = self.first_stage_optimizer.compute_gradients(loss, var_list=self.first_stage_trainable_var_list)
                            first_stage_gradients.append(first_clone_gradients)

                            second_stage_trainable_var_list = tf.trainable_variables()
                            second_clone_gradients = self.second_stage_optimizer.compute_gradients(loss, var_list=second_stage_trainable_var_list)
                            second_stage_gradients.append(second_clone_gradients)

        averaged_first_stage_gradients = self.sum_gradients(first_stage_gradients)
        first_stage_apply_grad_op = self.first_stage_optimizer.apply_gradients(averaged_first_stage_gradients)
        
        # shadow_variable = decay * shadow_variable + (1 - decay) * variable
        with tf.name_scope("define_weight_decay"):
            self.moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())
            
        # 会先执行定义的操作，再执行后续的操作
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([first_stage_apply_grad_op, self.global_step_update]):
                with tf.control_dependencies([self.moving_ave]):
                    self.train_op_with_frozen_variables = tf.no_op()

        averaged_second_stage_gradients = self.sum_gradients(second_stage_gradients)
        second_stage_apply_grad_op = self.second_stage_optimizer.apply_gradients(averaged_second_stage_gradients)
        # 会先执行定义的操作，再执行后续的操作
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([second_stage_apply_grad_op, self.global_step_update]):
                with tf.control_dependencies([self.moving_ave]):
                    self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss", self.giou_loss)
            tf.summary.scalar("conf_loss", self.conf_loss)
            tf.summary.scalar("prob_loss", self.prob_loss)
            tf.summary.scalar("total_loss", self.total_loss)
        
        logdir = "./data/log/"
        if os.path.exists(logdir): shutil.rmtree(logdir)
        os.mkdir(logdir)
        self.write_op = tf.summary.merge_all()
        self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(tf.global_variables())
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    def sum_gradients(self, clone_grads):
        """计算梯度
        Arguments:
            clone_grads -- 每个GPU所对应的梯度
        Returns:
            averaged_grads -- 平均梯度
        """                  
        averaged_grads = []
        for grad_and_vars in zip(*clone_grads):
            grads = []
            var = grad_and_vars[0][1]
            try:
                for g, v in grad_and_vars:
                    assert v == var
                    grads.append(g)
                grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
            except:
                import pdb
                pdb.set_trace()
            averaged_grads.append((grad, v))
        return averaged_grads

    def train(self):
        test_best_loss = 0;
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                print("first")
                train_op = self.train_op_with_frozen_variables
            else:
                print("second")
                train_op = self.train_op_with_all_variables

            pbar = trange(self.steps_per_period)
            test = trange(self.steps_test)
            train_epoch_loss, test_epoch_loss = [], []

            for i in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.total_loss, self.global_step],feed_dict={self.trainable:    True})

                train_epoch_loss.append(train_step_loss)
                pbar.set_description("train loss: %.2f" %train_step_loss)
                if int(global_step_val) % 100 == 0:
                    self.summary_writer.add_summary(summary, global_step_val)

            for j in test:
                test_step_loss, test_step_giou_loss, test_step_conf_loss, test_step_prob_loss \
                    = self.sess.run([self.total_loss, self.giou_loss, self.conf_loss, self.prob_loss],feed_dict={self.trainable:    False})

                test_epoch_loss.append(test_step_loss)
                test.set_description("test loss: %.2f, giou_loss: %.2f, conf_loss: %.2f, prob_loss: %.2f" %(test_step_loss, test_step_giou_loss, test_step_conf_loss, test_step_prob_loss))

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            if epoch == 1:
                test_best_loss = test_epoch_loss
            if test_epoch_loss <= test_best_loss:
                self.saver.save(self.sess, ckpt_file, global_step=epoch)
                print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
                test_best_loss = test_epoch_loss
            else:
                print("=> Epoch: %2d Time: %s we don't save model this epoch ..."
                            %(epoch, log_time))

if __name__ == '__main__': YoloTrain().train()




