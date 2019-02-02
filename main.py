import sys
import tensorflow as tf
import numpy as np
sys.path.insert(0, "/content/drive/My Drive/opt-project/codes/")

from data_generator import DataGenerator
import model
from bert import BertConfig
from train_utils import *
import HP

bert_config = BertConfig.from_json_file(HP.bert_config)
inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
segments = tf.placeholder(dtype=tf.int32, shape=[None, None])
inputs_length = tf.placeholder(dtype=tf.int32, shape=[None])
answers = tf.placeholder(dtype=tf.int32, shape=[None, None, 2])
answers_length = tf.placeholder(dtype=tf.int32, shape=[None])


mod = model.Model(bert_config, HP.is_training, HP.num_units, inputs, segments, inputs_length,
                  answers, answers_length, layers=HP.layers)

train_data = DataGenerator(HP.train_file, HP.max_seq_length, HP.batch_size)
dev_data = DataGenerator(HP.dev_file, HP.max_seq_length, HP.batch_size)

long_train_loss = {i: Average() for i in HP.train_layers}
long_train_accuracy = {i: Average() for i in HP.train_layers}

train_loss = {i: Average() for i in HP.train_layers}
train_accuracy = {i: Average() for i in HP.train_layers}

dev_loss = {i: Average() for i in HP.train_layers}
dev_accuracy = {i: Average() for i in HP.train_layers}

epoch_dev_loss = {i: Average() for i in HP.train_layers}
epoch_dev_accuracy = {i: Average() for i in HP.train_layers}


sess = tf.Session()
var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='bert')
var = {v.name: v for v in var}
start_saver = tf.train.Saver(var)
start_saver.restore(sess, HP.start_checkpoint)
sess.run(mod.init())

logger = Logger(HP.log_files)

loss = mod.losses(HP.train_layers)
accuracy = mod.accuracy(HP.train_layers)
train = mod.train(HP.learning_rate, HP.bert_train, HP.train_layers)

var = mod.weights
var = {v.name: v for v in var}
saver = tf.train.Saver(var)

step = 0
for epoch in range(1, HP.epochs + 1):
    while not train_data.has_ended():
        step = step + 1
        train_inputs, train_segments, train_inputs_length, train_answers, train_answers_length = \
            train_data.get_next()
        feedict = {inputs: train_inputs, segments: train_segments, inputs_length: train_inputs_length,
                   answers: train_answers, answers_length: train_answers_length}
        _loss, _accuracy, _ = sess.run([loss, accuracy, train], feed_dict=feedict)
        _batch_size = len(train_inputs_length)
        for i in HP.train_layers:
            train_loss[i].add(_loss[i], _batch_size)
            train_accuracy[i].add(_accuracy[i], _batch_size)
            long_train_loss[i].add(_loss[i], _batch_size)
            long_train_accuracy[i].add(_accuracy[i], _batch_size)
        if dev_data.has_ended():
            dev_data.reset(False)
        dev_inputs, dev_segments, dev_inputs_length, dev_answers, dev_answers_length = dev_data.get_next()
        feedict = {inputs: dev_inputs, segments: dev_segments, inputs_length: dev_inputs_length,
                   answers: dev_answers, answers_length: dev_answers_length}
        _loss, _accuracy = sess.run([loss, accuracy], feed_dict=feedict)
        _batch_size = len(dev_inputs_length)
        for i in HP.train_layers:
            dev_loss[i].add(_loss[i], _batch_size)
            dev_accuracy[i].add(_accuracy[i], _batch_size)
        if step % 500 == 0:
            for i in HP.train_layers:
                logger.log("train " + str(i), train_loss[i], train_accuracy[i], step)
                logger.log("dev " + str(i), dev_loss[i], dev_accuracy[i], step)
                train_loss[i].reset()
                train_accuracy[i].reset()
                dev_loss[i].reset()
                dev_accuracy[i].reset()
        if step % 1000 == 0:
            saver.save(sess, HP.end_checkpoint)
            logger.log_text("saving checkpoint")
    for i in HP.train_layers:
        logger.log("epoch train " + str(i), long_train_loss[i], long_train_accuracy[i], step)
        long_train_loss[i].reset()
        long_train_accuracy[i].reset()
    train_data.reset(True)
    dev_data.reset(True)
    while not dev_data.has_ended():
        dev_inputs, dev_segments, dev_inputs_length, dev_answers, dev_answers_length = dev_data.get_next()
        feedict = {inputs: dev_inputs, segments: dev_segments, inputs_length: dev_inputs_length,
                   answers: dev_answers, answers_length: dev_answers_length}
        _loss, _accuracy = sess.run([loss, accuracy], feed_dict=feedict)
        _batch_size = len(dev_inputs_length)
        for i in HP.train_layers:
            epoch_dev_loss[i].add(_loss[i], _batch_size)
            epoch_dev_accuracy[i].add(_accuracy[i], _batch_size)
    for i in HP.train_layers:
        logger.log("epoch validation " + str(i), epoch_dev_loss[i], epoch_dev_accuracy[i], epoch)
    dev_data.reset(True)

saver.save(sess, HP.end_checkpoint)









