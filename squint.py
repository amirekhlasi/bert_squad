import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, "/content/drive/My Drive/opt-project/codes/")

import model
import HP
from data_generator import DataGenerator
import json
from bert import BertConfig

bert_config = BertConfig.from_json_file(HP.bert_config)
inputs = tf.placeholder(dtype=tf.int32, shape=[1, None])
segments = tf.placeholder(dtype=tf.int32, shape=[1, None])
inputs_length = tf.placeholder(dtype=tf.int32, shape=[1])
answers = tf.placeholder(dtype=tf.int32, shape=[1, None, 2])
answers_length = tf.placeholder(dtype=tf.int32, shape=[1])


mod = model.Model(bert_config, HP.is_training, HP.num_units, inputs, segments, inputs_length,
                  answers, answers_length, layers=HP.layers)
loss = mod.losses(HP.train_layers)
loss = [loss[i] for i in HP.layers]
accuracy = mod.accuracy(HP.train_layers)
accuracy = list(accuracy[i] for i in HP.layers)

num_experts = len(HP.layers)
eta = np.arange(HP.eta[0], HP.eta[1], HP.eta[2], dtype=np.float32)
num_eta = len(eta)
log_probs = tf.get_variable(name='log_prob', shape=[num_eta, num_experts], dtype=tf.float32, trainable=False)
probs = tf.nn.softmax(log_probs, 1)

init = tf.assign(log_probs, tf.zeros_like(log_probs))

choice = tf.random.multinomial(log_probs, 1, output_dtype=tf.int32)
choice = tf.squeeze(choice, -1)
loss = tf.stack(loss)
accuracy = tf.stack(accuracy)
my_loss = tf.gather(loss, choice)
my_accuracy = tf.gather(accuracy, choice)

mean_loss = tf.reduce_sum((1 - accuracy) * probs, -1)
regret = tf.expand_dims(mean_loss, 1) - loss

log_update = tf.expand_dims(eta, 1) * regret
log_update = log_update - log_update**2
with tf.control_dependencies([my_loss, my_accuracy]):
    update = tf.assign(log_probs, log_probs + log_update)

dev_data = DataGenerator(HP.dev_file, HP.max_seq_length, 1)

weights = {v.name: v for v in mod.weights}

starter = tf.train.Saver(weights)

saver = tf.train.Saver({log_probs.name: log_probs})

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
starter.restore(sess, HP.start_checkpoint)

sess.run(init)

keys = ["loss", "accuracy", "my_loss", "my_accuracy", "probs"]
step = 0
saved_items = [[] for _ in keys]
with open(HP.log_files, 'w') as f:
    f.write("***START***\n\n\n")
print("***START***\n\n\n")
while not dev_data.has_ended():
    step = step + 1
    dev_inputs, dev_segments, dev_inputs_length, dev_answers, dev_answers_length = \
        dev_data.get_next()
    feed_dict = {inputs: dev_inputs, segments: dev_segments, inputs_length: dev_inputs_length,
                 answers: dev_answers, answers_length: dev_answers_length}
    items = sess.run([loss, accuracy, my_loss, my_accuracy, probs, update], feed_dict=feed_dict)
    items = items[:-1]
    for saved_item, item in zip(saved_items, items):
        saved_item.append(item.tolist())
    if step % 1000 == 0:
        text = "step: %d\nsaving checkpoint\n\n" % step
        print(text)
        with open(HP.log_files, 'a') as f:
            f.write(text)
        dictionary = dict(zip(keys, saved_items))
        with open(HP.weights_file, 'w') as f:
            json.dump(dictionary, f)
        saver.save(sess, HP.save_checkpoint)

dictionary = dict(zip(keys, saved_items))
with open(HP.weights_file, 'w') as f:
    json.dump(dictionary, f)
print("\n\n\nEND")
with open(HP.log_files, 'w') as f:
    f.write("***END***\n\n\n")
