import tensorflow as tf
import numpy as np
import model
import HP
from data_generator import DataGenerator
import json

bert_config = model.BertConfig.from_json_file(HP.bert_config)
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
eta = np.arange(0, 20, 0.1)[1:]
num_eta = len(eta)
probs = tf.get_variable(name='prob', shape=[num_eta, num_experts], dtype=tf.float32, trainable=False)
logits = -tf.log(probs)

init = tf.assign(probs, tf.ones_like(probs) / num_experts)

choice = tf.random.multinomial(logits, 1)
choice = tf.squeeze(choice, -1)
loss = tf.stack(loss)
accuracy = tf.stack(accuracy)
ch = tf.stack([tf.range(num_eta), choice], 1)
my_loss = tf.gather_nd(loss, ch)
my_accuracy = tf.gather_nd(accuracy, ch)

mean_loss = tf.reduce_sum(loss * probs, -1)
regret = tf.expand_dims(mean_loss, 1) - loss

log_update = tf.expand_dims(eta, 1) * regret
log_update = log_update - log_update**2
new_probs = probs * tf.exp(log_update)
new_probs = new_probs / tf.expand_dims(tf.reduce_sum(new_probs, 1), 1)
with tf.control_dependencies([my_loss, my_accuracy]):
    update = tf.assign(probs, new_probs)

dev_data = DataGenerator(HP.dev_file, HP.max_seq_length, 1)

weights = {v.name: v for v in mod.weights}

starter = tf.train.Saver(weights)

saver = tf.train.Saver()

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_memory_growth = True
sess = tf.Session(config=sess_config)

starter.restore(sess, HP.start_checkpoint)

sess.run(init)

keys = ["loss", "accuracy", "my_loss", "my_accuracy", "probs"]
step = 0
saved_items = [[] for _ in keys]
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
        saved_item.append(list(item))
    if step % 1000 == 0:
        print("step: %d\nsaving checkpoint" % step)
        dictionary = dict(zip(keys, saved_items))
        with open(HP.weights_file, 'w') as f:
            json.dump(dictionary, f)
        saver.save(sess, HP.save_checkpoint)

print("\n\n\nEND")
dictionary = dict(zip(keys, saved_items))
with open(HP.weights_file, 'w') as f:
    json.dump(dictionary, f)














