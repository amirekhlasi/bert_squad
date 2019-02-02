import tensorflow as tf

w = tf.Variable([2.0, 3.0])
opt = tf.train.AdamOptimizer()
x = tf.reduce_sum(w*w)
train = opt.minimize(x)
print(tf.global_variables())
print(tf.trainable_variables())