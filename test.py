import tensorflow as tf

import model

def softmax(x, axis):
    m = tf.reduce_max(x, axis=axis)
    m = tf.expand_dims(m, axis)
    x = x - m
    x = tf.exp(x)
    m = tf.reduce_sum(x, axis)
    m = tf.expand_dims(m, axis)
    x = x / m
    return x


answer = tf.constant([[[2, 3]], [[1, 1]], [[0, 2]]])
#print(answer.shape)
ans = tf.squeeze(answer, 1)
zz = tf.expand_dims(tf.range(3), 1)
ans = tf.concat([zz, ans], 1)

ans_mask = tf.scatter_nd(ans, tf.constant([True, True, True]), [3, 4, 4])

w = tf.Variable(initial_value=3.0)
x = tf.random_normal([3, 4, 4])
x = x * w
mask = tf.random.uniform(shape=[3, 4, 4])
mask = mask < 0.5
mask = tf.logical_or(mask, ans_mask)
mask = tf.cast(mask, tf.float32)
x = x * mask + (1 - mask) * -50
x = tf.reshape(x, [3, 16])
x = softmax(x, -1)
x = x / tf.expand_dims(tf.reduce_sum(x, -1), -1)
x = tf.reshape(x, [3, 4, 4])

x = -tf.log(x)

ans_mask = tf.cast(ans_mask, tf.float32)
loss1 = ans_mask * x
loss = tf.reshape(loss1, [3, 16])
loss = tf.reduce_sum(loss, -1)
loss = tf.reduce_mean(loss)

#loss = model.loss(x, answer)
init = tf.variables_initializer(var_list=[w])

opt = tf.train.GradientDescentOptimizer(0.0001)
train = opt.minimize(loss)

sess = tf.Session()
sess.run(init)
print(sess.run(x))
print(sess.run(loss))
sess.run(train)
print(sess.run(w))

