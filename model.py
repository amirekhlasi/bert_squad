import tensorflow as tf
import bert


class AnswerFinder(tf.keras.Model):
    def __init__(self, num_units, name=None):
        super().__init__(name=name)
        self._num_units = num_units
        self.dense = tf.layers.Dense(num_units, activation=bert.gelu, name="output")
        self.dense_1 = tf.layers.Dense(1, use_bias=False, name="start_dense")
        self.dense_2 = tf.layers.Dense(num_units, activation=bert.gelu, name="cond_dense")
        self.dense_3 = tf.layers.Dense(1, use_bias=False, name="end_dense")

    def call(self, inputs, training=None, mask=None):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        seq_len = shape[1]
        dim = shape[2]
        if mask is None:
            inputs_mask = tf.ones(shape=(batch_size, seq_len), dtype=tf.bool)
        else:
            inputs_mask = mask
        inputs = tf.boolean_mask(inputs, inputs_mask)
        inputs = self.dense(inputs)
        indices = tf.where(inputs_mask)
        inputs = tf.scatter_nd(indices, inputs, (batch_size, seq_len, self._num_units))
        x0 = tf.boolean_mask(inputs, inputs_mask)
        x = self.dense_1(x0)
        x = tf.squeeze(x, 1)
        start_logits = tf.scatter_nd(indices, x, shape=(batch_size, seq_len))
        x = self.dense_2(x0)
        start_cond = tf.scatter_nd(indices, x, shape=(batch_size, seq_len, self._num_units))
        second_inputs = tf.expand_dims(inputs, 1) + tf.expand_dims(start_cond, 2)
        x = tf.range(seq_len)
        x = tf.sequence_mask(x, x[-1] + 1)
        x = tf.logical_not(x)
        second_mask = tf.expand_dims(inputs_mask, -1)
        second_mask = tf.logical_and(second_mask, x)
        x = tf.expand_dims(inputs_mask, 1)
        second_mask = tf.logical_and(second_mask, x)
        second_indices = tf.where(second_mask)
        x = tf.boolean_mask(second_inputs, second_mask)
        x = self.dense_3(x)
        x = tf.squeeze(x, 1)
        end_logits = tf.scatter_nd(second_indices, x, shape=(batch_size, seq_len, seq_len))
        inputs_mask = tf.cast(inputs_mask, tf.float32)
        start_logits = start_logits * inputs_mask + (1 - inputs_mask) * -10
        second_mask = tf.cast(second_mask, tf.float32)
        end_logits = end_logits * second_mask + (1 - second_mask) * -10
        start_probs = tf.nn.softmax(start_logits, axis=-1)
        x = tf.reshape(end_logits, (batch_size, seq_len * seq_len))
        x = tf.nn.softmax(x, axis=-1)
        end_probs = tf.reshape(x, (batch_size, seq_len, seq_len))
        start_logprob = -tf.log(start_probs)
        end_logprob = -tf.log(end_probs)
        whole_logprob = tf.expand_dims(start_logprob, -1) + end_logprob
        return whole_logprob

    def __call__(self, inputs, mask=None, **kwargs):
        return super().__call__(inputs=inputs, mask=mask, **kwargs)



class Model(object):
    def __init__(self, config: bert.BertConfig, is_training, num_units, inputs, segments, inputs_length=None,
                 answers=None, answers_length=None, layers=None):
        if inputs_length is None:
            inputs_mask = tf.ones(tf.shape(inputs), dtype=tf.bool)
        else:
            inputs_mask = tf.sequence_mask(inputs_length)
        if answers is None:
            assert answers_length is None
        if layers is None:
            layers = [config.num_hidden_layers - 1]
        self.layers = layers
        self.bertmodel = bert.BertModel(config=config, is_training=is_training, input_ids=inputs,
                                        input_mask=inputs_mask, token_type_ids=segments, scope='bert')
        self._num_units = num_units
        outputs = self.bertmodel.get_all_encoder_layers()
        self.outputs = {i: outputs[i] for i in self.layers}
        #with tf.device("cpu:0"):
        self.answers_finders = {i: AnswerFinder(num_units, name="answer_layer_" + str(i))
                                for i in self.layers}
        self._logprobs = []
        for i, answer_finder in self.answers_finders.items():
            x = answer_finder(inputs=self.outputs[i], mask=tf.cast(segments, tf.bool))
            self._logprobs.append((i, x))
        self._logprobs = dict(self._logprobs)
        self._predicts = {i: predict(logprob) for i, logprob in self._logprobs.items()}
        if answers is not None:
            if answers_length is None:
                answers_mask = tf.ones(tf.shape(answers)[0:2], dtype=tf.bool)
            else:
                answers_mask = tf.sequence_mask(answers_length)
            self._losses = {i: loss(self._logprobs[i], answers, answers_mask) for i in self.layers}
            self._accuracy = {i: accuracy(self._predicts[i], answers, answers_mask) for i in self.layers}

    def accuracy(self, layers=None):
        res = self._accuracy
        if layers is not None:
            res = {i: res[i] for i in layers}
        return res

    def losses(self, layers=None):
        res = self._losses
        if layers is not None:
            res = {i: res[i] for i in layers}
        return res

    def logprobs(self, layers=None):
        res = self._logprobs
        if layers is not None:
            res = {i: res[i] for i in layers}
        return res

    def predicts(self, layers=None):
        res = self._predicts
        if layers is not None:
            res = {i: res[i] for i in layers}
        return res

    @property
    def weights(self):
        bert_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='bert')
        af_vars = [af.weights for af in self.answers_finders.values()]
        af_vars = sum(af_vars, [])
        return bert_vars + af_vars

    def init(self):
        af_vars = [af.weights for af in self.answers_finders.values()]
        af_vars = sum(af_vars, [])
        init = tf.variables_initializer(af_vars)
        return init

    def train(self, learning_rate, bert_train=True, layers=None):
        if layers is None:
            layers = self.layers
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        vars = [self.answers_finders[i].weights for i in layers]
        vars = sum(vars, [])
        if bert_train:
            vars = vars + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='bert')
        losses = [self._losses[lay] for lay in layers]
        losses = tf.stack(losses, -1)
        loss = tf.reduce_mean(losses, -1)
        opt = self.optimizer.minimize(loss, var_list=vars)
        return opt



def loss(logprobs, answers, answers_mask=None):
    shape = tf.shape(logprobs)
    batch_size = shape[0]
    seq_length = shape[1]
    asserts = []
    asserts.append(tf.assert_equal(seq_length, shape[2]))
    answers_shape = tf.shape(answers)
    asserts.append(tf.assert_equal(batch_size, answers_shape[0]))
    answers_num = answers_shape[1]
    if answers_mask is None:
        answers_mask = tf.ones(shape=(batch_size, answers_num), dtype=tf.bool)
    with tf.control_dependencies(asserts):
        x = tf.range(batch_size)
        x = tf.reshape(x, (batch_size, 1, 1))
        x = tf.tile(x, [1, answers_num, 1])
        answers_with_batch = tf.concat([x, answers], axis=-1)
        loss = tf.gather_nd(logprobs, answers_with_batch)
        answers_mask = tf.cast(answers_mask, tf.float32)
        loss = loss * answers_mask + (1 - answers_mask) * 10000
        loss = tf.reduce_min(loss, axis=1)
    return tf.reduce_mean(loss)


def predict(logprobs):
    shape = tf.shape(logprobs)
    batch_size = shape[0]
    j_max = tf.argmin(logprobs, 2)
    x = tf.reduce_min(logprobs, 2)
    i_max = tf.argmin(x, 1)
    i_max = tf.cast(i_max, tf.int32)
    j_max = tf.cast(j_max, tf.int32)
    bi = tf.stack([tf.range(batch_size, dtype=tf.int32), i_max], axis=1)
    j_max = tf.gather_nd(j_max, bi)
    pred = tf.stack([i_max, j_max], axis=1)
    return pred


def accuracy(preds, answers, answers_mask=None):
    shape = tf.shape(preds)
    batch_size = shape[0]
    asserts = []
    answers_shape = tf.shape(answers)
    asserts.append(tf.assert_equal(batch_size, answers_shape[0]))
    answers_num = answers_shape[1]
    if answers_mask is None:
        answers_mask = tf.ones(shape=(batch_size, answers_num), dtype=tf.bool)
    with tf.control_dependencies(asserts):
        preds = tf.expand_dims(preds, 1)
        correct = tf.equal(preds, answers)
        correct = tf.reduce_all(correct, -1)
        correct = tf.logical_and(correct, answers_mask)
        correct = tf.reduce_any(correct, 1)
        correct = tf.cast(correct, tf.float32)
    return tf.reduce_mean(correct)




















