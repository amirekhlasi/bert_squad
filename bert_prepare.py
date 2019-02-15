import tensorflow as tf
import bert
import HP

if __name__ == "__main__":
    bert_config = bert.BertConfig.from_json_file(HP.bert_config)
    x = tf.placeholder(dtype=tf.float32, shape=[1, 1])
    model = bert.BertModel(bert_config, False, x, scope='bert')
    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)
    saver1 = tf.train.Saver()
    saver2 = tf.train.Saver({v.name: v for v in tf.global_variables()})
    saver1.restore(sess, HP.start1_checkpoint)
    saver2.save(sess, HP.start1_checkpoint)
