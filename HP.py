import os

prefix = ""
data_path = prefix + "data/"
train_file = data_path + "train.json"
pre_train_file = data_path + "train-pre.json"
original_train_file = data_path + "train-v2.0.json"
dev_file = data_path + "dev.json"
pre_dev_file = data_path + "dev-pre.json"
original_dev_file = data_path + "dev-v2.0.json"
accuracy_file = data_path + "accuracy.csv"
layers = [8, 9, 10, 11]
bert_train = False
train_layers = [8, 9, 10, 11]
epochs = 2
bert_path = prefix + "BERT/"
bert_vocab = bert_path + "vocab.txt"
bert_config = bert_path + "bert_config.json"
start1_checkpoint = bert_path + "bert_model.ckpt"
checkpoint_path = prefix + "checkpoints/"
start2_checkpoint = checkpoint_path + "bert_model"
save1_checkpoint = checkpoint_path + "main"
save2_checkpoint = checkpoint_path + "expert_save"
end1_checkpoint = checkpoint_path + "bert_model"
weights_file = checkpoint_path + "weights.json"
max_seq_length = 256
is_training = False
batch_size = 16
log_files = prefix + "logs.txt"
learning_rate = 0.00002
num_units = 64
eta = [0.001, 1, 0.01]
output = prefix + "output/"

if __name__ == "__main__":
    paths = [bert_path, checkpoint_path, data_path, output]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

