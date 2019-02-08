drive_path = "/content/drive/My Drive/opt-project/codes/"
train_file = drive_path + "data/train.json"
dev_file = drive_path + "data/dev.json"
layers = [8, 9, 10, 11]
bert_train = False
train_layers = [8, 9, 10, 11]
epochs = 2
bert_config = drive_path + "BERT/bert_config.json"
start_checkpoint = drive_path + "checkpoints/bert_model"
save_checkpoint = drive_path + "checkpoints/expert_save"
end_checkpoint = drive_path + "checkpoints/bert_model"
weights_file = drive_path + "checkpoints/weights.json"
max_seq_length = 256
is_training = False
batch_size = 16
log_files = drive_path + "logs.txt"
learning_rate = 0.00002
num_units = 64
eta = [0.1, 20, 0.2]


