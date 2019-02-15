import json
import numpy as np
import HP
import pandas as pd


def probs(i, data):
    probs = data['probs']
    probs = np.array(probs)
    sample = probs[:, i, :]
    dat = pd.DataFrame(sample, columns=["9", "10", "11", "12"])
    fig = dat.plot().get_figure()
    fig.savefig(HP.output + "eta_%d.png" % i)


def all_accuracy(data):
    accuracy = data['my_accuracy']
    accuracy = np.array(accuracy)
    accuracy = accuracy.cumsum(0)
    ran = np.arange(accuracy.shape[0]) + 1
    ran = np.expand_dims(ran, 1)
    accuracy = accuracy / ran
    eta = np.arange(*HP.eta)
    dt = pd.DataFrame(accuracy[-1], columns=["accuracy"], index=eta)
    pl = dt.plot()
    fig = pl.get_figure()
    fig.savefig(HP.output + "all_eta_accuracy.png")


def id_title_dictionary(data):
    return {int(exm['id'], 16): exm['title'] for exm in data}


def dev_data(data):
    result = []
    for example in data:
        question_length = len(example['question'])
        context_length = len(example['context'])
        if context_length + question_length <= HP.max_seq_length:
            result.append(example)
    return result


def create_mean(books, data, id_title, accuracy):
    books_id = {book: i for i, book in enumerate(books)}
    books = []
    for example in data:
        i = example['id']
        i = int(i, 16)
        b = id_title[i]
        b_id = books_id[b]
        books.append(b_id)
    books = np.array(books, dtype=int)
    books = np.eye(len(books_id))[books]
    books_num = books.sum(0)
    x = np.expand_dims(books, 2) * np.expand_dims(accuracy, 1)
    x = x.sum(0)
    x = x / np.expand_dims(books_num, 1)
    return x, books_num.astype(int)


def accuracy_data_frame(result):
    with open(HP.pre_dev_file, 'r') as f:
        data = json.load(f)
    id_title = id_title_dictionary(data)
    with open(HP.dev_file, 'r') as f:
        data = json.load(f)
    data = dev_data(data)
    experts = result['accuracy']
    experts = np.array(experts)
    with open(HP.original_dev_file) as f:
        squad_data = json.load(f)['data']
    books = [book['title'] for book in squad_data]
    expert_mean, books_num = create_mean(books, data, id_title, experts)
    expert_df = pd.DataFrame(expert_mean, columns=['9', '10', '11', '12'], index=books)
    expert_df = expert_df.round(3)
    expert_df['question_num'] = books_num
    save = expert_df.to_csv(sep=',')
    with open(HP.output + "accuracy.csv", 'w') as f:
        f.write(save)


if __name__ == "__main__":
    with open(HP.weights_file, 'r') as f:
        data = json.load(f)
    accuracy_data_frame(data)
    all_accuracy(data)
    for i in [22, 39, 40]:
        probs(i, data)



