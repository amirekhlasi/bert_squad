import json

def make_data(data):
    new_data = []
    for book in data:
        title = book['title']
        for parapgraph in book['paragraphs']:
            par = parapgraph['context']
            qas = parapgraph['qas']
            for qa in qas:
                i = qa['id']
                question = qa['question']
                answers = qa['answers']
                if len(answers) == 0:
                    continue
                is_impossible = qa['is_impossible']
                if is_impossible:
                    continue
                nd = {'id': i, 'title': title, 'question': question, 'answers': answers, 'context': par}
                new_data.append(nd)
    return new_data


if __name__ == "__main__":
    with open("data/train-v2.0.json", 'r') as f:
        data = json.load(f)['data']
    new_data = make_data(data)
    with open("data/train-pre.json", 'w') as f:
        json.dump(new_data, f)
    new_data = None
    with open("data/dev-v2.0.json", 'r') as f:
        data = json.load(f)['data']
    new_data = make_data(data)
    with open("data/dev-pre.json", 'w') as f:
        json.dump(new_data, f)
