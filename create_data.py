import json
import re
import tokenization

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

def analyse(text):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset


def example_analyse(example):
    text = example['context']
    answers = example['answers']
    doc_tokens, char_to_word_offset = analyse(text)
    new_answers = []
    for ans in answers:
        answer_start = ans['answer_start']
        answer_length = len(ans['text'])
        answer_end = answer_start + answer_length - 1
        start = char_to_word_offset[answer_start]
        end = char_to_word_offset[answer_end]
        new_ans = {"start": start, "end": end}
        new_answers.append(new_ans)
    return doc_tokens, new_answers


def convert_example(example, tokenizer):
    question_tokens = tokenizer.tokenize(example['question'])
    question_tokens = ["[CLS]"] + question_tokens + ["[SEP]"]
    question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    doc_tokens, answers = example_analyse(example)
    bert_tokens = []
    doc_to_bert_offset = []
    for token in doc_tokens:
        b_tokens = tokenizer.tokenize(token)
        doc_to_bert_offset.append(len(bert_tokens))
        bert_tokens = bert_tokens + b_tokens
    bert_answers = []
    for ans in answers:
        start = doc_to_bert_offset[ans['start']]
        end = doc_to_bert_offset[ans['end']]
        new_ans = {"start": start, "end": end}
        bert_answers.append(new_ans)
    context_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    new_example = {"id": example['id'], "context": context_ids,
                   "question": question_ids, "answers": bert_answers}
    return new_example


def create_data(data, tokenizer):
    new_data = [convert_example(exp, tokenizer) for exp in data]
    return new_data


if __name__ == "__main__":
    tokenizer = tokenization.FullTokenizer("BERT/vocab.txt")
    with open("data/train-pre.json", 'r') as f:
        data = json.load(f)
    data = create_data(data, tokenizer)
    with open("data/train.json", 'w') as f:
        json.dump(data, f)
    with open("data/dev-pre.json", 'r') as f:
        data = json.load(f)
    data = create_data(data, tokenizer)
    with open("data/dev.json", 'w') as f:
        json.dump(data, f)





