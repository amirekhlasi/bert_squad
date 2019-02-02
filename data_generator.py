import numpy as np
import json


class DataGenerator(object):
    def __init__(self, data_path, max_seq_length, batch_size=32):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self._data = []
        for example in data:
            inputs = example['question'] + example['context']
            question_length = len(example['question'])
            context_length = len(example['context'])
            inputs_length = question_length + context_length
            if inputs_length > max_seq_length:
                continue
            answers = example['answers']
            exm = {"inputs": inputs, "inputs_length": inputs_length, "question_length": question_length,
                   "context_length": context_length, "answers": answers}
            self._data.append(exm)
        self._batch_size = batch_size
        self._size = len(self._data)
        self._start = 0

    def shuffle(self):
        l = len(self._data)
        perm = np.random.permutation(l) - 1
        self._data = [self._data[i] for i in perm]

    def reset(self, shuffle=False):
        self._start = 0
        if shuffle:
            self.shuffle()

    def has_ended(self):
        return self._start >= self._size

    def _get_next(self, batch_size=None):
        if self.has_ended():
            raise IndexError("generator is ended")
        if batch_size is None:
            batch_size = self._batch_size
        start = self._start
        end = min(self._size, self._start + batch_size)
        nxt = self._data[start:end]
        self._start = end
        return nxt

    @classmethod
    def get_answers(cls, answers, question_lengths):
        lens = []
        batch_index = []
        answer_index = []
        start = []
        end = []
        for b_index, ans in enumerate(answers):
            lens.append(len(ans))
            for a_index, answer in enumerate(ans):
                batch_index.append(b_index)
                answer_index.append(a_index)
                start.append(answer['start'])
                end.append(answer['end'])
        answers = np.zeros(shape=(len(lens), max(lens), 2), dtype=int)
        answers[batch_index, answer_index, 0] = start
        answers[batch_index, answer_index, 1] = end
        question_lengths = np.array(question_lengths, dtype=int)
        question_lengths = question_lengths.reshape((question_lengths.size, 1, 1))
        answers = answers + question_lengths
        return answers, lens

    @classmethod
    def get_inputs(cls, inputs):
        batch_index = []
        seq_index = []
        lens = []
        ids = []
        for b_index, text in enumerate(inputs):
            l = len(text)
            lens.append(l)
            ids = ids + text
            batch_index = batch_index + ([b_index] * l)
            seq_index = seq_index + list(range(l))
        inputs = np.zeros(shape=(len(lens), max(lens)), dtype=int)
        inputs[batch_index, seq_index] = ids
        return inputs, lens

    def get_next(self, batch_size=None):
        data = self._get_next(batch_size)
        inputs = []
        question_length = []
        answers = []
        context_length = []
        for example in data:
            inputs.append(example['inputs'])
            question_length.append(example['question_length'])
            context_length.append(example['context_length'])
            answers.append(example['answers'])
        inputs, inputs_length = self.get_inputs(inputs)
        max_seq = inputs.shape[-1]
        segments = []
        for q, c in zip(question_length, context_length):
            d = max_seq - q - c
            seg = [0]*q + [1]*c + [0]*d
            segments.append(seg)
        answers, answers_length = self.get_answers(answers, question_length)
        return inputs, segments, inputs_length, answers, answers_length

    def get_all(self):
        data = self._data
        inputs = []
        question_length = []
        answers = []
        context_length = []
        for example in data:
            inputs.append(example['inputs'])
            question_length.append(example['question_length'])
            context_length.append(example['context_length'])
            answers.append(example['answers'])
        inputs, inputs_length = self.get_inputs(inputs)
        max_seq = inputs.shape[-1]
        segments = []
        for q, c in zip(question_length, context_length):
            d = max_seq - q - c
            seg = [0] * q + [1] * c + [0] * d
            segments.append(seg)
        answers, answers_length = self.get_answers(answers, question_length)
        return inputs, segments, inputs_length, answers, answers_length


























