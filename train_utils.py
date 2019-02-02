class Average(object):
    def __init__(self):
        self._average = 0
        self._size = 0

    def add(self, value, batch_size):
        w = self._size / (self._size + batch_size)
        self._average = w * self._average + (1 - w) * value
        self._size = self._size + batch_size

    @property
    def size(self):
        return self._size

    @property
    def average(self):
        return self._average

    def reset(self):
        self.__init__()


class Logger(object):
    def __init__(self, file_path):
        self._file_path = file_path

    def log(self, title, loss: Average, accuracy: Average, step):
        output = []
        output.append(title + "  step: " + str(step))
        output.append("loss: " + str(loss.average))
        output.append("accuracy: " + str(accuracy.average))
        output.append("")
        output = "\n".join(output)
        print(output)
        with open(self._file_path, 'a') as f:
            f.write(output)

    def log_text(self, text):
        print(text)
        with open(self._file_path, 'a') as f:
            f.write(text)

