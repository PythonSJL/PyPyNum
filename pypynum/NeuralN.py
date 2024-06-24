from .maths import sigmoid
from .random import gauss


class NeuralNetwork:
    def __init__(self, _input, _hidden, _output):
        self.input = _input + 1
        self.hidden = _hidden
        self.output = _output
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output
        self.wi = [[gauss(0, 1) for _ in range(self.hidden)] for _ in range(self.input)]
        self.wo = [[gauss(0, 1) for _ in range(self.output)] for _ in range(self.hidden)]
        self.ci = [[0.0 for _ in range(self.hidden)] for _ in range(self.input)]
        self.co = [[0.0 for _ in range(self.output)] for _ in range(self.hidden)]

    def feedforward(self, inputs):
        if len(inputs) != self.input - 1:
            raise ValueError("Input quantity error")
        for i in range(self.input - 1):
            self.ai[i] = inputs[i]
        for j in range(self.hidden):
            _sum = 0.0
            for i in range(self.input):
                _sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(_sum)
        for k in range(self.output):
            _sum = 0.0
            for j in range(self.hidden):
                _sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(_sum)
        return self.ao[:]

    def backpropagate(self, targets, n):
        def dsigmoid(y):
            return y * (1.0 - y)

        if len(targets) != self.output:
            raise ValueError("Target quantity error")
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] -= n * change + self.co[j][k]
                self.co[j][k] = change
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] -= n * change + self.ci[i][j]
                self.ci[i][j] = change
        error = 0.0
        for k in range(len(targets)):
            error += (targets[k] - self.ao[k]) ** 2
        return 0.5 * error

    def train(self, x, y, iterations=3000, n=0.0002):
        for i in range(iterations):
            error = 0.0
            for p in zip(x, y):
                inputs = p[0]
                targets = p[1]
                self.feedforward(inputs)
                error = self.backpropagate(targets, n)
            if i < 100 or i % 100 == 99:
                print("epoch {}: error {:.9f}".format(i + 1, error))

    def predict(self, x):
        predictions = []
        for p in x:
            predictions.append(self.feedforward(p))
        return predictions


def neuraln(_input, _hidden, _output):
    return NeuralNetwork(_input, _hidden, _output)
