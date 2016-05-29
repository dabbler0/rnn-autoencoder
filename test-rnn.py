import theano
import numpy
import theano.tensor as T
import rnn

import sys
sys.setrecursionlimit(50000)

n = 1500
nin = 256
nout = 256

k = 100

if __name__ == "__main__":
    # Initialize a pseudorandom number generator
    rng = numpy.random.RandomState(1234)

    # Build the neural network
    x = T.matrix('x', dtype=theano.config.floatX)
    h0 = T.vector('h0', dtype=theano.config.floatX)

    inp = rnn.InputLayer(x, nin)
    hidden = rnn.RecurrenceLayer(h0, n)

    # Collect all the info together
    together = rnn.ConcatLayer(inp, hidden)

    # Determine how much of history we want to use
    # to determine the new value to put in
    use_signal = rnn.TransformLayer(rng, together, n, activation = T.nnet.sigmoid)
    used = rnn.MulLayer(use_signal, hidden)
    new_info = rnn.ConcatLayer(inp, used)

    # Use that info to come up with a new value
    new_value = rnn.TransformLayer(rng, new_info, n)

    # Use all the info to figure out how much
    # of the hidden value we want to replace
    update_signal = rnn.TransformLayer(rng, together, n, activation = T.nnet.sigmoid)
    inverse_update_signal = rnn.MapLayer(update_signal, lambda x: 1 - x)

    # Add together as much of the old and new parts of history
    # as is appropriate
    old_use = rnn.MulLayer(hidden, inverse_update_signal)
    new_use = rnn.MulLayer(new_value, update_signal)

    new_hidden = rnn.AddLayer(old_use, new_use)
    hidden.set_recurrence(new_hidden)

    out = rnn.TransformLayer(rng, new_hidden, nout, activation = T.nnet.softmax)

    trainer = rnn.Trainer(out, k, lr = 0.001)

    # Tokenization functions
    def to_matrix(substring):
        matrix = numpy.zeros((len(substring), 256), dtype=theano.config.floatX)
        for i in range(len(substring)):
            matrix[i][max(0, min(255, ord(substring[i])))] = 1

        return matrix

    def from_vector(vector):
        return chr(numpy.random.choice(list(range(len(vector)))), p = vector)

    # Generation function
    fn = theano.function(
        [x, h0],
        (out.unfold(0), new_hidden.unfold(0))
    )
    def generate(length):
        value = numpy.zeros((1, nin), dtype=theano.config.floatX)
        hidden = numpy.zeros(n, dtype=theano.config.floatX)
        string = ''

        for i in range(length):
            value, hidden = fn(value, hidden)
            string += from_vector(value)
            value = numpy.array([value])

        return string

    string = open('OANC.txt', 'r').read()

    # Run an epoch
    def run_epoch():
        index = numpy.random.randint(len(string) - k)

        inputs = to_matrix(string[index:index + k])
        outputs = to_matrix(string[index + 1:index + k + 1])

        print(trainer.train({
            x: inputs,
            h0: numpy.zeros(n, dtype=theano.config.floatX)
        }, outputs))

    # Run the aforementioned epochs
    i = 0
    while True:
        i += 1
        run_epoch()
        if (i % 100 == 0):
            print(generate(100))
