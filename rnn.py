import theano
import numpy
import theano.tensor as T

# LAYER SUPERCLASS
# ================
# The Layer object, a superclass for all the other Layers.
# A Layer knows some inputs and how to generate its unfolded output.
class Layer(object):

    # Find all the nodes that this layer could possibly depend on.
    # This will go through recurrent nodes, but stop when it sees something that
    # has already checked to avoid infinite looping.
    def iterate_dependencies(self, func, checked = None):
        if checked is None:
            checked = set()

        for dependency in self.dependencies:
            if dependency not in checked:
                checked.add(dependency)
                func(dependency)
                dependency.iterate_dependencies(func, checked = checked)

    # Iterate through all dependencies to find all learnable parameters
    # this node could depend on
    def collect_params(self):
        params = set()
        self.iterate_dependencies(lambda dependency: params.update(dependency.params))
        return params

    # Iterate through all dependencies to find all raw inputs this node
    # could depend on
    def collect_inputs(self):
        inputs = set()
        self.iterate_dependencies(lambda dependency: inputs.update(dependency.inputs))
        return inputs

    def unfold(self, i):
        if i not in self.unfold_cache:
            self.unfold_cache[i] = self._unfold(i)
        return self.unfold_cache[i]

    # Abstract method _unfold() needs to be implemented in all layers.
    def _unfold(self, i):
        raise NotImplementedError("Layer object %r has not implemented _unfold()" % self)

# INPUT LAYERS
# ============
# InputLayer -- a layer that receives a raw input.
class InputLayer(Layer):
    def __init__(self, var, length):
        self.var = var
        self.length = length

        self.inputs = set([var])
        self.dependencies = set([])
        self.params = set([])

        self.unfold_cache = {}

    # An InputLayer's var should be a matrix where the first
    # axis is time. As such, the input at time i will be self.var[i].
    def unfold(self, i):
        return self.var[i]

# RecurrenceLayer -- this layer should be situated at the bottom of the network,
# but is permitted to depend on things at the top of the network. It is what makes
# the networks recurrent.
#
# It requires a "var" argument that represents the initial state of this
# node.
class RecurrenceLayer(Layer):
    def __init__(self, var, length, recurrence = None):
        self.var = var
        self.length = length

        if recurrence is not None and recurrence.length != length:
            raise ValueError("Recurrence length does not match given length.")

        self.recurrence = recurrence

        self.inputs = set([var])
        if recurrence is not None:
            self.dependencies = set([recurrence])
        else:
            self.dependencies = set([])
        self.params = set([])

        self.unfold_cache = {}

    # Setter for the recurrence; also add this recurrence as a dependency
    def set_recurrence(self, recurrence):
        if recurrence.length != self.length:
            raise ValueError("Recurrence length does not match given length.")

        self.recurrence = recurrence
        self.dependencies = set([recurrence])

    # If we can (i.e. if this is not the first step),
    #return the node at the top of the network
    # on which we depend. Otherwise, return the initial state.
    def _unfold(self, i):
        if i > 0:
            return self.recurrence.unfold(i - 1)
        else:
            return self.var

# UNARY OPERATION LAYERS
# ======================
# TransformLayer -- the basic fully-connected neural network layer.
class TransformLayer(Layer):
    def __init__(self,
                rng, # Initialization RNG
                input, # Input layer
                length, # Output dimensionality

                activation=T.tanh, # Activation function
                W = None, # Initial edges, if desired
                B = None # Initial bias, if desired
            ):
        # Remember inputs
        self.input = input
        self.length = length
        self.activation = activation

        # Initialize W randomly.
        if W is None:
            # W is an (input length) x (output length) matrix
            if activation is T.nnet.softmax:
                W_values = numpy.zeros((input.length, length), dtype=theano.config.floatX)
            else:
                W_values = numpy.asarray(
                    rng.uniform(
                        low=-numpy.sqrt(6. / (input.length + length)),
                        high=numpy.sqrt(6. / (input.length + length)),
                        size=(input.length, length)
                    ),
                    dtype=theano.config.floatX
                )

            # Apparently, when using a sigmoid activation function,
            # you want to use initlialization values four times greater in
            # magnitude.
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        # Initialize B as zeros.
        if B is None:
            # b is a vectory of size (output length)
            B_values = numpy.zeros((length,), dtype=theano.config.floatX)
            B = theano.shared(value=B_values, name='b', borrow=True)

        self.B = B
        self.W = W

        self.params = set([W, B])
        self.dependencies = set([input])
        self.inputs = set([])

        self.unfold_cache = {}

    def _unfold(self, i):
        lin_output = T.dot(self.input.unfold(i), self.W) + self.B
        return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )

# MapLayer -- apply an elemwise function.
class MapLayer(Layer):
    def __init__(self, input, func):
        self.input = input
        self.func = func

        self.params = set([])
        self.dependencies = set([input])
        self.inputs = set([])

        self.unfold_cache = {}

    def _unfold(self, i):
        return self.func(self.input.unfold(i))

# BINARY OPERATION LAYERS
# ======================
# Simple binary ops.
class ConcatLayer(Layer):
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

        self.length = input1.length + input2.length

        self.params = set([])
        self.dependencies = set([input1, input2])
        self.inputs = set([])

        self.unfold_cache = {}

    def _unfold(self, i):
        return T.concatenate(self.input1.unfold(i), self.input2.unfold(i))

class AddLayer(Layer):
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

        if input1.length != input2.length:
            raise ValueError("Cannot add %r and %r of lengths %d != %d" %
                    (input1, input2, input1.length, input2.length))
        else:
            self.length = input1.length

        self.params = set([])
        self.dependencies = set([input1, input2])
        self.inputs = set([])

        self.unfold_cache = {}

    def _unfold(self, i):
        return self.input1.unfold(i) + self.input2.unfold(i)

def MulLayer(Layer):
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

        if input1.length != input2.length:
            raise ValueError("Cannot mul %r and %r of lengths %d != %d" %
                    (input1, input2, input1.length, input2.length))
        else:
            self.length = input1.length

        self.params = set([])
        self.dependencies = set([input1, input2])
        self.inputs = set([])

        self.unfold_cache = {}

    def _unfold(self, i):
        return self.input1.unfold(i) * self.input2.unfold(i)

# TRAINER
# =======
class Trainer(object):
    def __init__(self, layer, k, lr = 0.01):
        self.params = layer.collect_params()

        print(self.params)

        self.inputs = layer.collect_inputs()

        print(self.inputs)

        # The expression is going to be a k x n matrix,
        # where k is the given time length and n is the length of the
        # output vector.
        self.expression = T.stack(list(map(layer.unfold, range(k))))

        # The above scan() loop outputs a tensor of dimensions k x 1 x n;
        # reshape to get rid of the extra dimension
        self.expression = self.expression.dimshuffle(0, 2)

        # The target matrix should have the same type as the output expression
        self.target = self.expression.type()
        self.cost = T.nnet.categorical_crossentropy(self.expression, self.target).mean()

        # Take the derivative of the cost with respect to all the trainable parameters
        self.gparams = [
            T.grad(self.cost, param)
            for param in self.params
        ]

        # Write expressions for all the gradient descent updates
        self.updates = [
            (param, param - lr * gparam)
            for param, gparam in zip(self.params, self.gparams)
        ]

        self.function_inputs = [self.target] + list(self.inputs)

        self._train = theano.function(
            inputs = self.function_inputs,
            outputs = self.cost,
            updates = self.updates
        )

    def train(self, input_dict, target):
        true_input_vector = [target]
        for input_name in self.inputs:
            true_input_vector.append(input_dict[input_name])
        return self._train(*true_input_vector)
