import numpy as np

import backend
import nn


class Model(object):
    """Base model class for the different applications"""

    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)


class RegressionModel(Model):
    """
    TODO: Question 4 - [Application] Regression

    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.nodes = []
        self.learning_rate = .1

    def run(self, x, y=None):
        """
        TODO: Question 4 - [Application] Regression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"

        if y is not None:
            "*** YOUR CODE HERE ***"
            w1 = nn.Variable(len(x), len(x))
            w2 = nn.Variable(len(x), len(x))

            b1 = nn.Variable(len(x), 1)
            b2 = nn.Variable(len(x), 1)

            self.nodes = nn.Graph([w1, w2, b1, b2])
            input_x = nn.Input(self.nodes, x)

            xw1 = nn.MatrixMultiply(self.nodes, w1, input_x)
            xw1_plus_b1 = nn.MatrixVectorAdd(self.nodes, xw1, b1)
            relu_xw1b1 = nn.ReLU(self.nodes, xw1_plus_b1)
            input_y = nn.Input(self.nodes, y)
            loss1 = nn.SquareLoss(self.nodes, relu_xw1b1, input_y)

            xw2 = nn.MatrixMultiply(self.nodes, w2, input_x)
            xw2_plus_b2 = nn.MatrixVectorAdd(self.nodes, xw2, b2)
            relu_xw2b2 = nn.ReLU(self.nodes, xw2_plus_b2)
            loss2 = nn.SquareLoss(self.nodes, relu_xw2b2, input_y)

            nn.Add(self.nodes, loss1, loss2)
            return self.nodes
        else:
            "*** YOUR CODE HERE ***"
            pred1 = self.nodes.get_output(self.nodes.get_nodes()[-7])
            return pred1


class OddRegressionModel(Model):
    """
    TODO: Question 5 - [Application] OddRegression

    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.nodes = []
        self.inputs = []
        self.learning_rate = .1

    def run(self, x, y=None):
        """
        TODO: Question 5 - [Application] OddRegression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        if not self.nodes:
            w1 = nn.Variable(1, 50)
            w2 = nn.Variable(50, 50)
            w3 = nn.Variable(50, 1)
            b1 = nn.Variable(1, 50)
            b2 = nn.Variable(1, 50)
            b3 = nn.Variable(1, 1)
            self.nodes = nn.Graph([w1, w2, w3, b1, b2, b3])
            self.inputs = [w1, w2, w3,  b1, b2, b3]
        w1 = self.inputs[0]
        w2 = self.inputs[1]
        w3 = self.inputs[2]
        b1 = self.inputs[3]
        b2 = self.inputs[4]
        b3 = self.inputs[5]
        self.nodes = nn.Graph(self.inputs)
        input_x = nn.Input(self.nodes, x)
        if y is not None:
            input_y = nn.Input(self.nodes, y)

        negation = nn.Input(self.nodes, np.matrix([-1.0]))

        xw1 = nn.MatrixMultiply(self.nodes, input_x, w1)
        xw1_plus_b1 = nn.MatrixVectorAdd(self.nodes, xw1, b1)
        relu_xw1b1 = nn.ReLU(self.nodes, xw1_plus_b1)



        xw2 = nn.MatrixMultiply(self.nodes, relu_xw1b1, w2)
        xw2_plus_b2 = nn.MatrixVectorAdd(self.nodes, xw2, b2)
        relu_xw2b2 = nn.ReLU(self.nodes, xw2_plus_b2)

        xw3 = nn.MatrixMultiply(self.nodes, relu_xw2b2, w3)
        final1 = nn.MatrixVectorAdd(self.nodes, xw3, b3)


        #deep breath, now calculations for negative x (might put this on a loop  if i have time)
        x_neg = nn.MatrixMultiply(self.nodes, input_x, negation)
        xw1 = nn.MatrixMultiply(self.nodes, x_neg)
        xw1_plus_b1 = nn.MatrixVectorAdd(self.nodes, xw1, b1)
        relu_xw1b1 = nn.ReLU(self.nodes, xw1_plus_b1)

        xw2 = nn.MatrixMultiply(self.nodes, relu_xw1b1, w2)
        xw2_b2 = nn.MatrixVectorAdd(self.nodes, xw2, b2)
        relu_xw1b1 = nn.ReLU(self.nodes, xw2_b2)

        xw3 = nn.MatrixMultiply(self.nodes, relu_xw1b1, w3)
        xw3_b3 = nn.MatrixVectorAdd(self.nodes, xw3, b3)
        final2 = nn.MatrixMultiply(self.nodes, xw3_b3, negation)


        final = nn.MatrixVectorAdd(self.nodes, final1, final2)

        if y is not None:
            "*** YOUR CODE HERE ***"
            loss = nn.SquareLoss(self.nodes, final, input_y)
            return self.nodes

        else:
            "*** YOUR CODE HERE ***"
            return self.nodes.get_output(self.nodes.get_nodes()[-1])


class DigitClassificationModel(Model):
    """
    TODO: Question 6 - [Application] Digit Classification

    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.25
        self.nodes = None

    def run(self, x, y=None):
        """
        TODO: Question 6 - [Application] Digit Classification

        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"

        if not self.nodes:
            w1 = nn.Variable(x.shape[1], x.shape[0])
            w2 = nn.Variable(x.shape[0], x.shape[0])
            w3 = nn.Variable(x.shape[0], 10)
            b1 = nn.Variable(1, x.shape[0])
            b2 = nn.Variable(1, x.shape[0])
            b3 = nn.Variable(1, 10)
            self.nodes = nn.Graph([w1, w2, w3, b1, b2, b3])
            self.inputs = [w1, w2, w3,  b1, b2, b3]
        w1 = self.inputs[0]
        w2 = self.inputs[1]
        w3 = self.inputs[2]
        b1 = self.inputs[3]
        b2 = self.inputs[4]
        b3 = self.inputs[5]
        self.nodes = nn.Graph([w1, w2, w3, b1, b2, b3])
        input_x = nn.Input(self.nodes, x)

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(self.nodes, y)

        xw1 = nn.MatrixMultiply(self.nodes, input_x, w1)
        xw1b1 = nn.MatrixVectorAdd(self.nodes, xw1, b1)
        xw1relu = nn.ReLU(self.nodes, xw1b1)

        xw2 = nn.MatrixMultiply(self.nodes, xw1relu, w2)
        xw2b2 = nn.MatrixVectorAdd(self.nodes, xw2, b2)
        xw2relu = nn.ReLU(self.nodes, xw2b2)

        xw3 = nn.MatrixMultiply(self.nodes, xw2relu, w3)
        final = nn.MatrixVectorAdd(self.nodes, xw3, b3)

        if y is not None:
            "*** YOUR CODE HERE ***"
            loss = nn.SoftmaxLoss(self.nodes, final, input_y)
            return self.nodes
        else:
            "*** YOUR CODE HERE ***"
            return self.nodes.get_output(self.nodes.get_nodes()[-1])


class DeepQModel(Model):
    """
    TODO: Question 7 - [Application] Reinforcement Learning

    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.nodes = None


    def run(self, states, Q_target=None):
        """
        TODO: Question 7 - [Application] Reinforcement Learning

        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        if not self.nodes:
            w1 = nn.Variable(states.shape[1], states.shape[0])
            w2 = nn.Variable(states.shape[0], states.shape[0])
            w3 = nn.Variable(states.shape[0], 2)
            b1 = nn.Variable(1, states.shape[0])
            b2 = nn.Variable(1, states.shape[0])
            b3 = nn.Variable(1, 2)
            self.nodes = nn.Graph([w1, w2, w3, b1, b2, b3])
            self.inputs = [w1, w2, w3,  b1, b2, b3]
        w1 = self.inputs[0]
        w2 = self.inputs[1]
        w3 = self.inputs[2]
        b1 = self.inputs[3]
        b2 = self.inputs[4]
        b3 = self.inputs[5]
        self.nodes = nn.Graph([w1, w2, w3, b1, b2, b3])
        input_x = nn.Input(self.nodes, states)

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(self.nodes, Q_target)

        xw1 = nn.MatrixMultiply(self.nodes, input_x, w1)
        xw1b1 = nn.MatrixVectorAdd(self.nodes, xw1, b1)
        xw1relu = nn.ReLU(self.nodes, xw1b1)

        xw2 = nn.MatrixMultiply(self.nodes, xw1relu, w2)
        xw2b2 = nn.MatrixVectorAdd(self.nodes, xw2, b2)
        xw2relu = nn.ReLU(self.nodes, xw2b2)

        xw3 = nn.MatrixMultiply(self.nodes, xw2relu, w3)
        final = nn.MatrixVectorAdd(self.nodes, xw3, b3)

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            loss = nn.SquareLoss(self.nodes, final, input_y)
            return self.nodes
        else:
            "*** YOUR CODE HERE ***"
            return self.nodes.get_output(self.nodes.get_nodes()[-1])

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    TODO: Question 8 - [Application] Language Identification

    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars  = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        #between .01 - .005
        self.learning_rate = 0.028
        self.nodes = None

    def run(self, xs, y=None):
        """
        TODO: Question 8 - [Application] Language Identification

        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "*** YOUR CODE HERE ***"
        if not self.nodes:
            w1 = nn.Variable(self.num_chars, self.num_chars)
            w2 = nn.Variable(self.num_chars, self.num_chars)
            w3 = nn.Variable(self.num_chars, self.num_chars)
            w4 = nn.Variable(self.num_chars, 5)
            b1 = nn.Variable(1,self.num_chars)
            b2 = nn.Variable(1, self.num_chars)
            b3 = nn.Variable(1, self.num_chars)
            b4 = nn.Variable(1, 5)
            h = nn.Variable(1, self.num_chars)
            bonusw = nn.Variable(self.num_chars, self.num_chars)
            bonusb = nn.Variable(1, self.num_chars)
            self.nodes = nn.Graph([w1, w2, w3, w4, b1, b2, b3, b4, h, bonusw, bonusb])
            self.inputs = [w1, w2, w3, w4,  b1, b2, b3, b4, h, bonusw, bonusb]
        w1 = self.inputs[0]
        w2 = self.inputs[1]
        w3 = self.inputs[2]
        w4 = self.inputs[3]
        b1 = self.inputs[4]
        b2 = self.inputs[5]
        b3 = self.inputs[6]
        b4 = self.inputs[7]
        h = self.inputs[8]
        bonusw = self.inputs[9]
        bonusb = self.inputs[10]
        self.nodes = nn.Graph([w1, w2, w3, w4, b1, b2, b3, b4, h])
        zeros = nn.Input(self.nodes, np.zeros((batch_size, self.num_chars)))
        h = nn.MatrixVectorAdd(self.nodes, zeros, h)

        word = []
        for s in xs:
            ch = nn.Input(self.nodes, s)
            h_sum = nn.MatrixVectorAdd(self.nodes, h, ch)
            hw1 = nn.MatrixMultiply(self.nodes, h_sum, w1)
            hw1b1 = nn.MatrixVectorAdd(self.nodes, hw1, b1)
            h = nn.ReLU(self.nodes, hw1b1)

            #hw2b = nn.MatrixMultiply(self.nodes, hw1relu, bonusw)
            #h2 = nn.MatrixVectorAdd(self.nodes, hw2b, bonusb)
            #h = nn.ReLU(self.nodes, hw2b2b)
            word.append(ch)

        hw2 = nn.MatrixMultiply(self.nodes, h, w2)
        hw2b2 = nn.MatrixVectorAdd(self.nodes, hw2, b2)
        hw2relu = nn.ReLU(self.nodes, hw2b2)

        hw3 = nn.MatrixMultiply(self.nodes, hw2relu, w3)
        hw3b3 = nn.MatrixVectorAdd(self.nodes, hw3, b3)
        hw3relu = nn.ReLU(self.nodes, hw3b3)

        hw4 = nn.MatrixMultiply(self.nodes, hw3relu, w4)
        final = nn.MatrixVectorAdd(self.nodes, hw4, b4)


        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(self.nodes, y)
            finalloss = nn.SoftmaxLoss(self.nodes, final, input_y)
            return self.nodes
        else:
            "*** YOUR CODE HERE ***"
            return self.nodes.get_output(self.nodes.get_nodes()[-1])
