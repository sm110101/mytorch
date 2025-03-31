import numpy as np 

# =============
# Activation Functions
# =============

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

def softmax(x):
    # Subtracting max for numerical stability
    exps = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)




# =============
# Base Class
# =============

class Model:
    """
    Base class from which all of our models (linear, logistic, Dense Feed-Forward)
    Will inherit

    Tasks:
        - Store model params (W, b) in dictionary
        - Define forward pass
        - Provide methods to get&set params for optimization
    """

    def __init__(self):
        # Dict to hold parameter name -> np array
        self._parameters = {}
        # Keep track of parameter shapes for flattening/unflattening
        self._param_shapes = {}

    def forward(self, x):
        """
        Perform forward pass
        Subclasses must override this method 
        """
        raise NotImplementedError("Forward pass not implemented in base Model class.")
    
    def __call__(self, x):
        """
        For convenience -- allows model instances to be called like a function
        """
        return self.forward(x)
    
    def get_parameters(self):
        """
        Returns model parameters dict.
        Used when I want to directly inspect or manipulate parameters
        """
        return self._parameters
    
    def set_parameters(self, param_dict):
        """
        Replaces current params with those in params_dict
        *params_dict should match with shape/keys of self._parameters
        """
        for key, value in param_dict.items():
            if key not in self._parameters:
                raise KeyError(f"Parameter {key} not found in the model.")
            if value.shape != self._parameters[key].shape:
                raise ValueError(f"Shape mismatch for parameter {key}. "
                                 f"Expected {self._parameters[key].shape}, got {value.shape}.")
            
            self._parameters[key] = value

    def get_flat_parameters(self):
        """
        Returns a 1D numpy array of all parameters concatenated
        Also stores shapes for reconstructing them later if needed
        """
        flat_params = []
        self._param_shapes = {}
        start = 0
        for key, value in self._parameters.items():
            shape = value.shape
            size = value.size
            self._param_shapes[key] = (shape, start, start + size)
            flat_params.append(value.ravel())
            start += size
        return np.concatenate(flat_params)
    
    def set_flat_parameters(self, flat_params):
        """
        Takes a 1D numpy array of parameters (matching get_flat_parameters)
        and sets each parameter in the model accordingly.
        """
        if not self._param_shapes:
            raise RuntimeError("Parameter shapes not found. Call get_flat_parameters() first.")
        for key, (shape, start, end) in self._param_shapes.items():
            size = end - start
            self._parameters[key] = flat_params[start:end].reshape(shape)


# =============
# Linear Model
# =============
            
class LinearModel(Model):
    """
    Simple linear model: y = Wx + b
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # initialize params
        self._parameters['W'] = np.random.randn(output_dim, input_dim)
        self._parameters['b'] = np.random.randn(output_dim, 1)

    def forward(self, x):
        """
        Forward pass for the linear model
        Expects x to be shape (input_dim, N) or (input_dim,)
        depending on usage (batch vs. single example)
        """
        W = self._parameters['W']
        b = self._parameters['b']
        # If x is a single example of shape (input_dim,)
        if x.ndim == 1:
            x = x.reshape(-1, 1) # Make it (input_dim, 1) for matmult
        return np.dot(W, x) + b
    

# =============
# Logistic Regression Model
# =============
    
class LogisticRegression(Model):
    """
    Logistic regression model: y = sigmoid(Wx + b)
    """
    def __init__(self, input_dim):
        super().__init__()
        # Binary classification -> output_dim = 1
        self._parameters['W'] = np.random.randn(1, input_dim)
        self._parameters['b'] = np.random.randn(1, 1)

    def forward(self, x):
        W = self._parameters['W']
        b = self._parameters['b']
        # handling batch v single example
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        z = np.dot(W, x) + b
        return sigmoid(z)
    

# =============
# Dense Feed Forward Network
# =============
    
class DenseFeedForwardNetwork(Model):
    """
    Feed-forward netural network with modular layer sizes and activation functions
    """
    def __init__(self, layer_sizes, activations=None):
        """
        layer_sizes: list of integers specifying the number of neurons in each layer
                     e.g. [input_dim, hidden1, hidden2, ..., output_dim]
        activations: list of activation functions (e.g. [relu, sigmoid])
                     should have length = len(layer_sizes) - 1
                     If None, defaults to usig relu for all hidden layers and sigmoid for final layer (CHANGE IF USING FOR REGRESSION)
        """
        super().__init__()
        self.layer_sizes = layer_sizes

        # If activations not provided, default to relu for each hidden layer and sigmoid for output layer
        if activations is None:
            self.activations = [relu] * (len(layer_sizes) - 2) + [sigmoid]
        else:
            if len(activations) != len(layer_sizes) - 1:
                raise ValueError("Number of activation functions must be one less than number of layers")
            self.activations = activations

        # initialize parameters for each later
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            self._parameters[f'W{i}'] = np.random.randn(out_dim, in_dim) * 0.01
            self._parameters[f'b{i}'] = np.zeros((out_dim, 1)) # Starting bias out at zero

    def forward(self, x):
        """
        Forward pass through all layers of the network
        Expects x to be shape (input_dim, N) or (input_dim,) for a single example
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        a = x
        for i in range(len(self.activations)):
            W = self._parameters[f'W{i}']
            b = self._parameters[f'b{i}']
            z = np.dot(W, a) + b
            a = self.activations[i](z)
        return a


