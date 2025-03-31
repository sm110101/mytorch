import numpy as np

# =============
# Loss Functions
# =============

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error for regression tasks
    """
    return np.mean((y_true - y_pred) ** 2)

def bce_loss(y_true, y_pred, epsilon=1e-12):
    """
    Binary Cross Entropy Loss for binary classification tasks
    Epsilon added to avoid log(0)
    y_true, y_pred ∈ [0, 1]
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def cce_loss(y_true, y_pred, epsilon=1e-12):
    """
    Categorical Cross Entropy Loss for multiclass classification tasks
    Epsilon added to avoid log(0)
    y_true one-hot encoded, y_pred ∈ [0, 1] and sum to 1 across classes
    """
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), ))


# =============
# Finite Difference Gradient Calculation 
# =============

def compute_gradients(model, loss_fn, X, Y,
                      l1_reg=0.0, l2_reg=0.0,
                      epsilon=1e-5, method='two-sided'):
    """
    Computes numerical gradients of the loss function w.r.t model params
    using FD approximation

    Params:
        - model: Instnace of the Model class (with get_flat_parameters, set_flat_parameters)
        - loss_fn: Loss function to call when using FD
        - X: Input data (features)
        - Y: Ground truth labels/targets
        - l1_reg: L1 Regularization coefficient
        - l2_reg: L2 Regularization coefficient
        - epsilon: Step size for finite difference
        - method: 'two-sided' or 'one-sided' depending on whether user want uni- or bi-directional perturbations
    Returns:
        - 1D NumPy array of gradients with respect to the flattened model parameters
    """
    original_params = model.get_flat_parameters()
    num_params = len(original_params)
    grads = np.zeros(num_params)

    def total_loss(params):
        # Helper to compute total loss (data loss + regularization) for given params
        model.set_flat_parameters(params)
        y_pred = model(X)
        data_loss = loss_fn(Y, y_pred)

        l1_term = l1_reg * np.sum(np.abs(params)) # Lasso
        l2_term = l2_reg * 0.5 * np.sum(params**2) # Ridge
        return data_loss + l1_term + l2_term
    
    base_loss = total_loss(original_params)

    for i in range(num_params):
        old_val = original_params[i]

        if method=='two-sided':
            # + eps
            original_params[i] = old_val + epsilon
            loss_plus = total_loss(original_params)
            # - eps
            original_params[i] = old_val - epsilon
            loss_minus = total_loss(original_params)
            # Restore parameter
            original_params[i] = old_val

            # Append FD approximation to gradients
            grads[i] = (loss_plus - loss_minus) / (2.0 * epsilon)

        else: # one-sided
            # + eps
            original_params[i] = old_val + epsilon
            loss_plus = total_loss(original_params)
            # restore param
            original_params[i] = old_val

            # Append FD approximation 
            grads[i] = (loss_plus - base_loss) / epsilon

    # Restore original params
    model.set_flat_parameters(original_params)
    return grads


# =============
# Optimization using Gradient Descent 
# =============

class Optimizer:
    """
    Optimizer class that handles:
        - FD gradient computation
        - Optimization algorithms (GD, Momentum, RMSProp)
        - Early Stopping
        - Regularization (L1, L2)
    """
    def __init__(self, model, loss_fn,
                 lr=1e-2,
                 l1_reg=0.0,
                 l2_reg=0.0,
                 method='gd',
                 epsilon=1e-5,
                 grad_approx_method='two-sided',
                 beta=0.9,
                 beta2=0.999):
        """
        Params:
            - model: The model to optimize (instance of Model class)
            - loss_fn: Loss function to use (mse, bce, cce)
            - lr: learning rate
            - l1_reg: L1 Regularization coefficient
            - l2_reg: L2 Regularization coefficient
            - method: 'gd', 'momentum', 'rmsprop'
            - epsilon: small C for stability in RMSprop
            - grad_approx_method: 'two-sided' or 'one-sided' for finite difference calculation
            - beta: Momentum/RMSprop parameter 
            - beta2: additional param
        """
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.method = method
        self.epsilon = epsilon
        self.grad_approx_method = grad_approx_method

        # For momentum
        self.beta = beta
        self.velocity = None # Will init upon first update

        # For RMSprop
        self.beta2 = beta2
        self.sq_grad_avg = None # will init upon first update

    def step(self, X, Y):
        """
        Performs a single optimization step on the model params
        using FD calculations and the chosen update method
        """
        # 1. Compute Gradients
        grads = compute_gradients(
            model=self.model,
            loss_fn=self.loss_fn,
            X=X,
            Y=Y,
            l1_reg=self.l1_reg,
            l2_reg=self.l2_reg,
            epsilon=self.epsilon,
            method=self.grad_approx_method
        )

        # 2. Update params based on self.method
        current_params = self.model.get_flat_parameters()

        if self.method == 'gd':
            # Gradient Descent
            new_params = current_params - self.lr * grads
        elif self.method == 'momentum':
            # Momentum
            if self.velocity is None:
                self.velocity = np.zeros_like(current_params)
            self.velocity = self.beta * self.velocity + (1 - self.beta) * grads
            new_params = current_params - self.lr * self.velocity
        elif self.method == 'rmsprop':
            # RMSprop
            if self.sq_grad_avg is None:
                self.sq_grad_avg = np.zeros_like(current_params)
            self.sq_grad_avg = self.beta * self.sq_grad_avg + (1 - self.beta) * (grads**2)
            denom = np.sqrt(self.sq_grad_avg) + self.epsilon
            new_params = current_params - self.lr * grads/denom
        else:
            raise ValueError(f"Unknown optimization method '{self.method}'")
        
        # 3. Set updated parameters
        self.model.set_flat_parameters(new_params)

    def compute_loss(self, X, Y):
        """
        Compute the current loss (including regularization)
        Will be used to track training/validation loss over epochs
        """
        # Evaluate forward pass
        y_pred = self.model(X)
        data_loss = self.loss_fn(Y, y_pred)

        # L1 and L2 terms on flattened params
        flat_params = self.model.get_flat_parameters()
        l1_term = self.l1_reg * np.sum(np.abs(flat_params))
        l2_term = self.l2_reg * 0.5 * np.sum(flat_params ** 2)

        return data_loss + l1_term + l2_term
    


# =============
# Fit with early stopping
# =============

    def fit(self, X_train, y_train,
            X_val=None, y_val=None,
            epochs=100,
            early_stopping_patience=5):
        """
        Training loop with optional early stopping based on 
        validation loss
        """
        best_val_loss = float('inf')
        patience_counter = 0

        history = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(epochs):
            # Single update step
            self.step(X_train, y_train)

            # Compute training loss
            train_loss = self.compute_loss(X_train, y_train)
            history['train_loss'].append(train_loss)

            #Compute validation loss if provided
            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(X_val, y_val)
                history['val_loss'].append(val_loss)

                # Early stoppage check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            else:
                # If no validation set if provided, skip early stopping
                history['val_loss'].append(None)

            # Print progress
            if epoch%10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}",
                      end='')
                if X_val is not None and y_val is not None:
                    print(f" | Val Loss: {val_loss:.4f}")
                else:
                    print()

        return history