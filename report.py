import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)

class Reporter:
    """
    Reporter class to generate plots and performance metrics
    for regression or classification tasks.
    """

    def __init__(self,
                 model,
                 X_test,
                 y_test,
                 y_pred=None,
                 task='regression',
                 metrics=None,
                 save_path=None):
        """
        Params:
        - model: The trained model to evaluate.
        - X_test: Test dataset features.
        - y_test: True labels or outputs for the test dataset.
        - y_pred: (Optional) Predicted labels or outputs. If None, we will call model(X_test).
        - task: A string specifying the type of task. Allowed values:
                     'regression', 'binary_classification', 'multiclass_classification'.
        - metrics: A list of additional metrics to compute.
        - save_path: A path where plots will be saved (optional).
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.task = task.lower()
        self.metrics = metrics if metrics is not None else []
        self.save_path = save_path

        # If no predictions provided, generate them
        if self.y_pred is None and self.model is not None:
            self.y_pred = self.model(X_test)

        # Convert shapes if necessary (e.g., for binary classification predictions)
        # If model output is shape (1, N), we might want (N,)
        if isinstance(self.y_pred, np.ndarray) and self.y_pred.ndim == 2:
            # For classification, especially binary, sometimes you have shape (1, N)
            if self.y_pred.shape[0] == 1:
                self.y_pred = self.y_pred.flatten()

        # Flatten y_test if it's 2D with one row
        if isinstance(self.y_test, np.ndarray) and self.y_test.ndim == 2:
            if self.y_test.shape[0] == 1:
                self.y_test = self.y_test.flatten()


    def generate_report(self, history=None):
        """
        Generate a full report including:
          - Loss curves (if history is provided)
          - Parity plots (for regression)
          - Confusion matrix + classification metrics (for classification)
          - Optional sample predictions
        """
        # 1. Loss Curves
        if history is not None:
            self._plot_loss_curves(history)

        # 2. Task-Specific Reporting
        if self.task == 'regression':
            self._report_regression()
        elif self.task in ['binary_classification', 'multiclass_classification']:
            self._report_classification()
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        # Optionally, compute any additional metrics
        self._compute_additional_metrics()

    # ========
    # Plot Loss Curves
    # ========
    def _plot_loss_curves(self, history):
        """
        Plots training and validation loss curves vs. epochs,
        given a history dict with 'train_loss' and 'val_loss' entries.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history and any(v is not None for v in history['val_loss']):
            plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        if self.save_path:
            plt.savefig(f"{self.save_path}/loss_curves.png", dpi=150, bbox_inches='tight')
        plt.show()

    # =========
    # Regression Reporting
    # =========
    def _report_regression(self):
        """
        For regression tasks:
          - Parity plot of predicted vs. actual values
          - Compute and display basic regression metrics (e.g., MSE)
        """
        # Parity Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 color='red', linestyle='--')
        plt.title("Parity Plot (Regression)")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)

        if self.save_path:
            plt.savefig(f"{self.save_path}/parity_plot.png", dpi=150, bbox_inches='tight')
        plt.show()

        # Basic regression metrics (MSE)
        mse = np.mean((self.y_test - self.y_pred) ** 2)
        print(f"Mean Squared Error (MSE): {mse:.4f}")

    # =========
    # Classification Reporting
    # =========
    def _report_classification(self):
        """
        For classification tasks:
          - Confusion matrix
          - Accuracy, Precision, Recall, F1
        """
        # Convert predictions to class labels if needed
        # For binary classification, threshold at 0.5
        # For multiclass, pick argmax
        if self.task == 'binary_classification':
            # If y_pred is continuous probabilities, threshold at 0.5
            y_pred_labels = (self.y_pred >= 0.5).astype(int)
        else:  # multiclass
            # If y_pred has shape (C, N), get argmax over rows
            if self.y_pred.ndim == 2 and self.y_pred.shape[0] > 1:
                y_pred_labels = np.argmax(self.y_pred, axis=0)
            else:
                # If already class indices
                y_pred_labels = self.y_pred.astype(int)

        cm = confusion_matrix(self.y_test, y_pred_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        if self.save_path:
            plt.savefig(f"{self.save_path}/confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.show()

        # Classification metrics
        acc = accuracy_score(self.y_test, y_pred_labels)
        prec = precision_score(self.y_test, y_pred_labels, average='macro')  # or 'binary' for binary
        rec = recall_score(self.y_test, y_pred_labels, average='macro')
        f1 = f1_score(self.y_test, y_pred_labels, average='macro')

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

    # =========
    # Additional Custom Metrics
    # =========
    def _compute_additional_metrics(self):
        """
        Compute any user-specified metrics passed via self.metrics.
        This is optional and depends on how you define or need extra metrics.
        """
        if not self.metrics:
            return

        print("\nAdditional Metrics:")
        for metric_fn in self.metrics:
            metric_value = metric_fn(self.y_test, self.y_pred)
            print(f"{metric_fn.__name__}: {metric_value:.4f}")
