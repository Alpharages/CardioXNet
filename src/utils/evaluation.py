import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, model, test_dataset, results_dir='../../results'):
        self.model = model
        self.test_dataset = test_dataset
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Get predictions and true labels
        self.y_true = []
        self.y_pred = []
        self._get_predictions()
    
    def _get_predictions(self):
        """Get predictions and true labels from test dataset efficiently."""
        print("\nGenerating predictions...")
        # Calculate total steps for progress bar
        total_steps = tf.data.experimental.cardinality(self.test_dataset).numpy()
        
        # Use tqdm for progress tracking
        for images, labels in tqdm(self.test_dataset, total=total_steps, desc="Evaluating batches"):
            # Process in batches
            predictions = self.model.predict(images, verbose=0)  # Disable internal progress bar
            self.y_pred.extend(predictions.flatten())
            self.y_true.extend(labels.numpy().flatten())
        
        self.y_pred = np.array(self.y_pred)
        self.y_true = np.array(self.y_true)
        print(f"Processed {len(self.y_true)} samples")
    
    def plot_confusion_matrix(self, threshold=0.5):
        """Plot confusion matrix."""
        y_pred_binary = (self.y_pred >= threshold).astype(int)
        cm = confusion_matrix(self.y_true, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Cardiac Failure'],
                   yticklabels=['Normal', 'Cardiac Failure'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.results_dir / 'confusion_matrix.png')
        plt.close()
    
    def plot_roc_curve(self):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.results_dir / 'roc_curve.png')
        plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(self):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred)
        avg_precision = average_precision_score(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(self.results_dir / 'precision_recall_curve.png')
        plt.close()
        
        return avg_precision
    
    def plot_training_history(self, history):
        """Plot training history metrics."""
        metrics = ['loss', 'accuracy', 'auc']
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history[metric], label=f'Training {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'Model {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.savefig(self.results_dir / f'training_{metric}.png')
            plt.close()
    
    def generate_classification_report(self, threshold=0.5):
        """Generate and save classification report."""
        y_pred_binary = (self.y_pred >= threshold).astype(int)
        report = classification_report(
            self.y_true, y_pred_binary,
            target_names=['Normal', 'Cardiac Failure'],
            output_dict=True
        )
        
        # Save report as text file
        with open(self.results_dir / 'classification_report.txt', 'w') as f:
            f.write(classification_report(
                self.y_true, y_pred_binary,
                target_names=['Normal', 'Cardiac Failure']
            ))
        
        return report
    
    def plot_prediction_distribution(self):
        """Plot distribution of prediction probabilities."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.y_pred, bins=50, kde=True)
        plt.title('Distribution of Prediction Probabilities')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.savefig(self.results_dir / 'prediction_distribution.png')
        plt.close()
    
    def evaluate_model(self, history=None):
        """Run comprehensive evaluation."""
        print("Generating evaluation metrics and visualizations...")
        
        # Generate metrics in parallel using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all plotting tasks
            roc_future = executor.submit(self.plot_roc_curve)
            pr_future = executor.submit(self.plot_precision_recall_curve)
            cm_future = executor.submit(self.plot_confusion_matrix)
            report_future = executor.submit(self.generate_classification_report)
            dist_future = executor.submit(self.plot_prediction_distribution)
            
            if history:
                history_future = executor.submit(self.plot_training_history, history)
            
            # Get results
            roc_auc = roc_future.result()
            avg_precision = pr_future.result()
            report = report_future.result()
        
        # Save summary metrics
        summary = {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }
        
        with open(self.results_dir / 'metrics_summary.txt', 'w') as f:
            for metric, value in summary.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        print("Evaluation complete! Results saved in", self.results_dir)
        return summary

def evaluate_model(model, test_dataset, history=None, results_dir='../../results'):
    """Convenience function to run full evaluation."""
    evaluator = ModelEvaluator(model, test_dataset, results_dir)
    return evaluator.evaluate_model(history) 