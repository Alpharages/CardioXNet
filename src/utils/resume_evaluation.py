import os
from pathlib import Path
import tensorflow as tf
from dotenv import load_dotenv
import pandas as pd

from src.data.preprocess import ChestXRayPreprocessor
from src.utils.evaluation import evaluate_model
from src.training.train import create_dataset

def resume_evaluation():
    """Resume model evaluation from saved model."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Set up paths
        base_dir = Path(__file__).parent.parent.parent
        
        # Use environment variables for paths
        data_dir = Path(os.getenv('DATA_DIR', 'data'))
        if not data_dir.is_absolute():
            data_dir = base_dir / data_dir

        results_dir = Path(os.getenv('RESULTS_DIR', 'results'))
        if not results_dir.is_absolute():
            results_dir = base_dir / results_dir

        # Load the saved model
        print("\nLoading saved model...")
        model_path = results_dir / 'best_model_accuracy.keras'  # or 'best_model_auc.keras'
        if not model_path.exists():
            model_path = results_dir / 'final_model.keras'
        
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model found in {results_dir}")
            
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")

        # Preprocess data
        print("\nPreprocessing data...")
        preprocessor = ChestXRayPreprocessor()

        # Use environment variables for metadata and image paths
        metadata_path = Path(os.getenv('METADATA_FILE', 'data/filename_label.csv'))
        if not metadata_path.is_absolute():
            metadata_path = base_dir / metadata_path

        image_dir = Path(os.getenv('IMAGE_DIR', 'data/images'))
        if not image_dir.is_absolute():
            image_dir = base_dir / image_dir

        processed_metadata = preprocessor.preprocess_dataset(metadata_path, image_dir)
        
        # Create test dataset
        print("\nCreating test dataset...")
        batch_size = 16
        test_dataset = create_dataset(
            processed_metadata,
            batch_size=batch_size,
            split='test'
        )
        
        # Configure test dataset for efficient evaluation
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache()

        # Evaluate the model
        print("\nEvaluating model on test set...")
        metrics_summary = evaluate_model(
            model,
            test_dataset,
            results_dir=results_dir / 'model_evaluation'
        )

        # Print summary metrics
        print("\nModel Performance Summary:")
        for metric, value in metrics_summary.items():
            print(f"{metric}: {value:.4f}")

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    resume_evaluation() 