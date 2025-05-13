import os
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.data.preprocess import ChestXRayPreprocessor
from src.models.cardiac_model import CardiacFailureModel
from src.utils.evaluation import evaluate_model

def create_dataset(metadata_path, batch_size=32, split='train'):
    """Create TensorFlow dataset from processed data."""
    def load_image(image_path):
        image = np.load(image_path.numpy().decode('utf-8'))
        return image

    def parse_function(image_path, label):
        image = tf.py_function(load_image, [image_path], tf.float32)
        image.set_shape([224, 224, 3])
        return image, label

    # Load metadata
    metadata = pd.read_csv(metadata_path)
    split_metadata = metadata[metadata['split'] == split]

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (split_metadata['image_path'], split_metadata['labels'])
    ).map(parse_function).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def main():
    # Set up paths
    base_dir = Path(__file__).parent.parent.parent

    # Use environment variables for paths
    data_dir = Path(os.getenv('DATA_DIR', 'data'))
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    results_dir = Path(os.getenv('RESULTS_DIR', 'results'))
    if not results_dir.is_absolute():
        results_dir = base_dir / results_dir
    results_dir.mkdir(exist_ok=True)

    # Create TensorBoard log directory with timestamp
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = results_dir / 'logs' / current_time
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard callback with enhanced configuration
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,  # Log weight histograms every epoch
        write_graph=True,  # Log model graph
        write_images=True,  # Log weight images
        update_freq='epoch',  # Log metrics every epoch
        profile_batch='500,520'  # Profile a few batches
    )

    # Preprocess data
    print("Preprocessing data...")
    preprocessor = ChestXRayPreprocessor()  # Will use environment variables

    # Use environment variables for metadata and image paths
    metadata_path = Path(os.getenv('METADATA_FILE', 'data/filename_label.csv'))
    if not metadata_path.is_absolute():
        metadata_path = base_dir / metadata_path

    image_dir = Path(os.getenv('IMAGE_DIR', 'data/images'))
    if not image_dir.is_absolute():
        image_dir = base_dir / image_dir

    processed_metadata = preprocessor.preprocess_dataset(metadata_path, image_dir)

    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dataset(
        processed_metadata,
        batch_size=32,
        split='train'
    )

    val_dataset = create_dataset(
        processed_metadata,
        batch_size=32,
        split='val'
    )

    test_dataset = create_dataset(
        processed_metadata,
        batch_size=32,
        split='test'
    )

    # Initialize model
    print("Initializing model...")
    model = CardiacFailureModel()

    # Print model summary
    model.model.summary()

    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(results_dir / 'best_model.h5'),
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tensorboard_callback,  # Use the enhanced TensorBoard callback
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]

    # Train model
    print("Starting training...")
    history = model.train(
        train_dataset,
        val_dataset,
        epochs=50,  # Increased epochs for custom model
        callbacks=callbacks
    )

    # Evaluate model
    print("Evaluating model...")
    metrics_summary = evaluate_model(
        model.model,
        test_dataset,
        history=history,
        results_dir=results_dir
    )

    # Print summary metrics
    print("\nModel Performance Summary:")
    for metric, value in metrics_summary.items():
        print(f"{metric}: {value:.4f}")

    # Save final model
    model.save(results_dir / 'final_model.h5')
    print("\nTraining and evaluation complete!")
    print(f"\nTo view TensorBoard, run: tensorboard --logdir={log_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))
