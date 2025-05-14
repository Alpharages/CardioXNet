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
        try:
            # Convert the tensor to a string
            path_str = image_path.numpy().decode('utf-8')

            # Check if the file exists, if not, try adding .npy extension
            if not os.path.exists(path_str) and not path_str.endswith('.npy'):
                path_str = path_str + '.npy'

            # Load the image
            try:
                image = np.load(path_str)
                if image is None or image.size == 0:
                    print(f"Warning: Empty image loaded from {path_str}")
                    return np.zeros((224, 224, 3), dtype=np.float32)
                return image
            except FileNotFoundError:
                print(f"Warning: File not found: {path_str}")
                return np.zeros((224, 224, 3), dtype=np.float32)
            except Exception as e:
                print(f"Error loading image {path_str}: {e}")
                return np.zeros((224, 224, 3), dtype=np.float32)
        except Exception as e:
            print(f"Error in load_image: {e}")
            return np.zeros((224, 224, 3), dtype=np.float32)

    def parse_function(image_path, label):
        try:
            image = tf.py_function(load_image, [image_path], tf.float32)
            image.set_shape([224, 224, 3])
            return image, label
        except Exception as e:
            print(f"Error in parse_function: {e}")
            return tf.zeros([224, 224, 3]), label

    # Load metadata - handle both DataFrame and path to CSV
    if isinstance(metadata_path, pd.DataFrame):
        metadata = metadata_path
    else:
        metadata = pd.read_csv(metadata_path)
    split_metadata = metadata[metadata['split'] == split]
    
    print(f"\nCreating {split} dataset:")
    print(f"Number of samples: {len(split_metadata)}")
    print(f"Sample paths: {split_metadata['image_path'].head()}")

    # Create dataset with repeat and shuffle
    dataset = tf.data.Dataset.from_tensor_slices(
        (split_metadata['image_path'], split_metadata['labels'])
    )
    
    # Add shuffling for training set
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=min(len(split_metadata), 1000), reshuffle_each_iteration=True)
    
    # Map, batch, and repeat
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Repeat indefinitely
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def main():
    try:
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
        print("\nProcessed metadata summary:")
        print(f"Total samples: {len(processed_metadata)}")
        print(f"Training samples: {len(processed_metadata[processed_metadata['split'] == 'train'])}")
        print(f"Validation samples: {len(processed_metadata[processed_metadata['split'] == 'val'])}")
        print(f"Test samples: {len(processed_metadata[processed_metadata['split'] == 'test'])}")

        # Create datasets with smaller batch size for better generalization
        print("\nCreating datasets...")
        batch_size = 16  # Reduced batch size for better generalization

        print("Creating training dataset...")
        train_dataset = create_dataset(
            processed_metadata,
            batch_size=batch_size,
            split='train'
        )

        print("Creating validation dataset...")
        val_dataset = create_dataset(
            processed_metadata,
            batch_size=batch_size,
            split='val'
        )

        print("Creating test dataset...")
        test_dataset = create_dataset(
            processed_metadata,
            batch_size=batch_size,
            split='test'
        )

        # Initialize model with custom CNN architecture
        print("\nInitializing model with custom CNN architecture...")
        model = CardiacFailureModel()

        # Add L2 regularization to all dense layers to prevent overfitting
        for layer in model.model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.0001)

        # Recompile the model with the added regularization
        model.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        # Print model summary
        model.model.summary()

        # Calculate class weights to handle potential class imbalance
        print("\nCalculating class weights...")
        train_labels = []
        steps = len(processed_metadata[processed_metadata['split'] == 'train']) // batch_size
        for i, (_, labels) in enumerate(train_dataset):
            if i >= steps:
                break
            train_labels.extend(labels.numpy().flatten())

        # Count occurrences of each class
        class_counts = np.bincount(np.array(train_labels, dtype=int))
        total_samples = len(train_labels)

        # Calculate class weights: inversely proportional to class frequency
        class_weights = {
            0: total_samples / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
            1: total_samples / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
        }

        print(f"\nUsing class weights: {class_weights}")

        # Set up enhanced callbacks
        print("\nSetting up callbacks...")
        callbacks = [
            # Save best model based on accuracy
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(results_dir / 'best_model_accuracy.keras'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            # Save best model based on AUC
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(results_dir / 'best_model_auc.keras'),
                save_best_only=True,
                monitor='val_auc',
                mode='max'
            ),
            # Early stopping with increased patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Monitor accuracy instead of loss
                patience=10,  # Increased patience
                restore_best_weights=True
            ),
            # Enhanced TensorBoard callback
            tensorboard_callback,
            # Cosine decay learning rate scheduler
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: float(lr * np.cos(epoch / 50 * np.pi / 2))
                if epoch > 0 else float(lr)
            ),
            # Backup learning rate reduction on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,  # Increased patience
                min_lr=1e-7  # Lower minimum learning rate
            )
        ]

        # Train the model with class weights
        print("\nStarting training...")
        print(f"Training samples: {len(processed_metadata[processed_metadata['split'] == 'train'])}")
        print(f"Validation samples: {len(processed_metadata[processed_metadata['split'] == 'val'])}")
        print(f"Steps per epoch: {len(processed_metadata[processed_metadata['split'] == 'train']) // batch_size}")
        print(f"Validation steps: {len(processed_metadata[processed_metadata['split'] == 'val']) // batch_size}")
        print(f"Batch size: {batch_size}")
        print(f"Number of epochs: 60")
        print("\nTraining progress:")

        history = model.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=60,  # Increased epochs for better training
            callbacks=callbacks,
            class_weight=class_weights,
            steps_per_epoch=len(processed_metadata[processed_metadata['split'] == 'train']) // batch_size,
            validation_steps=len(processed_metadata[processed_metadata['split'] == 'val']) // batch_size,
            verbose=1  # Ensure verbose output
        )

        # Print training summary
        print("\nTraining completed!")
        print("\nFinal metrics:")
        print(f"Training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Training AUC: {history.history['auc'][-1]:.4f}")
        print(f"Validation AUC: {history.history['val_auc'][-1]:.4f}")

        # Evaluate the model
        print("\nEvaluating model on test set...")
        metrics_summary = evaluate_model(
            model.model,
            test_dataset,
            history=history,
            results_dir=results_dir / 'model_evaluation'
        )

        # Print summary metrics
        print("\nModel Performance Summary:")
        for metric, value in metrics_summary.items():
            print(f"{metric}: {value:.4f}")

        # Save final model in Keras format
        model.save(results_dir / 'final_model.keras')
        print("\nTraining and evaluation complete!")
        print(f"\nTo view TensorBoard, run: tensorboard --logdir={log_dir}")

        # Print a message about the enhanced custom CNN model
        print("\nModel Architecture Information:")
        print("Using an enhanced custom CNN architecture with the following features:")
        print("1. Deep network with 4 convolutional blocks (8 convolutional layers total)")
        print("2. Increased filters (64→128→256→512) for better feature extraction")
        print("3. Double convolutional layers in each block for improved representation")
        print("4. Same padding to preserve spatial information")
        print("5. Increased capacity in dense layers (1024→512 neurons)")
        print("6. Batch normalization and dropout for regularization")
        print("7. L2 regularization on dense layers to prevent overfitting")
        print("8. Class weights to handle potential class imbalance")
        print("9. Cosine decay learning rate scheduling")
        print("\nThis custom architecture is designed to achieve high accuracy without using pre-trained models or fine-tuning.")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
