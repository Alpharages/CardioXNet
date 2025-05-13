import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import albumentations as A
from pathlib import Path
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChestXRayPreprocessor:
    def __init__(self, data_dir=None, output_dir=None):
        # Use environment variables if parameters are not provided
        self.data_dir = Path(data_dir if data_dir is not None else os.getenv('DATA_DIR', '../../data'))
        self.output_dir = Path(output_dir if output_dir is not None else os.getenv('PROCESSED_DATA_DIR', '../../data/processed'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define image transformations for training
        self.train_transform = A.Compose([
            A.Resize(224, 224),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Define image transformations for validation/test
        self.val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_metadata(self, metadata_path):
        """Load the dataset metadata."""
        return pd.read_csv(metadata_path)

    def preprocess_image(self, image_path, is_training=True):
        """Preprocess a single X-ray image."""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply transformations
            transform = self.train_transform if is_training else self.val_transform
            transformed = transform(image=image)
            return transformed['image']
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def preprocess_dataset(self, metadata_path, image_dir):
        """Preprocess the entire dataset."""
        # Load metadata
        metadata = self.load_metadata(metadata_path)

        # Create processed data directories
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        test_dir = self.output_dir / 'test'

        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(exist_ok=True)

        # Split data
        train_df, temp_df = train_test_split(metadata, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Process each split
        splits = {
            'train': (train_df, train_dir, True),
            'val': (val_df, val_dir, False),
            'test': (test_df, test_dir, False)
        }

        processed_data = []
        for split_name, (df, output_dir, is_training) in splits.items():
            split_data = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name} set"):
                image_path = Path(image_dir) / row['filename']
                if not image_path.exists():
                    continue

                # Preprocess image
                processed_image = self.preprocess_image(image_path, is_training)
                if processed_image is not None:
                    # Save processed image
                    output_path = output_dir / f"processed_{row['filename']}"
                    np.save(output_path, processed_image)

                    # Add to processed data
                    split_data.append({
                        'image_path': str(output_path),
                        'labels': int(row['label']),  # Convert label to integer
                        'split': split_name
                    })

            # Save split metadata
            split_df = pd.DataFrame(split_data)
            split_df.to_csv(output_dir / 'metadata.csv', index=False)
            processed_data.extend(split_data)

        # Save combined metadata
        processed_metadata = pd.DataFrame(processed_data)
        processed_metadata.to_csv(self.output_dir / 'processed_metadata.csv', index=False)

        return processed_metadata

if __name__ == "__main__":
    # Example usage
    base_dir = Path(__file__).parent.parent.parent

    # Use environment variables
    data_dir = Path(os.getenv('DATA_DIR', 'data'))
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    preprocessor = ChestXRayPreprocessor()

    metadata_path = Path(os.getenv('METADATA_FILE', 'data/filename_label.csv'))
    if not metadata_path.is_absolute():
        metadata_path = base_dir / metadata_path

    image_dir = Path(os.getenv('IMAGE_DIR', 'data/images'))
    if not image_dir.is_absolute():
        image_dir = base_dir / image_dir

    print("Starting dataset preprocessing...")
    processed_metadata = preprocessor.preprocess_dataset(metadata_path, image_dir)
    print(f"Preprocessing complete. Processed {len(processed_metadata)} images.") 
