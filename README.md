# AI-Based Detection of Cardiac Failure Using Chest X-Ray Images

This project implements a custom Convolutional Neural Network (CNN) for detecting cardiac failure using chest X-ray images. The model is designed to provide early, accessible, and cost-effective diagnosis of cardiac failure, potentially improving patient outcomes through timely intervention.

## Features

- **Custom CNN Architecture**: A 4-layer deep CNN with BatchNormalization, MaxPooling, and Dropout layers
- **Advanced Data Augmentation**: Comprehensive image transformations including rotations, flips, elastic transforms, and brightness adjustments
- **Robust Training Pipeline**: Early stopping, learning rate scheduling, and model checkpointing
- **Comprehensive Evaluation**: Multiple evaluation metrics including ROC curves, precision-recall curves, and confusion matrices
- **Interactive Visualization**: Training history plots and prediction analysis
- **TensorBoard Integration**: Real-time training monitoring and model visualization
- **Google Colab Support**: Ready-to-use notebook for easy experimentation

## Model Architecture

The enhanced custom CNN architecture consists of:

1. **Convolutional Blocks** (4 blocks with 8 convolutional layers total):
   - Double Conv2D layers in each block with increasing filters (64 → 128 → 256 → 512)
   - Same padding to preserve spatial information
   - BatchNormalization for stable training
   - MaxPooling2D for dimensionality reduction
   - Dropout for regularization

2. **Dense Layers**:
   - Two fully connected layers with increased capacity (1024 and 512 units)
   - BatchNormalization and Dropout
   - L2 regularization to prevent overfitting
   - Sigmoid output layer for binary classification

3. **Pre-trained Model Support**:
   - Optional integration with pre-trained models (EfficientNetB0, ResNet50, DenseNet121, MobileNetV2)
   - Architecture-specific adjustments for each pre-trained model
   - Two-phase fine-tuning capability

## Training Process

The training pipeline includes several advanced techniques to improve model performance:

1. **Training Configuration**:
   - Reduced batch size (16) for better generalization
   - Extended training for up to 60 epochs
   - L2 regularization on dense layers to prevent overfitting

2. **Class Imbalance Handling**:
   - Automatic calculation of class weights based on class distribution
   - Weights inversely proportional to class frequency

3. **Advanced Callbacks**:
   - Model checkpointing based on both accuracy and AUC
   - Early stopping with increased patience (10 epochs)
   - Cosine decay learning rate scheduling
   - Learning rate reduction on plateau with factor 0.2
   - Comprehensive TensorBoard logging

4. **Fine-tuning Process** (for pre-trained models):
   - Two-phase approach: train top layers first, then fine-tune unfrozen layers
   - Progressively lower learning rates (0.001 → 0.0001 → 0.00001)
   - Selective layer unfreezing for transfer learning

## Data Preprocessing

The preprocessing pipeline includes:

1. **Training Augmentations**:
   - Resizing to 224x224
   - Random rotations (90°) and flips (horizontal and vertical)
   - Affine transformations (scale, translate, rotate)
   - Elastic transformations, grid distortion, optical distortion
   - Brightness/contrast adjustments, gamma adjustments, hue/saturation/value adjustments, CLAHE
   - Coarse dropout and Gaussian noise
   - Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

2. **Validation/Test Transformations**:
   - Resizing to 224x224
   - Normalization using ImageNet statistics

3. **Dataset Splitting**:
   - 70% training, 15% validation, 15% test

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Alpharages/CardioXNet.git
cd CardioXNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Troubleshooting TensorFlow Installation

If you encounter an error like "The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine", you have several options:

1. **Use the modified training script**: The training script has been updated to try to work around AVX instruction issues.

2. **Install a CPU-specific build of TensorFlow**:
```bash
pip uninstall -y tensorflow
pip install tensorflow-cpu==2.10.0
```

3. **Use Docker**: Run TensorFlow in a Docker container that's compatible with your CPU.

4. **Use Google Colab**: Run the notebook version of this project in Google Colab, which provides a pre-configured environment.

## Environment Variables

This project uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:

```
# Data directories
DATA_DIR=data
PROCESSED_DATA_DIR=data/processed
IMAGE_DIR=data/images

# Results directory
RESULTS_DIR=results

# Metadata file
METADATA_FILE=data/filename_label.csv
```

You can customize these paths according to your setup. A sample `.env.example` file is provided as a template.

## Usage

### Data Preparation

1. Download the ChestX-ray8 dataset from NIH
2. Place the data in the following structure:
```
data/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── Data_Entry_2017.csv
```

### Training

1. Preprocess the dataset:
```bash
python src/data/preprocess.py
```

2. Train the model:
```bash
python src/training/train.py
```

### Monitoring Training with TensorBoard

TensorBoard is integrated into the training pipeline and provides real-time visualization of:

- Training and validation metrics
- Model graph visualization
- Weight distributions and histograms
- Performance profiling
- Learning rate changes

To launch TensorBoard:

1. After starting training, open a new terminal and run:
```bash
tensorboard --logdir=results/logs
```

2. Open your web browser and navigate to:
```
http://localhost:6006
```

The TensorBoard interface will automatically update as training progresses, allowing you to monitor your model's performance in real-time.

### Google Colab

1. Open `cardiac_failure_detection.ipynb` in Google Colab
2. Mount your Google Drive
3. Update the data paths
4. Run the cells sequentially

## Project Structure

```
cardiac-failure-detection/
├── data/
│   ├── images/              # Raw X-ray images
│   └── processed/           # Preprocessed data
├── src/
│   ├── data/
│   │   └── preprocess.py    # Data preprocessing
│   ├── models/
│   │   └── cardiac_model.py # Custom CNN model
│   ├── training/
│   │   └── train.py        # Training pipeline
│   └── utils/
│       └── evaluation.py    # Evaluation metrics
├── results/                 # Saved models and logs
│   └── logs/               # TensorBoard logs
├── requirements.txt         # Project dependencies
├── cardiac_failure_detection.ipynb  # Google Colab notebook
└── README.md               # Project documentation
```

## Model Performance and Evaluation

The model is evaluated using a comprehensive evaluation pipeline that includes:

1. **Performance Metrics**:
   - ROC AUC (Area Under the Receiver Operating Characteristic curve)
   - Precision-Recall AUC (Area Under the Precision-Recall curve)
   - Accuracy (overall correct predictions)
   - Precision (positive predictive value)
   - Recall (sensitivity)
   - F1 Score (harmonic mean of precision and recall)
   - Confusion Matrix (true positives, false positives, true negatives, false negatives)

2. **Visualization Capabilities**:
   - Training history plots (loss, accuracy, AUC over epochs)
   - ROC curves with AUC values
   - Precision-Recall curves with Average Precision values
   - Confusion matrices as heatmaps
   - Prediction probability distribution histograms

3. **Results Saving**:
   - All plots saved as PNG files in the results directory
   - Classification report saved as text file
   - Metrics summary saved as text file
   - Best models saved based on both accuracy and AUC

The evaluation process automatically runs after training and provides a comprehensive assessment of the model's performance on the test dataset.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ChestX-ray8 dataset from NIH
- TensorFlow and Keras for deep learning framework
- Albumentations for image augmentations
- TensorBoard for training visualization 
