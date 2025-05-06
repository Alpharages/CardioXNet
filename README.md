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

The custom CNN architecture consists of:

1. **Convolutional Blocks** (4 layers):
   - Conv2D layers with increasing filters (32 → 64 → 128 → 256)
   - BatchNormalization for stable training
   - MaxPooling2D for dimensionality reduction
   - Dropout for regularization

2. **Dense Layers**:
   - Two fully connected layers (512 and 256 units)
   - BatchNormalization and Dropout
   - Sigmoid output layer for binary classification

## Data Preprocessing

The preprocessing pipeline includes:

1. **Training Augmentations**:
   - Random rotations and flips
   - Elastic transformations
   - Brightness and contrast adjustments
   - Grid and optical distortions

2. **Validation/Test Transformations**:
   - Resizing to 224x224
   - Normalization using ImageNet statistics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cardiac-failure-detection.git
cd cardiac-failure-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

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

## Model Performance

The model is evaluated using multiple metrics:
- ROC AUC
- Precision-Recall AUC
- Accuracy
- F1 Score
- Confusion Matrix

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ChestX-ray8 dataset from NIH
- TensorFlow and Keras for deep learning framework
- Albumentations for image augmentations
- TensorBoard for training visualization 