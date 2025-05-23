import json
import os

# Define the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {
                "id": "view-in-github",
                "colab_type": "text"
            },
            "source": [
                "<a href=\"https://colab.research.google.com/github/Alpharages/CardioXNet/blob/main/cardiac_failure_detection.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AI-Based Detection of Cardiac Failure Using Chest X-Ray Images\n",
                "\n",
                "This notebook implements a custom Convolutional Neural Network (CNN) for detecting cardiac failure using chest X-ray images. The model is designed to provide early, accessible, and cost-effective diagnosis of cardiac failure, potentially improving patient outcomes through timely intervention.\n",
                "\n",
                "## Table of Contents\n",
                "1. [Setup](#setup)\n",
                "2. [Data Preparation](#data-preparation)\n",
                "3. [Data Preprocessing](#data-preprocessing)\n",
                "4. [Model Building](#model-building)\n",
                "5. [Training](#training)\n",
                "6. [Evaluation](#evaluation)\n",
                "7. [Visualization](#visualization)\n",
                "8. [Inference](#inference)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup <a name=\"setup\"></a>\n",
                "\n",
                "First, let's set up our environment by installing the required dependencies and cloning the repository."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check if running in Colab\n",
                "import sys\n",
                "IN_COLAB = 'google.colab' in sys.modules\n",
                "print(f\"Running in Colab: {IN_COLAB}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install dependencies\n",
                "!pip install -q tensorflow\n",
                "!pip install -q numpy\n",
                "!pip install -q pandas\n",
                "!pip install -q matplotlib\n",
                "!pip install -q scikit-learn\n",
                "!pip install -q opencv-python\n",
                "!pip install -q pillow\n",
                "!pip install -q albumentations\n",
                "!pip install -q tqdm\n",
                "!pip install -q tensorboard\n",
                "!pip install -q python-dotenv\n",
                "!pip install -q seaborn"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone the repository if in Colab\n",
                "if IN_COLAB:\n",
                "    !git clone https://github.com/Alpharages/CardioXNet.git\n",
                "    %cd CardioXNet\n",
                "    \n",
                "    # Create necessary directories\n",
                "    !mkdir -p data/images\n",
                "    !mkdir -p data/processed\n",
                "    !mkdir -p results/logs"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import necessary libraries\n",
                "import os\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import tensorflow as tf\n",
                "from pathlib import Path\n",
                "from datetime import datetime\n",
                "from dotenv import load_dotenv\n",
                "from google.colab import files\n",
                "\n",
                "# Set random seeds for reproducibility\n",
                "np.random.seed(42)\n",
                "tf.random.set_seed(42)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Preparation <a name=\"data-preparation\"></a>\n",
                "\n",
                "For this project, we'll use the ChestX-ray8 dataset from NIH. In Colab, we have two options:\n",
                "1. Upload a small sample dataset directly to Colab\n",
                "2. Mount Google Drive and access a larger dataset stored there\n",
                "\n",
                "Let's implement both options:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Option 1: Upload a small sample dataset directly to Colab\n",
                "def upload_sample_data():\n",
                "    print(\"Please upload your metadata file (filename_label.csv):\")\n",
                "    uploaded = files.upload()\n",
                "    for filename in uploaded.keys():\n",
                "        !mv \"{filename}\" data/filename_label.csv\n",
                "    \n",
                "    print(\"\\nPlease upload your X-ray images (you can select multiple files):\")\n",
                "    uploaded = files.upload()\n",
                "    for filename in uploaded.keys():\n",
                "        !mv \"{filename}\" data/images/\n",
                "    \n",
                "    print(f\"Uploaded {len(uploaded)} images to data/images/\")\n",
                "    return Path('data/filename_label.csv'), Path('data/images')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Option 2: Mount Google Drive and access data stored there\n",
                "def mount_drive_data():\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive')\n",
                "    \n",
                "    # Specify the path to your data in Google Drive\n",
                "    drive_metadata_path = input(\"Enter the path to your metadata file in Google Drive: \")\n",
                "    drive_images_path = input(\"Enter the path to your images folder in Google Drive: \")\n",
                "    \n",
                "    # Create symbolic links to the data\n",
                "    !ln -s \"{drive_metadata_path}\" data/filename_label.csv\n",
                "    !ln -s \"{drive_images_path}\" data/images\n",
                "    \n",
                "    return Path('data/filename_label.csv'), Path('data/images')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Choose your data source\n",
                "data_source = input(\"Choose your data source (1 for upload, 2 for Google Drive): \")\n",
                "\n",
                "if data_source == \"1\":\n",
                "    metadata_path, image_dir = upload_sample_data()\n",
                "elif data_source == \"2\":\n",
                "    metadata_path, image_dir = mount_drive_data()\n",
                "else:\n",
                "    print(\"Invalid choice. Using upload option by default.\")\n",
                "    metadata_path, image_dir = upload_sample_data()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create a .env file with the paths\n",
                "with open('.env', 'w') as f:\n",
                "    f.write(f\"DATA_DIR=data\\n\")\n",
                "    f.write(f\"PROCESSED_DATA_DIR=data/processed\\n\")\n",
                "    f.write(f\"IMAGE_DIR={image_dir}\\n\")\n",
                "    f.write(f\"METADATA_FILE={metadata_path}\\n\")\n",
                "    f.write(f\"RESULTS_DIR=results\\n\")\n",
                "\n",
                "# Load environment variables\n",
                "load_dotenv()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Preprocessing <a name=\"data-preprocessing\"></a>\n",
                "\n",
                "Now, let's implement the enhanced data preprocessing pipeline using the ChestXRayPreprocessor class. This pipeline includes advanced augmentations such as vertical flips, affine transformations, CLAHE, coarse dropout, and Gaussian noise to improve model generalization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import the preprocessor\n",
                "from src.data.preprocess import ChestXRayPreprocessor\n",
                "\n",
                "# Initialize the preprocessor\n",
                "preprocessor = ChestXRayPreprocessor()\n",
                "\n",
                "# Preprocess the dataset\n",
                "print(\"Starting dataset preprocessing...\")\n",
                "processed_metadata = preprocessor.preprocess_dataset(metadata_path, image_dir)\n",
                "print(f\"Preprocessing complete. Processed {len(processed_metadata)} images.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create TensorFlow datasets\n",
                "from src.training.train import create_dataset\n",
                "\n",
                "# Path to the processed metadata\n",
                "processed_metadata_path = os.path.join(os.getenv('PROCESSED_DATA_DIR', 'data/processed'), 'processed_metadata.csv')\n",
                "\n",
                "# Create datasets\n",
                "print(\"Creating datasets...\")\n",
                "train_dataset = create_dataset(\n",
                "    processed_metadata_path,\n",
                "    batch_size=32,\n",
                "    split='train'\n",
                ")\n",
                "\n",
                "val_dataset = create_dataset(\n",
                "    processed_metadata_path,\n",
                "    batch_size=32,\n",
                "    split='val'\n",
                ")\n",
                "\n",
                "test_dataset = create_dataset(\n",
                "    processed_metadata_path,\n",
                "    batch_size=32,\n",
                "    split='test'\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Model Building <a name=\"model-building\"></a>\n",
                "\n",
                "Now, let's build our enhanced custom CNN model for cardiac failure detection. This model features increased capacity with double convolutional layers in each block, increased filter counts, and larger dense layers for better feature extraction."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import the model\n",
                "from src.models.cardiac_model import CardiacFailureModel\n",
                "\n",
                "# Initialize the model\n",
                "print(\"Initializing model...\")\n",
                "model = CardiacFailureModel()\n",
                "\n",
                "# Print model summary\n",
                "model.model.summary()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Alternatively, you can use a pre-trained model\n",
                "use_pretrained = input(\"Do you want to use a pre-trained model? (y/n): \")\n",
                "\n",
                "if use_pretrained.lower() == 'y':\n",
                "    # Choose a pre-trained model\n",
                "    print(\"Available pre-trained models:\")\n",
                "    print(\"1. EfficientNetB0\")\n",
                "    print(\"2. ResNet50\")\n",
                "    print(\"3. DenseNet121\")\n",
                "    print(\"4. MobileNetV2\")\n",
                "    \n",
                "    model_choice = input(\"Choose a model (1-4): \")\n",
                "    \n",
                "    if model_choice == \"1\":\n",
                "        base_model_name = 'efficientnetb0'\n",
                "    elif model_choice == \"2\":\n",
                "        base_model_name = 'resnet50'\n",
                "    elif model_choice == \"3\":\n",
                "        base_model_name = 'densenet121'\n",
                "    elif model_choice == \"4\":\n",
                "        base_model_name = 'mobilenetv2'\n",
                "    else:\n",
                "        print(\"Invalid choice. Using EfficientNetB0 by default.\")\n",
                "        base_model_name = 'efficientnetb0'\n",
                "    \n",
                "    # Create the pre-trained model\n",
                "    model = CardiacFailureModel()\n",
                "    model.create_pretrained_model(base_model_name=base_model_name, fine_tune_layers=10)\n",
                "    \n",
                "    # Print model summary\n",
                "    model.model.summary()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Training <a name=\"training\"></a>\n",
                "\n",
                "Now, let's train our model with the preprocessed data using advanced training techniques. The training process includes class imbalance handling, cosine decay learning rate scheduling, enhanced callbacks for model checkpointing, and comprehensive TensorBoard logging."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set up TensorBoard\n",
                "# Load the TensorBoard extension\n",
                "%load_ext tensorboard\n",
                "\n",
                "# Create TensorBoard log directory with timestamp\n",
                "current_time = datetime.now().strftime(\"%Y%m%d-%H%M%S\")\n",
                "log_dir = os.path.join(os.getenv('RESULTS_DIR', 'results'), 'logs', current_time)\n",
                "os.makedirs(log_dir, exist_ok=True)\n",
                "\n",
                "# Create TensorBoard callback with enhanced configuration\n",
                "tensorboard_callback = tf.keras.callbacks.TensorBoard(\n",
                "    log_dir=log_dir,\n",
                "    histogram_freq=1,  # Log weight histograms every epoch\n",
                "    write_graph=True,  # Log model graph\n",
                "    write_images=True,  # Log weight images\n",
                "    update_freq='epoch',  # Log metrics every epoch\n",
                "    profile_batch='500,520'  # Profile a few batches\n",
                ")\n",
                "\n",
                "# Launch TensorBoard\n",
                "%tensorboard --logdir={log_dir}"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set up callbacks\n",
                "results_dir = Path(os.getenv('RESULTS_DIR', 'results'))\n",
                "results_dir.mkdir(exist_ok=True)\n",
                "\n",
                "callbacks = [\n",
                "    tf.keras.callbacks.ModelCheckpoint(\n",
                "        filepath=str(results_dir / 'best_model'),\n",
                "        save_best_only=True,\n",
                "        monitor='val_auc',\n",
                "        mode='max',\n",
                "        save_format='tf'\n",
                "    ),\n",
                "    tf.keras.callbacks.EarlyStopping(\n",
                "        monitor='val_loss',\n",
                "        patience=5,\n",
                "        restore_best_weights=True\n",
                "    ),\n",
                "    tensorboard_callback,\n",
                "    tf.keras.callbacks.ReduceLROnPlateau(\n",
                "        monitor='val_loss',\n",
                "        factor=0.2,\n",
                "        patience=3,\n",
                "        min_lr=1e-6\n",
                "    )\n",
                "]"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train the model\n",
                "print(\"Starting training...\")\n",
                "\n",
                "if use_pretrained.lower() == 'y':\n",
                "    # Fine-tune the pre-trained model\n",
                "    history1, history2 = model.fine_tune(\n",
                "        train_dataset,\n",
                "        val_dataset,\n",
                "        epochs=20,  # Initial training epochs\n",
                "        callbacks=callbacks,\n",
                "        fine_tune_epochs=10  # Fine-tuning epochs\n",
                "    )\n",
                "    # Combine histories for evaluation\n",
                "    history = history2\n",
                "    history.history.update({k: history1.history[k] + history2.history[k] for k in history1.history.keys()})\n",
                "else:\n",
                "    # Train the custom model\n",
                "    history = model.train(\n",
                "        train_dataset,\n",
                "        val_dataset,\n",
                "        epochs=50,\n",
                "        callbacks=callbacks\n",
                "    )\n",
                "\n",
                "# Save the final model\n",
                "model.save(results_dir / 'final_model')\n",
                "print(\"Training complete!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Evaluation <a name=\"evaluation\"></a>\n",
                "\n",
                "Now, let's evaluate our trained model on the test dataset using a comprehensive evaluation pipeline. This includes multiple performance metrics (accuracy, precision, recall, F1 score, ROC AUC, PR AUC) and detailed visualizations to thoroughly assess model performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import the evaluation module\n",
                "from src.utils.evaluation import evaluate_model\n",
                "\n",
                "# Evaluate the model\n",
                "print(\"Evaluating model...\")\n",
                "metrics_summary = evaluate_model(\n",
                "    model.model,\n",
                "    test_dataset,\n",
                "    history=history,\n",
                "    results_dir=results_dir\n",
                ")\n",
                "\n",
                "# Print summary metrics\n",
                "print(\"\\nModel Performance Summary:\")\n",
                "for metric, value in metrics_summary.items():\n",
                "    print(f\"{metric}: {value:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Visualization <a name=\"visualization\"></a>\n",
                "\n",
                "Let's visualize the evaluation results."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display the evaluation plots\n",
                "def display_evaluation_plots():\n",
                "    plot_files = [\n",
                "        'confusion_matrix.png',\n",
                "        'roc_curve.png',\n",
                "        'precision_recall_curve.png',\n",
                "        'prediction_distribution.png',\n",
                "        'training_loss.png',\n",
                "        'training_accuracy.png',\n",
                "        'training_auc.png'\n",
                "    ]\n",
                "    \n",
                "    for plot_file in plot_files:\n",
                "        plot_path = results_dir / plot_file\n",
                "        if plot_path.exists():\n",
                "            plt.figure(figsize=(10, 8))\n",
                "            img = plt.imread(plot_path)\n",
                "            plt.imshow(img)\n",
                "            plt.axis('off')\n",
                "            plt.title(plot_file.replace('.png', '').replace('_', ' ').title())\n",
                "            plt.show()\n",
                "        else:\n",
                "            print(f\"Plot file not found: {plot_path}\")\n",
                "\n",
                "# Display the plots\n",
                "display_evaluation_plots()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Inference <a name=\"inference\"></a>\n",
                "\n",
                "Finally, let's implement a function to make predictions on new X-ray images."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Function to preprocess a single image for prediction\n",
                "def preprocess_image_for_prediction(image_path):\n",
                "    import cv2\n",
                "    import albumentations as A\n",
                "    \n",
                "    # Define the transformation\n",
                "    transform = A.Compose([\n",
                "        A.Resize(224, 224),\n",
                "        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n",
                "    ])\n",
                "    \n",
                "    # Read and preprocess the image\n",
                "    image = cv2.imread(image_path)\n",
                "    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
                "    transformed = transform(image=image)\n",
                "    processed_image = transformed['image']\n",
                "    \n",
                "    # Add batch dimension\n",
                "    return np.expand_dims(processed_image, axis=0)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Function to make predictions on new images\n",
                "def predict_cardiac_failure(model, image_path):\n",
                "    # Preprocess the image\n",
                "    processed_image = preprocess_image_for_prediction(image_path)\n",
                "    \n",
                "    # Make prediction\n",
                "    prediction = model.predict(processed_image)[0][0]\n",
                "    \n",
                "    # Display the image and prediction\n",
                "    plt.figure(figsize=(8, 8))\n",
                "    img = plt.imread(image_path)\n",
                "    plt.imshow(img, cmap='gray')\n",
                "    plt.title(f\"Prediction: {'Cardiac Failure' if prediction > 0.5 else 'Normal'} ({prediction:.4f})\")\n",
                "    plt.axis('off')\n",
                "    plt.show()\n",
                "    \n",
                "    return prediction"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Upload a new image for prediction\n",
                "def predict_on_uploaded_image():\n",
                "    print(\"Please upload an X-ray image for prediction:\")\n",
                "    uploaded = files.upload()\n",
                "    \n",
                "    for filename in uploaded.keys():\n",
                "        print(f\"\\nPredicting on {filename}...\")\n",
                "        prediction = predict_cardiac_failure(model.model, filename)\n",
                "        print(f\"Prediction probability: {prediction:.4f}\")\n",
                "        print(f\"Diagnosis: {'Cardiac Failure' if prediction > 0.5 else 'Normal'}\")\n",
                "\n",
                "# Make predictions on uploaded images\n",
                "predict_on_uploaded_image()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Conclusion\n",
                "\n",
                "In this notebook, we've implemented a complete pipeline for cardiac failure detection using chest X-ray images:\n",
                "\n",
                "1. Set up the environment and data\n",
                "2. Preprocessed the X-ray images with enhanced augmentation techniques (vertical flips, affine transformations, CLAHE, coarse dropout, and Gaussian noise)\n",
                "3. Built and trained an enhanced custom CNN model with increased capacity (double convolutional layers, increased filters, larger dense layers)\n",
                "4. Implemented advanced training techniques (class imbalance handling, cosine decay learning rate scheduling, enhanced callbacks)\n",
                "5. Evaluated the model using comprehensive metrics (accuracy, precision, recall, F1 score, ROC AUC, PR AUC)\n",
                "6. Visualized the results with detailed plots\n",
                "7. Implemented inference on new images\n",
                "\n",
                "This enhanced model can be used as a tool to assist healthcare professionals in the early detection of cardiac failure, potentially improving patient outcomes through timely intervention. The advanced techniques implemented in this notebook help achieve better performance and generalization compared to simpler approaches."
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to a file
with open('cardiac_failure_detection.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook created successfully!")
