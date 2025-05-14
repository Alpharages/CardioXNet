import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    EfficientNetB0,
    ResNet50,
    DenseNet121,
    MobileNetV2
)

class CardiacFailureModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """Build a custom CNN model architecture with enhanced capacity."""
        model = models.Sequential([
            # Input Layer
            layers.Input(shape=self.input_shape),

            # First Convolutional Block - increased filters
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Convolutional Block - increased filters
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third Convolutional Block - increased filters
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fourth Convolutional Block - increased filters
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Flatten and Dense Layers - increased capacity
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Output Layer
            layers.Dense(self.num_classes, activation='sigmoid')
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        return model

    def train(self, train_dataset, val_dataset, epochs=10, callbacks=None):
        """Train the model."""
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        return history

    def evaluate(self, test_dataset):
        """Evaluate the model."""
        return self.model.evaluate(test_dataset)

    def predict(self, image):
        """Make predictions on new images."""
        return self.model.predict(image)

    def save(self, path):
        """Save the model in SavedModel format."""
        self.model.save(path, save_format='tf')

    def load(self, path):
        """Load a saved model."""
        self.model = models.load_model(path)
        return self.model

    def _get_architecture_specific_layers(self, base_model_name):
        """
        Get architecture-specific layers for different pre-trained models.

        Args:
            base_model_name (str): Name of the pre-trained model

        Returns:
            list: List of layers to be added on top of the base model
        """
        if base_model_name in ['resnet50', 'densenet121']:
            # These models benefit from additional regularization and capacity
            return [
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dense(1024, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='sigmoid')
            ]
        elif base_model_name in ['efficientnetb0', 'mobilenetv2']:
            # Standard architecture for other models
            return [
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='sigmoid')
            ]
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")

    def create_pretrained_model(self, base_model_name='efficientnetb0', fine_tune_layers=0):
        """
        Create a model based on a pre-trained architecture.

        Args:
            base_model_name (str): Name of the pre-trained model to use. Options:
                - 'efficientnetb0': EfficientNetB0 (default)
                - 'resnet50': ResNet50
                - 'densenet121': DenseNet121
                - 'mobilenetv2': MobileNetV2
            fine_tune_layers (int): Number of layers to fine-tune from the end of the base model
        """
        # Load the pre-trained model without the top classification layer
        if base_model_name == 'efficientnetb0':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'densenet121':
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'mobilenetv2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}. "
                           f"Supported models are: efficientnetb0, resnet50, densenet121, mobilenetv2")

        # Freeze all layers initially
        base_model.trainable = False

        # Unfreeze the last 'fine_tune_layers' layers for fine-tuning
        if fine_tune_layers > 0:
            for layer in base_model.layers[-fine_tune_layers:]:
                layer.trainable = True

        # Create the model with architecture-specific adjustments
        model = models.Sequential([
            base_model,
            *self._get_architecture_specific_layers(base_model_name)
        ])

        # Compile model with a lower learning rate for fine-tuning
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.model = model
        return model

    def fine_tune(self, train_dataset, val_dataset, epochs=10, callbacks=None, fine_tune_epochs=5):
        """
        Fine-tune the pre-trained model in two phases:
        1. Train only the top layers
        2. Fine-tune the unfrozen layers

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs (int): Number of epochs for initial training
            callbacks: List of callbacks
            fine_tune_epochs (int): Number of epochs for fine-tuning
        """
        # Phase 1: Train only the top layers
        print("Phase 1: Training top layers...")
        history1 = self.train(train_dataset, val_dataset, epochs=epochs, callbacks=callbacks)

        # Phase 2: Fine-tune the unfrozen layers
        print("Phase 2: Fine-tuning unfrozen layers...")
        # Recompile with a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        history2 = self.train(train_dataset, val_dataset, epochs=fine_tune_epochs, callbacks=callbacks)

        return history1, history2 
