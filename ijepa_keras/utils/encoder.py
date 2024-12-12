from abc import ABC, abstractmethod

from tensorflow.keras.models import clone_model
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input


class BaseEncoder(ABC):
    """
    Abstract base class for defining an encoder with context and target encoders.

    This class initializes the context and target encoders during object creation
    and provides a method to rebuild them as needed.

    Attributes:
        context_encoder (Model): The primary encoder used as the context encoder.
        target_encoder (Model): A clone of the context encoder used as the target encoder.
    """

    def __init__(self):
        """
        Initializes the BaseEncoder by creating the context and target encoders.
        """
        self.context_encoder = None
        self.target_encoder = None
        self.build_encoder()

    @abstractmethod
    def create_encoder(self):
        """
        Abstract method to create the encoder architecture. Subclasses must implement this.

        Returns:
            Model: A Keras model representing the encoder architecture.
        """
        pass

    def build_encoder(self):
        """
        Builds and initializes the context and target encoders.

        - The context encoder is created using the subclass implementation of `create_encoder`.
        - The target encoder is cloned from the context encoder.
        """
        self.context_encoder = self.create_encoder()
        self.target_encoder = clone_model(self.context_encoder)

    def get_context_encoder(self):
        """
        Retrieves the context encoder.

        Returns:
            Model: The context encoder.
        """
        return self.context_encoder

    def get_target_encoder(self):
        """
        Retrieves the target encoder.

        Returns:
            Model: The target encoder.
        """
        return self.target_encoder


class VGG16Encoder(BaseEncoder):
    """
    Concrete implementation of BaseEncoder using the VGG16 architecture.
    """

    def create_encoder(self):
        """
        Creates a VGG16-based encoder without the fully connected layers.

        Returns:
            Model: A Keras model for VGG16 without the top layers, 
                   with global average pooling added.
        """
        base_model = VGG16(include_top=False, input_shape=(32, 32, 3), weights=None)
        for layer in base_model.layers:
            layer.trainable = True

        # Add a global average pooling layer to collapse spatial dimensions
        x = GlobalAveragePooling2D()(base_model.output)

        # Create the final encoder model
        return Model(inputs=base_model.input, outputs=x)

class MobileNetV2Encoder(BaseEncoder):
    """
    Concrete implementation of BaseEncoder using the MobileNetV2 architecture.

    Attributes:
        context_encoder (Model): The primary encoder used as the context encoder.
        target_encoder (Model): A clone of the context encoder used as the target encoder

    Methods:
        create_encoder(): Creates a MobileNetV2-based encoder without the fully connected layers.
        get_context_encoder(): Retrieves the context encoder.
        get_target_encoder(): Retrieves the target encoder.
    """

    def create_encoder(self):
        """
        Creates a MobileNetV2-based encoder without the fully connected layers.

        Returns:
            Model: A Keras model for MobileNetV2 without the top layers, 
                   with global average pooling added.
        """
        base_model = MobileNetV2(include_top=False, input_shape=(32, 32, 3), weights=None)
        for layer in base_model.layers:
            layer.trainable = True

        # Add a global average pooling layer to collapse spatial dimensions
        x = GlobalAveragePooling2D()(base_model.output)

        # Create the final encoder model
        return Model(inputs=base_model.input, outputs=x)
# Alexnet is not included in the Keras. Creating from scratch
class AlexNetEncoder(BaseEncoder):
    """
    Concrete implementation of BaseEncoder using the AlexNet architecture.
    """

    def create_encoder(self):
        """
        Creates an AlexNet-based encoder for feature extraction.

        Returns:
            Model: A Keras model implementing AlexNet architecture with global average pooling.
        """
        # Define the AlexNet architecture
        input_layer = Input(shape=(32, 32, 3))  # Input shape for CIFAR-10

        # First convolution + pooling layers
        x = Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu")(input_layer)
        x = MaxPooling2D(pool_size=2, strides=2)(x)

        # Second convolution + pooling layers
        x = Conv2D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=2, strides=2)(x)

        # Third convolution layers
        x = Conv2D(256, kernel_size=3, strides=1, padding="same", activation="relu")(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=2, strides=2)(x)

        # Global Average Pooling directly on feature maps
        x = GlobalAveragePooling2D()(x)

        # Optionally add Dense layers for feature extraction
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation="relu")(x)

        # Create the AlexNet model
        model = Model(inputs=input_layer, outputs=x)
        return model