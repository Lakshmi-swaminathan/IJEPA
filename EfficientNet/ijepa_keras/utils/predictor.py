from abc import ABC, abstractmethod

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate


class BasePredictor(ABC):
    """
    Abstract base class for defining a predictor model.

    Subclasses can implement the `build_predictor` method to customize the architecture.
    """

    def __init__(self, context_input_shape, coords_input_shape):
        """
        Initializes the BasePredictor with input shapes and builds the predictor model.

        Args:
            context_input_shape (tuple): Shape of the context input.
            coords_input_shape (tuple): Shape of the coordinates input.
        """
        self.context_input_shape = context_input_shape
        self.coords_input_shape = coords_input_shape
        self.model = None
        self.build_predictor()

    @abstractmethod
    def define_architecture(self, inputs):
        """
        Abstract method to define the architecture of the predictor model.

        Args:
            inputs (tuple): A tuple containing:
                - context_input (Input): The context input layer.
                - coords_input (Input): The coordinates input layer.

        Returns:
            output (Tensor): The output tensor of the model.
        """
        pass

    def build_predictor(self):
        """
        Builds the predictor model using the defined architecture.
        """
        # Define input layers
        context_input = Input(shape=self.context_input_shape, name='context_input')
        coords_input = Input(shape=self.coords_input_shape, name='coords_input')

        # Get the architecture defined by the subclass
        output = self.define_architecture((context_input, coords_input))

        # Create the model
        self.model = Model(
            inputs=[context_input, coords_input],
            outputs=output, name='predictor_model'
        )

    def get_predictor(self):
        """
        Retrieves the built predictor model.

        Returns:
            Model: The Keras predictor model.
        """
        return self.model


class VGG16Predictor(BasePredictor):
    """
    Concrete implementation of BasePredictor for use with VGG16 encoders.

    This class defines the predictor architecture specific to VGG16-style encodings.

    Attributes:
        context_input_shape (tuple): The shape of the context input.
        coords_input_shape (tuple): The shape of the coordinates input.
        model (Model): The Keras model for the predictor.
    
    Methods:
        define_architecture(inputs): Defines the architecture for the VGG16 predictor model.
        build_predictor(): Builds the predictor model using the defined architecture.
        get_predictor(): Retrieves the built predictor model.
    """

    def __init__(self):
        """
        Initializes the VGG16Predictor with the input shapes for context and coordinates.
        """
        super().__init__((512,), (4,))

    def define_architecture(self, inputs):
        """
        Defines the architecture for the VGG16 predictor model.

        Args:
            inputs (tuple): A tuple containing:
                - context_input (Input): The context input layer.
                - coords_input (Input): The coordinates input layer.

        Returns:
            output (Tensor): The output tensor of the model.
        """
        context_input, coords_input = inputs

        # Concatenate the inputs
        x = Concatenate(name='concat_features')([context_input, coords_input])

        # Dense layers to process the combined input
        x = Dense(1024, activation='relu', name='dense1')(x)

        # Output layer
        output = Dense(512, activation='relu', name='output_vector')(x)
        return output

class AlexNetPredictor(BasePredictor):
    """
    Concrete implementation of BasePredictor for use with AlexNet encoders.

    This class defines the predictor architecture specific to AlexNet-style encodings.

    Attributes:
        context_input_shape (tuple): The shape of the context input.
        coords_input_shape (tuple): The shape of the coordinates input.
        model (Model): The Keras model for the predictor.
    
    Methods:
        define_architecture(inputs): Defines the architecture for the AlexNet predictor model.
        build_predictor(): Builds the predictor model using the defined architecture.
        get_predictor(): Retrieves the built predictor model.
    """

    def __init__(self):
        """
        Initializes the AlexNetPredictor with the input shapes for context and coordinates.
        """
        super().__init__((256,), (4,))  # AlexNet typically has a 256-dimensional feature vector

    def define_architecture(self, inputs):
        """
        Defines the architecture for the AlexNet predictor model.

        Args:
            inputs (tuple): A tuple containing:
                - context_input (Input): The context input layer.
                - coords_input (Input): The coordinates input layer.

        Returns:
            output (Tensor): The output tensor of the model.
        """
        context_input, coords_input = inputs

        # Concatenate the inputs
        x = Concatenate(name='concat_features')([context_input, coords_input])

        # Dense layers to process the combined input
        x = Dense(1024, activation='relu', name='dense1')(x)
        x = Dense(512, activation='relu', name='dense2')(x)

        # Output layer
        output = Dense(256, activation='relu', name='output_vector')(x)
        return output

class EfficientNetPredictor(BasePredictor):
    """
    Concrete implementation of BasePredictor for use with EfficientNet encoders.

    This class defines the predictor architecture specific to EfficientNet-style encodings.

    Attributes:
        context_input_shape (tuple): The shape of the context input.
        coords_input_shape (tuple): The shape of the coordinates input.
        model (Model): The Keras model for the predictor.
    """

    def __init__(self):
        """
        Initializes the EfficientNetPredictor with the input shapes for context and coordinates.
        """
        super().__init__((1408,), (4,))  # EfficientNetB2 outputs 1408-dimensional feature vectors

    def define_architecture(self, inputs):
        """
        Defines the architecture for the EfficientNet predictor model.

        Args:
            inputs (tuple): A tuple containing:
                - context_input (Input): The context input layer.
                - coords_input (Input): The coordinates input layer.

        Returns:
            output (Tensor): The output tensor of the model.
        """
        context_input, coords_input = inputs

        # Concatenate the inputs
        x = Concatenate(name='concat_features')([context_input, coords_input])

        # Dense layers to process the combined input
        x = Dense(2048, activation='relu', name='dense1')(x)
        x = Dense(1024, activation='relu', name='dense2')(x)

        # Output layer
        output = Dense(1408, activation='relu', name='output_vector')(x)
        return output
