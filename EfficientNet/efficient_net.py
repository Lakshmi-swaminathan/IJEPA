from datetime import datetime
import os

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense

from ijepa_keras.utils.encoder import EfficientNetEncoder
from ijepa_keras.utils.pretrain import JEPA
from ijepa_keras.utils.predictor import EfficientNetPredictor

EPOCHS = 10
BATCH_SIZE = 64
SAVE_MODEL_PATH = f"./training_results_EfficientNetB2_{int(datetime.now().timestamp())}/" 


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Initialize EfficientNet encoder
efficientnet_encoder = EfficientNetEncoder()
context_encoder = efficientnet_encoder.get_context_encoder()
target_encoder = efficientnet_encoder.get_target_encoder()

# Initialize EfficientNet predictor
predictor = EfficientNetPredictor()
predictor_model = predictor.get_predictor()

# I-JEPA Pretraining
jepa = JEPA(context_encoder=context_encoder,
             target_encoder=target_encoder,
             predictor_model=predictor_model,
             optimizer=Adam(),
             loss_fn=MeanSquaredError(),
             model_save_path=SAVE_MODEL_PATH)

jepa.train(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Load saved models from disk
target_encoder = load_model(os.path.join(SAVE_MODEL_PATH, "best_target_encoder.keras"))

# Freeze the target encoder
for layer in target_encoder.layers:
    layer.trainable = False

# Build linear probing model
def build_linear_probe_model(encoder, num_classes):
    # Input for image
    input_layer = Input(shape=(32, 32, 3), name="embedding input")

    # Pass input through the frozen encoder
    x = encoder(input_layer)

    # Add linear classification head
    output_layer = Dense(num_classes, activation="softmax", name="classification_head")(x)

    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the linear probing model
num_classes = 10  # CIFAR-10 has 10 classes
linear_probe_model = build_linear_probe_model(target_encoder, num_classes)

# Compile the model
linear_probe_model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the linear probing model
history = linear_probe_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=64,
    epochs=20
)
