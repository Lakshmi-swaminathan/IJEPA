import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from ijepa_keras.utils.data import ImageBlockGenerator


class JEPA:
    """
    Joint Encoder-Pretraining Algorithm (JEPA) for pretraining a predictor model.

    Attributs:
        context_encoder (Model): The context encoder model.
        target_encoder (Model): The target encoder model.
        predictor_model (Model): The predictor model.
        optimizer (Optimizer): The optimizer to use for training.
        loss_fn (function): The loss function to use for training.
        save_best_model (bool): Whether to save the best model during training.
        model_save_path (str): The path to save the trained models.
        block_generator (ImageBlockGenerator): The image block generator.

    Methods:
        train(x_train, epochs=10, batch_size=32, initial_momentum=0.96, final_momentum=1.0):
            Trains the predictor model using the JEPA algorithm.
    """
    def __init__(self,
                 context_encoder,
                 target_encoder,
                 predictor_model,
                 optimizer,
                 loss_fn,
                 save_best_model=True,
                 model_save_path="./best_model/"):
        """
        Initializes the JEPA model with encoders, predictor, optimizer, and loss function.

        Args:
            context_encoder (Model): The context encoder model.
            target_encoder (Model): The target encoder model.
            predictor_model (Model): The predictor model.
            optimizer (Optimizer): The optimizer to use for training.
            loss_fn (function): The loss function to use for training.
        """
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor_model = predictor_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_best_model = save_best_model
        self.model_save_path = model_save_path
        self.block_generator = ImageBlockGenerator()

        # Ensure the save directory exists
        os.makedirs(self.model_save_path, exist_ok=True)

    def train(self, x_train, epochs=10, batch_size=32, initial_momentum=0.96, final_momentum=1.0):
        """
        Trains the predictor model using the JEPA algorithm.

        Args:
            x_train (ndarray): The training images.
            epochs (int): The number of epochs to train for.
            batch_size (int): The batch size for training.
            initial_momentum (float): The initial EMA momentum value.
            final_momentum (float): The final EMA momentum value.
        """
        # Initialize best loss
        best_loss = float('inf')
        epoch_losses = []  # Store epoch losses for plotting

        # Open log file
        log_file_path = os.path.join(self.model_save_path, "training_log.txt")
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            # Add a header to the log file
            log_file.write("Epoch\tLoss\tMomentum\n")

            # Training Loop
            for epoch in range(epochs):
                total_loss = 0  # Track cumulative loss for the epoch
                # Total number of batches
                num_batches = len(x_train) // batch_size
                current_momentum = initial_momentum + \
                    (epoch / epochs) * (final_momentum - initial_momentum)

                with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                    for step in range(num_batches):
                        # Create a batch of images
                        batch_images = x_train[step *
                                               batch_size:(step + 1) * batch_size]

                        # Generate context and target patches
                        batch_contexts, batch_targets, batch_coords = [], [], []
                        for img in batch_images:
                            blocks = self.block_generator.generate_blocks(img)
                            batch_contexts.append(blocks['context'])
                            for target in blocks['targets']:
                                batch_targets.append(target['patch'])
                                batch_coords.append(
                                    list(target['coords'].values()))

                        # Convert to numpy arrays and preprocess
                        batch_contexts = np.array(batch_contexts) / 255.0
                        batch_targets = np.array(batch_targets) / 255.0
                        batch_coords = np.array(
                            batch_coords, dtype=np.float32)

                        # Encode context and target patches (silent predictions)
                        batch_context_encodings = self.context_encoder.predict(
                            batch_contexts, verbose=0)
                        batch_target_encodings = self.target_encoder.predict(
                            batch_targets, verbose=0)

                        # Repeat context encodings to match target patches
                        batch_context_encodings = np.repeat(
                            batch_context_encodings, len(batch_coords) // batch_size, axis=0)

                        # Training step
                        with tf.GradientTape() as tape:
                            # Predict using predictor model
                            predictions = self.predictor_model(
                                [batch_context_encodings, batch_coords])
                            # Calculate loss
                            loss = self.loss_fn(
                                batch_target_encodings, predictions)

                        trainable_params = self.predictor_model.trainable_variables + \
                            self.context_encoder.trainable_variables

                        # Compute gradients
                        gradients = tape.gradient(loss, trainable_params)
                        # Apply gradients
                        self.optimizer.apply_gradients(
                            zip(gradients, trainable_params))

                        # Update loss and progress bar
                        total_loss += loss.numpy()
                        pbar.update(1)  # Increment progress bar

                        # Update target encoder weights at the end of the epoch
                        for var, target_var in zip(self.context_encoder.variables, self.target_encoder.variables):
                            target_var.assign(
                                current_momentum * target_var + (1 - current_momentum) * var)

                # Calculate epoch loss
                epoch_loss = total_loss / num_batches
                epoch_losses.append(epoch_loss)  # Store epoch loss

                # Write epoch and loss to log file
                log_file.write(
                    f"{epoch + 1}\t{epoch_loss:.4f}\t{current_momentum:.4f}\n")
                log_file.flush()  # Ensure data is written immediately

                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Momentum: {current_momentum:.4f}")

                # Save best model
                if self.save_best_model and epoch_loss < best_loss:
                    print(f"Saving best models with loss:{epoch_loss}...")
                    best_loss = epoch_loss
                    self.predictor_model.save(
                        self.model_save_path + "best_predictor.keras")
                    self.context_encoder.save(
                        self.model_save_path + "best_context_encoder.keras")
                    self.target_encoder.save(
                        self.model_save_path + "best_target_encoder.keras")

        # Plot and save epoch loss graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), epoch_losses, marker='o', label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("IJEPA Pretraining Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.model_save_path, "epoch_loss_plot.png"))
        plt.close()
