"""Experiment 5: TensorFlow/Keras Variational Autoencoder for MNIST generation."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def train_vae(epochs: int = 3, batch_size: int = 128, latent_dim: int = 2, output_dir: str | Path = "outputs/exp05") -> dict[str, Any]:
    """Train a compact VAE on MNIST and export plots and Keras models.

    The experiment manual suggests 30 epochs. The default is deliberately shorter
    for an interactive lab demo; set epochs=30 for the full run.
    """
    import numpy as np
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    image_shape = (28, 28, 1)

    @tf.keras.utils.register_keras_serializable(package="GenAILab")
    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            mean, log_var = inputs
            epsilon = tf.random.normal(shape=tf.shape(mean))
            return mean + tf.exp(0.5 * log_var) * epsilon

    encoder_inputs = tf.keras.Input(shape=image_shape, name="encoder_input")
    x = tf.keras.layers.Flatten()(encoder_inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z_sampling")([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling_input")
    x = tf.keras.layers.Dense(128, activation="relu")(latent_inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(28 * 28, activation="sigmoid")(x)
    decoder_outputs = tf.keras.layers.Reshape(image_shape)(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    class VAE(tf.keras.Model):
        def __init__(self, enc, dec, **kwargs):
            super().__init__(**kwargs)
            self.encoder = enc
            self.decoder = dec
            self.total_tracker = tf.keras.metrics.Mean(name="loss")
            self.recon_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
            self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")

        def call(self, inputs, training=False):
            _mean, _log_var, sampled = self.encoder(inputs, training=training)
            return self.decoder(sampled, training=training)

        @property
        def metrics(self):
            return [self.total_tracker, self.recon_tracker, self.kl_tracker]

        def _loss_parts(self, data, training):
            mean, log_var, sampled = self.encoder(data, training=training)
            reconstruction = self.decoder(sampled, training=training)
            # BCE reduces the last (channel) axis; sum pixels to get per-example loss.
            bce = tf.keras.backend.binary_crossentropy(data, reconstruction)
            recon_loss = tf.reduce_mean(tf.reduce_sum(bce, axis=[1, 2]))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1.0 + log_var - tf.square(mean) - tf.exp(log_var), axis=1
            ))
            return recon_loss + kl_loss, recon_loss, kl_loss

        def train_step(self, data):
            if isinstance(data, tuple):
                data = data[0]
            with tf.GradientTape() as tape:
                total, recon, kl = self._loss_parts(data, training=True)
            grads = tape.gradient(total, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_tracker.update_state(total)
            self.recon_tracker.update_state(recon)
            self.kl_tracker.update_state(kl)
            return {m.name: m.result() for m in self.metrics}

        def test_step(self, data):
            if isinstance(data, tuple):
                data = data[0]
            total, recon, kl = self._loss_parts(data, training=False)
            self.total_tracker.update_state(total)
            self.recon_tracker.update_state(recon)
            self.kl_tracker.update_state(kl)
            return {m.name: m.result() for m in self.metrics}

    vae = VAE(encoder, decoder, name="vae")
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    history = vae.fit(
        x_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_data=x_test,
        verbose=2,
    )

    encoder_path = out / "encoder.keras"
    decoder_path = out / "decoder.keras"
    encoder.save(encoder_path)
    decoder.save(decoder_path)

    # Plot training curves.
    loss_plot = out / "vae_training_losses.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in [("loss", "Total loss"), ("reconstruction_loss", "Reconstruction loss"), ("kl_loss", "KL divergence")]:
        if key in history.history:
            ax.plot(history.history[key], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("VAE training losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_plot, dpi=140)
    plt.close(fig)

    # Original vs reconstructed examples.
    n = 10
    mu, logvar, z_values = encoder.predict(x_test[:n], verbose=0)
    reconstructed = decoder.predict(z_values, verbose=0)
    recon_plot = out / "vae_reconstructions.png"
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3))
    for i in range(n):
        axes[0, i].imshow(x_test[i, ..., 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i, ..., 0], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    fig.tight_layout()
    fig.savefig(recon_plot, dpi=140)
    plt.close(fig)

    # Latent-space visualization for 2D latent spaces.
    latent_plot = out / "vae_latent_space.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    if int(latent_dim) == 2:
        means, _, _ = encoder.predict(x_test[:5000], batch_size=256, verbose=0)
        scatter = ax.scatter(means[:, 0], means[:, 1], c=y_test[:len(means)], cmap="tab10", s=4, alpha=0.6)
        fig.colorbar(scatter, ax=ax, ticks=range(10), label="Digit class")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title("2D VAE latent space")
    else:
        ax.text(0.5, 0.5, "Set latent_dim=2 to visualize a 2D latent space", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(latent_plot, dpi=140)
    plt.close(fig)

    # Generate new digits from N(0, I).
    generated = decoder.predict(np.random.normal(size=(25, int(latent_dim))).astype("float32"), verbose=0)
    generated_plot = out / "vae_generated_digits.png"
    fig, axes = plt.subplots(5, 5, figsize=(5, 5))
    for ax, image in zip(axes.ravel(), generated):
        ax.imshow(image[..., 0], cmap="gray")
        ax.axis("off")
    fig.suptitle("New MNIST-like images sampled from the latent space")
    fig.tight_layout()
    fig.savefig(generated_plot, dpi=140)
    plt.close(fig)

    return {
        "history": history.history,
        "encoder_path": str(encoder_path),
        "decoder_path": str(decoder_path),
        "plots": [str(loss_plot), str(recon_plot), str(latent_plot), str(generated_plot)],
        "output_dir": str(out),
    }


if __name__ == "__main__":
    result = train_vae(epochs=3)
    print(result)
