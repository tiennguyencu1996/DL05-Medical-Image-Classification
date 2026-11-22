"""Training-mode gradient preflight for DL-05 CNN V2.

Uses eight balanced examples from Train only. It verifies finite, non-None
training-mode gradients with Dropout active and BatchNorm in training mode in
both eager and graph execution. It does not compare gradient values between
those two runs because Dropout is stochastic. It also checks that BatchNorm
moving statistics update and that one graph-mode optimizer step is finite.

No Validation/Test data, project checkpoint, or experiment directory is used.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import tensorflow as tf

from src.data.input_pipeline import (
    build_dataset,
    load_split_metadata,
    normalize_for_scratch,
)
from src.models.cnn_from_scratch_v2 import build_cnn_from_scratch_v2

SEED = 42


def configure_runtime() -> dict[str, Any]:
    options = tf.config.optimizer.get_experimental_options()
    options["disable_meta_optimizer"] = True
    tf.config.optimizer.set_experimental_options(options)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        threading = "configured intra_op=1, inter_op=2"
    except RuntimeError as exc:
        threading = f"unchanged: {exc}"
    actual = tf.config.optimizer.get_experimental_options()
    if actual.get("disable_meta_optimizer") is not True:
        raise RuntimeError("Could not disable TensorFlow Meta Optimizer")
    return {"optimizer_options": actual, "threading": threading}


def select_balanced_train_batch():
    paths, labels, rows = load_split_metadata("train")
    chosen = []
    for label in (0, 1):
        indices = [i for i, value in enumerate(labels) if value == label]
        if len(indices) < 4:
            raise RuntimeError(f"Not enough Train examples for label {label}")
        chosen.extend(indices[:4])
    chosen.sort(key=lambda i: (labels[i], i))
    return (
        [paths[i] for i in chosen],
        [labels[i] for i in chosen],
        [rows[i] for i in chosen],
    )


def prepare(images, labels):
    images, labels = normalize_for_scratch(images, labels)
    labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)
    return images, labels


def gradient_report(model, images, labels, mode: str):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    variables = model.trainable_variables

    if mode == "eager":
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, variables)
    elif mode == "graph":
        @tf.function(reduce_retracing=True)
        def calculate(x, y):
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss_value = loss_fn(y, predictions)
            grads = tape.gradient(loss_value, variables)
            return loss_value, grads
        loss, gradients = calculate(images, labels)
    else:
        raise ValueError(mode)

    if not np.isfinite(float(loss.numpy())):
        raise RuntimeError(f"{mode}: loss is not finite")
    if len(gradients) != len(variables):
        raise RuntimeError(f"{mode}: gradient count mismatch")

    finite_count = 0
    nonzero_count = 0
    for index, (variable, gradient) in enumerate(zip(variables, gradients)):
        if gradient is None:
            raise RuntimeError(
                f"{mode}: gradient {index} for {variable.name} is None"
            )
        tf.debugging.assert_all_finite(
            gradient,
            message=f"{mode}: non-finite gradient for {variable.name}",
        )
        dense_gradient = tf.convert_to_tensor(gradient)
        finite_count += 1
        if bool(tf.reduce_any(tf.not_equal(dense_gradient, 0)).numpy()):
            nonzero_count += 1

    return {
        "loss": float(loss.numpy()),
        "gradient_tensors": finite_count,
        "nonzero_gradient_tensors": nonzero_count,
    }


def main():
    print("=" * 75)
    print("CNN V2 TRAINING-MODE GRADIENT PREFLIGHT")
    print("=" * 75)

    runtime = configure_runtime()
    tf.keras.utils.set_random_seed(SEED)
    print("TensorFlow:", tf.__version__)
    print("Keras:", tf.keras.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("Threading:", runtime["threading"])
    print("Optimizer options:", json.dumps(runtime["optimizer_options"], default=str))

    paths, labels, rows = select_balanced_train_batch()
    dataset = build_dataset(paths, labels, training=False)
    batch_images, batch_labels = next(iter(dataset.take(1)))
    images, targets = prepare(batch_images, batch_labels)
    print("Train examples:", len(paths))
    print("Class counts:", {"NORMAL": labels.count(0), "PNEUMONIA": labels.count(1)})
    print("Unique patients:", len({row["patient_id"] for row in rows}))

    # Use independent, identically seeded models for the two stochastic
    # training-mode passes; compare finiteness, not exact stochastic values.
    tf.keras.utils.set_random_seed(SEED)
    eager_model = build_cnn_from_scratch_v2()
    bn_layers = [
        layer for layer in eager_model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    bn_before = [layer.moving_mean.numpy().copy() for layer in bn_layers]
    eager_result = gradient_report(eager_model, images, targets, "eager")
    bn_after = [layer.moving_mean.numpy().copy() for layer in bn_layers]
    bn_updated = any(
        not np.array_equal(before, after)
        for before, after in zip(bn_before, bn_after)
    )
    if not bn_updated:
        raise RuntimeError("BatchNormalization moving statistics did not update in training mode")

    tf.keras.utils.set_random_seed(SEED)
    graph_model = build_cnn_from_scratch_v2()
    graph_result = gradient_report(graph_model, images, targets, "graph")

    if eager_result["nonzero_gradient_tensors"] == 0:
        raise RuntimeError("Eager training produced no nonzero gradients")
    if graph_result["nonzero_gradient_tensors"] == 0:
        raise RuntimeError("Graph training produced no nonzero gradients")

    # Verify one actual graph-mode training step on an ephemeral model.
    graph_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        run_eagerly=False,
    )
    step_metrics = graph_model.train_on_batch(
        images, targets, return_dict=True
    )
    for name, value in step_metrics.items():
        if not np.isfinite(float(value)):
            raise RuntimeError(f"Non-finite training step metric {name}: {value}")
    for weight in graph_model.weights:
        tf.debugging.assert_all_finite(
            weight, message=f"Non-finite weight after train_on_batch: {weight.name}"
        )

    print("\nEAGER training=True:", eager_result)
    print("GRAPH training=True:", graph_result)
    print("BatchNorm moving statistics updated:", bn_updated)
    print("Graph train_on_batch:", {k: float(v) for k, v in step_metrics.items()})
    print("All model weights finite: PASS")
    print("\nPASS: training-mode gradient preflight completed")
    print("Only eight Train images were used; Validation/Test were not loaded.")
    print("No project checkpoint or experiment directory was created.")
    print("=" * 75)


if __name__ == "__main__":
    main()
