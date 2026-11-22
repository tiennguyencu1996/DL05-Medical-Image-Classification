"""Preflight/smoke test for DL-05 CNN From Scratch V2.

Safety properties:
- Does not read the Validation or Test split.
- Does not modify V1 files or experiment directories.
- Uses only eight balanced examples from the Train split.
- Enables disable_meta_optimizer before model construction/tracing.
- Checks eager/graph forward outputs and gradients, one train step, and
  save/reload reproducibility before a full experiment is created.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.data.input_pipeline import (
    BATCH_SIZE,
    load_split_metadata,
    build_dataset,
    normalize_for_scratch,
)
from src.models.cnn_from_scratch_v2 import build_cnn_from_scratch_v2

RANDOM_SEED = 42
EXPECTED_BATCH_SIZE = 8
FORWARD_ATOL = 1e-6
GRADIENT_ATOL = 1e-4
GRADIENT_RTOL = 1e-4


def configure_tensorflow_runtime() -> dict[str, Any]:
    """Configure runtime before model construction or graph tracing."""
    options = tf.config.optimizer.get_experimental_options()
    options["disable_meta_optimizer"] = True
    tf.config.optimizer.set_experimental_options(options)

    # Set conservative thread counts before the runtime is initialized.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        threading_status = "configured: intra_op=1, inter_op=2"
    except RuntimeError as exc:
        threading_status = f"unchanged (runtime already initialized): {exc}"

    actual_options = tf.config.optimizer.get_experimental_options()
    if actual_options.get("disable_meta_optimizer") is not True:
        raise RuntimeError("Failed to disable TensorFlow Meta Optimizer.")

    return {
        "optimizer_options": actual_options,
        "threading": threading_status,
    }


def prepare_scratch_batch(images: tf.Tensor, labels: tf.Tensor):
    images, labels = normalize_for_scratch(images, labels)
    labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)
    return images, labels


def select_balanced_train_examples():
    """Select four NORMAL and four PNEUMONIA images from Train metadata."""
    paths, labels, records = load_split_metadata("train")
    normal_indices = [i for i, label in enumerate(labels) if label == 0]
    pneumonia_indices = [i for i, label in enumerate(labels) if label == 1]

    if len(normal_indices) < 4 or len(pneumonia_indices) < 4:
        raise RuntimeError("Train split does not contain enough examples per class.")

    selected_indices = normal_indices[:4] + pneumonia_indices[:4]
    selected_paths = [paths[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]
    selected_records = [records[i] for i in selected_indices]

    if sum(selected_labels) != 4:
        raise AssertionError("Smoke-test batch is not balanced 4/4.")

    return selected_paths, selected_labels, selected_records


def max_abs_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def check_finite_tensors(name: str, tensors) -> None:
    for index, tensor in enumerate(tensors):
        if tensor is None:
            raise RuntimeError(f"{name}[{index}] is None.")
        tf.debugging.assert_all_finite(
            tensor,
            message=f"Non-finite values in {name}[{index}]",
        )


def main() -> None:
    print("=" * 75)
    print("DL-05 CNN V2 CONTROLLED-RUNTIME PREFLIGHT")
    print("=" * 75)

    runtime = configure_tensorflow_runtime()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("Python/TensorFlow:", tf.__version__)
    print("Keras:", tf.keras.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("Runtime threading:", runtime["threading"])
    print("Optimizer options:", json.dumps(runtime["optimizer_options"], default=str))
    print("Meta Optimizer disabled:", runtime["optimizer_options"].get("disable_meta_optimizer"))

    # Load only Train metadata. Validation and Test are not accessed here.
    paths, labels, records = select_balanced_train_examples()
    print("\nSelected Train examples:", len(paths))
    print("NORMAL:", labels.count(0), "PNEUMONIA:", labels.count(1))
    print("Selected patient IDs:", len({r["patient_id"] for r in records}))

    raw_dataset = build_dataset(paths, labels, training=False)
    batch_images, batch_labels = next(iter(raw_dataset.take(1)))
    images, y = prepare_scratch_batch(batch_images, batch_labels)

    if int(images.shape[0]) != EXPECTED_BATCH_SIZE:
        raise RuntimeError(f"Expected batch size 8, got {images.shape[0]}.")
    if tuple(images.shape[1:]) != (224, 224, 3):
        raise RuntimeError(f"Unexpected image shape: {images.shape}")
    if tuple(y.shape) != (EXPECTED_BATCH_SIZE, 1):
        raise RuntimeError(f"Unexpected label shape: {y.shape}")

    print("Image batch shape:", tuple(images.shape))
    print("Label batch shape:", tuple(y.shape))
    print("Image dtype:", images.dtype.name)
    print("Pixel range:", float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))

    model = build_cnn_from_scratch_v2()
    print("\nModel:", model.name)
    print("Input:", model.input_shape)
    print("Output:", model.output_shape)
    print("Total parameters:", f"{model.count_params():,}")
    print("Trainable parameters:", f"{sum(int(tf.keras.backend.count_params(w)) for w in model.trainable_weights):,}")
    print("Non-trainable parameters:", f"{sum(int(tf.keras.backend.count_params(w)) for w in model.non_trainable_weights):,}")

    # 1) Compare eager and graph forward inference with identical weights/input.
    eager_predictions = model(images, training=False)

    @tf.function(reduce_retracing=True)
    def graph_forward(x):
        return model(x, training=False)

    graph_predictions = graph_forward(images)
    eager_np = eager_predictions.numpy()
    graph_np = graph_predictions.numpy()
    forward_diff = max_abs_difference(eager_np, graph_np)
    forward_match = bool(np.allclose(eager_np, graph_np, atol=FORWARD_ATOL, rtol=0))

    print("\n[1] EAGER VS GRAPH FORWARD")
    print("Maximum absolute difference:", forward_diff)
    print("Allclose:", forward_match)
    if not forward_match:
        raise RuntimeError("CNN V2 eager/graph forward outputs do not match.")

    # 2) Compare finite eager and graph gradients on the same deterministic
    #    inference-mode path (Dropout disabled; BN uses moving statistics).
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    variables = model.trainable_variables

    with tf.GradientTape() as tape:
        eager_pred_for_grad = model(images, training=False)
        eager_loss = loss_fn(y, eager_pred_for_grad)
    eager_grads = tape.gradient(eager_loss, variables)
    check_finite_tensors("Eager gradients", eager_grads)

    @tf.function(reduce_retracing=True)
    def graph_loss_and_gradients(x, labels_tensor):
        with tf.GradientTape() as graph_tape:
            predictions = model(x, training=False)
            loss = loss_fn(labels_tensor, predictions)
        gradients = graph_tape.gradient(loss, variables)
        return loss, gradients

    graph_loss, graph_grads = graph_loss_and_gradients(images, y)
    check_finite_tensors("Graph gradients", graph_grads)
    loss_diff = abs(float(eager_loss.numpy()) - float(graph_loss.numpy()))
    gradient_diffs = [
        max_abs_difference(a.numpy(), b.numpy())
        for a, b in zip(eager_grads, graph_grads)
    ]
    max_gradient_diff = max(gradient_diffs, default=0.0)
    gradients_match = all(
        np.allclose(
            a.numpy(), b.numpy(),
            atol=GRADIENT_ATOL,
            rtol=GRADIENT_RTOL,
        )
        for a, b in zip(eager_grads, graph_grads)
    )

    print("\n[2] EAGER VS GRAPH GRADIENTS")
    print("Eager loss:", float(eager_loss.numpy()))
    print("Graph loss:", float(graph_loss.numpy()))
    print("Absolute loss difference:", loss_diff)
    print("Maximum gradient absolute difference:", max_gradient_diff)
    print("Gradients allclose:", gradients_match)
    if not gradients_match:
        raise RuntimeError("CNN V2 eager/graph gradients differ beyond tolerance.")

    # 3) Compile and run exactly one graph-mode training batch. This only
    #    updates the temporary in-memory smoke-test model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="auroc", curve="ROC", num_thresholds=1000),
        ],
        run_eagerly=False,
    )
    train_metrics = model.train_on_batch(images, y, return_dict=True)
    for name, value in train_metrics.items():
        if not np.isfinite(float(value)):
            raise RuntimeError(f"Non-finite train_on_batch metric: {name}={value}")

    for variable in model.weights:
        tf.debugging.assert_all_finite(
            variable,
            message=f"Non-finite model weight after smoke training: {variable.name}",
        )

    print("\n[3] ONE GRAPH-MODE TRAINING BATCH")
    for name, value in train_metrics.items():
        print(f"{name}: {float(value):.8f}")
    print("All model weights finite: PASS")

    # 4) Save/reload round-trip in a temporary OS directory only.
    with tempfile.TemporaryDirectory(prefix="dl05_cnn_v2_smoke_") as temp_dir:
        checkpoint_path = Path(temp_dir) / "cnn_v2_smoke.keras"
        model.save(checkpoint_path)
        reloaded = tf.keras.models.load_model(checkpoint_path, compile=False)
        original_after_training = model(images, training=False).numpy()
        reloaded_predictions = reloaded(images, training=False).numpy()
        serialization_diff = max_abs_difference(
            original_after_training,
            reloaded_predictions,
        )
        serialization_match = bool(
            np.allclose(
                original_after_training,
                reloaded_predictions,
                atol=FORWARD_ATOL,
                rtol=0,
            )
        )

    print("\n[4] CHECKPOINT SAVE/RELOAD")
    print("Maximum prediction difference:", serialization_diff)
    print("Round-trip allclose:", serialization_match)
    if not serialization_match:
        raise RuntimeError("CNN V2 save/reload predictions do not match.")

    print("\n" + "=" * 75)
    print("PASS: CNN V2 PREFLIGHT COMPLETED")
    print("- Only eight Train images were used.")
    print("- Validation split was not loaded.")
    print("- Test split was not loaded.")
    print("- No experiment directory or project checkpoint was created/modified.")
    print("- Full training has NOT started.")
    print("=" * 75)


if __name__ == "__main__":
    main()
