"""CNN From Scratch V2 model definition for DL-05.

V2 intentionally preserves the V1 architecture and parameter count. The
purpose of V2 is to run a new experiment with a controlled TensorFlow runtime,
not to silently change the model architecture.
"""

import tensorflow as tf

from src.models.cnn_from_scratch import (
    build_cnn_from_scratch,
    calculate_manual_parameters,
)

MODEL_NAME = "CNNFromScratch_V2_Controlled"
EXPECTED_TOTAL_PARAMS = 1_208_161
EXPECTED_TRAINABLE_PARAMS = 1_206_241
EXPECTED_NON_TRAINABLE_PARAMS = 1_920


def build_cnn_from_scratch_v2() -> tf.keras.Model:
    """Build the same verified V1 topology under a distinct V2 model name.

    The original V1 builder is reused so that the architecture is not
    inadvertently changed while investigating the runtime issue. A new Model
    wrapper gives the experiment an explicit V2 identity. The V1 source and
    artifacts are not modified.
    """
    base_model = build_cnn_from_scratch()
    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.output,
        name=MODEL_NAME,
    )

    manual = calculate_manual_parameters()
    expected = {
        "total": EXPECTED_TOTAL_PARAMS,
        "trainable": EXPECTED_TRAINABLE_PARAMS,
        "non_trainable": EXPECTED_NON_TRAINABLE_PARAMS,
    }
    actual = {
        "total": model.count_params(),
        "trainable": sum(
            int(tf.TensorShape(weight.shape).num_elements())
            for weight in model.trainable_weights
        ),
        "non_trainable": sum(
            int(tf.TensorShape(weight.shape).num_elements())
            for weight in model.non_trainable_weights
        ),
    }

    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise AssertionError(
                f"CNN V2 {key} parameter count mismatch: "
                f"expected {expected_value:,}, got {actual[key]:,}"
            )
        if manual[key] != expected_value:
            raise AssertionError(
                f"Manual {key} parameter count mismatch: "
                f"expected {expected_value:,}, got {manual[key]:,}"
            )

    if tuple(model.input_shape) != (None, 224, 224, 3):
        raise AssertionError(f"Unexpected input shape: {model.input_shape}")
    if tuple(model.output_shape) != (None, 1):
        raise AssertionError(f"Unexpected output shape: {model.output_shape}")

    return model
