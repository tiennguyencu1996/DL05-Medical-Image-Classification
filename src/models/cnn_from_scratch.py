
from pathlib import Path

import tensorflow as tf


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "CNNFromScratch_V1"

IMAGE_SIZE = 224
INPUT_CHANNELS = 3

RANDOM_SEED = 42

# CNN architecture
FILTERS = [32, 64, 128, 256]

KERNEL_SIZE = 3
POOL_SIZE = 2

DROPOUT_RATES = [0.10, 0.15, 0.20, 0.25]

DENSE_UNITS = 128

# Expected model parameters
EXPECTED_TOTAL_PARAMS = 1_208_161
EXPECTED_TRAINABLE_PARAMS = 1_206_241
EXPECTED_NON_TRAINABLE_PARAMS = 1_920


# ============================================================
# 2. CNN CONVOLUTIONAL BLOCK
# ============================================================

def convolutional_block(
    inputs,
    filters,
    dropout_rate,
    block_number,
):
    """
    Build one convolutional block.

    Architecture:

        Conv2D (3x3, same, no bias)
        -> Batch Normalization
        -> ReLU

        Conv2D (3x3, same, no bias)
        -> Batch Normalization
        -> ReLU

        MaxPooling2D (2x2)
        -> Dropout

    """

    prefix = f"block{block_number}"

    # --------------------------------------------------------
    # First convolution
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=f"{prefix}_conv1",
    )(inputs)

    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        center=True,
        scale=True,
        name=f"{prefix}_bn1",
    )(x)

    x = tf.keras.layers.Activation(
        "relu",
        name=f"{prefix}_relu1",
    )(x)

    # --------------------------------------------------------
    # Second convolution
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=f"{prefix}_conv2",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        center=True,
        scale=True,
        name=f"{prefix}_bn2",
    )(x)

    x = tf.keras.layers.Activation(
        "relu",
        name=f"{prefix}_relu2",
    )(x)

    # --------------------------------------------------------
    # Spatial downsampling
    # --------------------------------------------------------

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE),
        strides=(POOL_SIZE, POOL_SIZE),
        padding="valid",
        name=f"{prefix}_pool",
    )(x)

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    x = tf.keras.layers.Dropout(
        rate=dropout_rate,
        name=f"{prefix}_dropout",
    )(x)

    return x


# ============================================================
# 3. BUILD CNN FROM SCRATCH
# ============================================================

def build_cnn_from_scratch():
    """
    Build the CNN architecture from scratch.

    Input:
        (224, 224, 3)

    Output:
        One sigmoid probability:
        P(PNEUMONIA)
    """

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Input layer
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, INPUT_CHANNELS),
        dtype=tf.float32,
        name="image_input",
    )

    x = inputs

    # --------------------------------------------------------
    # Four convolutional blocks
    # --------------------------------------------------------

    input_channels = INPUT_CHANNELS

    for index, filters in enumerate(FILTERS):

        block_number = index + 1

        print(
            f"Building Block {block_number}: "
            f"{input_channels} -> {filters} filters"
        )

        x = convolutional_block(
            inputs=x,
            filters=filters,
            dropout_rate=DROPOUT_RATES[index],
            block_number=block_number,
        )

        input_channels = filters

    # --------------------------------------------------------
    # Global Average Pooling
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    # --------------------------------------------------------
    # Dense classification head
    # --------------------------------------------------------

    x = tf.keras.layers.Dense(
        units=DENSE_UNITS,
        activation="relu",
        use_bias=True,
        name="dense_128",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=0.40,
        name="dense_dropout",
    )(x)

    # --------------------------------------------------------
    # Binary classification output
    # --------------------------------------------------------

    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        use_bias=True,
        name="output_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=MODEL_NAME,
    )

    return model


# ============================================================
# 4. MANUAL PARAMETER CALCULATION
# ============================================================

def calculate_manual_parameters():
    """
    Calculate model parameters independently.

    Conv2D with use_bias=False:

        P_conv = K_h * K_w * C_in * C_out

    Batch Normalization:

        Trainable:
            Gamma + Beta = 2 * C

        Non-trainable:
            Moving Mean + Moving Variance = 2 * C

    Dense:

        P_dense = input_units * output_units
                  + output_units (bias)
    """

    block_results = []

    total_trainable = 0
    total_non_trainable = 0

    input_channels = INPUT_CHANNELS

    # --------------------------------------------------------
    # Calculate four convolutional blocks
    # --------------------------------------------------------

    for block_number, filters in enumerate(FILTERS, start=1):

        # First convolution
        conv1_params = (
            KERNEL_SIZE
            * KERNEL_SIZE
            * input_channels
            * filters
        )

        # Second convolution
        conv2_params = (
            KERNEL_SIZE
            * KERNEL_SIZE
            * filters
            * filters
        )

        conv_params = conv1_params + conv2_params

        # Two Batch Normalization layers.
        bn_trainable = 2 * filters * 2
        bn_non_trainable = 2 * filters * 2

        block_trainable = conv_params + bn_trainable

        block_non_trainable = bn_non_trainable

        block_total = (
            block_trainable + block_non_trainable
        )

        block_results.append({
            "block": block_number,
            "filters": filters,
            "conv1": conv1_params,
            "conv2": conv2_params,
            "conv_total": conv_params,
            "bn_trainable": bn_trainable,
            "bn_non_trainable": bn_non_trainable,
            "trainable": block_trainable,
            "non_trainable": block_non_trainable,
            "total": block_total,
        })

        total_trainable += block_trainable
        total_non_trainable += block_non_trainable

        input_channels = filters

    # --------------------------------------------------------
    # Global Average Pooling
    # --------------------------------------------------------

    gap_features = FILTERS[-1]

    # Global Average Pooling has no trainable parameters.
    gap_params = 0

    # --------------------------------------------------------
    # Dense 128
    # --------------------------------------------------------

    dense_params = (
        gap_features * DENSE_UNITS
        + DENSE_UNITS
    )

    # --------------------------------------------------------
    # Output Dense 1
    # --------------------------------------------------------

    output_params = (
        DENSE_UNITS * 1
        + 1
    )

    dense_head_params = dense_params + output_params

    total_trainable += dense_head_params

    # --------------------------------------------------------
    # Final total
    # --------------------------------------------------------

    total_params = (
        total_trainable + total_non_trainable
    )

    return {
        "blocks": block_results,
        "gap": gap_params,
        "dense": dense_params,
        "output": output_params,
        "dense_head": dense_head_params,
        "trainable": total_trainable,
        "non_trainable": total_non_trainable,
        "total": total_params,
    }


# ============================================================
# 5. PRINT MANUAL PARAMETER BREAKDOWN
# ============================================================

def print_manual_parameters(manual):
    """Print parameter calculations for each component."""

    print("\n" + "=" * 75)
    print("MANUAL PARAMETER CALCULATION")
    print("=" * 75)

    print(
        f"{'Component':<18}"
        f"{'Conv1':>12}"
        f"{'Conv2':>12}"
        f"{'BN trainable':>16}"
        f"{'BN non-train':>16}"
        f"{'Total':>12}"
    )

    print("-" * 86)

    for block in manual["blocks"]:

        print(
            f"Block {block['block']:<12}"
            f"{block['conv1']:>12,}"
            f"{block['conv2']:>12,}"
            f"{block['bn_trainable']:>16,}"
            f"{block['bn_non_trainable']:>16,}"
            f"{block['total']:>12,}"
        )

    print("-" * 86)

    print(
        f"{'GAP':<18}"
        f"{manual['gap']:>68,}"
    )

    print(
        f"{'Dense 128':<18}"
        f"{manual['dense']:>68,}"
    )

    print(
        f"{'Output Dense 1':<18}"
        f"{manual['output']:>68,}"
    )

    print("-" * 86)

    print(
        f"{'Trainable':<30}"
        f"{manual['trainable']:>56,}"
    )

    print(
        f"{'Non-trainable':<30}"
        f"{manual['non_trainable']:>56,}"
    )

    print(
        f"{'TOTAL PARAMETERS':<30}"
        f"{manual['total']:>56,}"
    )

    print("=" * 75)


# ============================================================
# 6. VERIFY MODEL PARAMETERS AGAINST KERAS
# ============================================================

def count_keras_weights(weights):
    """Count the number of scalar parameters in a weight list."""

    return sum(
        int(weight.shape.num_elements())
        for weight in weights
    )


def verify_parameter_counts(model, manual):
    """
    Compare manual parameter counts with Keras.

    Verify:
        Total parameters
        Trainable parameters
        Non-trainable parameters
    """

    keras_total = model.count_params()

    keras_trainable = count_keras_weights(
        model.trainable_weights
    )

    keras_non_trainable = count_keras_weights(
        model.non_trainable_weights
    )

    print("\n" + "=" * 75)
    print("MANUAL VS KERAS PARAMETER VERIFICATION")
    print("=" * 75)

    print(
        f"{'Parameter type':<25}"
        f"{'Manual':>20}"
        f"{'Keras':>20}"
        f"{'Match':>12}"
    )

    print("-" * 77)

    comparisons = [
        (
            "Trainable",
            manual["trainable"],
            keras_trainable,
        ),
        (
            "Non-trainable",
            manual["non_trainable"],
            keras_non_trainable,
        ),
        (
            "Total",
            manual["total"],
            keras_total,
        ),
    ]

    for name, manual_value, keras_value in comparisons:

        matched = manual_value == keras_value

        print(
            f"{name:<25}"
            f"{manual_value:>20,}"
            f"{keras_value:>20,}"
            f"{'PASS' if matched else 'FAIL':>12}"
        )

        if not matched:
            raise AssertionError(
                f"{name} parameter mismatch: "
                f"manual={manual_value}, Keras={keras_value}"
            )

    # Verify against fixed expected counts.
    assert keras_total == EXPECTED_TOTAL_PARAMS
    assert keras_trainable == EXPECTED_TRAINABLE_PARAMS
    assert keras_non_trainable == EXPECTED_NON_TRAINABLE_PARAMS

    print("\nPASS: Manual parameter counts match Keras.")
    print("PASS: Model parameter totals match expected values.")

    print("=" * 75)

    return {
        "total": keras_total,
        "trainable": keras_trainable,
        "non_trainable": keras_non_trainable,
    }


# ============================================================
# 7. VERIFY INTERMEDIATE TENSOR SHAPES
# ============================================================

def verify_tensor_shapes(model):
    """
    Verify output shapes of convolutional blocks,
    pooling layers, GAP, and classification head.
    """

    expected_shapes = {
        "block1_conv1": (None, 224, 224, 32),
        "block1_bn1": (None, 224, 224, 32),
        "block1_relu1": (None, 224, 224, 32),
        "block1_conv2": (None, 224, 224, 32),
        "block1_bn2": (None, 224, 224, 32),
        "block1_relu2": (None, 224, 224, 32),
        "block1_pool": (None, 112, 112, 32),
        "block1_dropout": (None, 112, 112, 32),

        "block2_conv1": (None, 112, 112, 64),
        "block2_bn1": (None, 112, 112, 64),
        "block2_relu1": (None, 112, 112, 64),
        "block2_conv2": (None, 112, 112, 64),
        "block2_bn2": (None, 112, 112, 64),
        "block2_relu2": (None, 112, 112, 64),
        "block2_pool": (None, 56, 56, 64),
        "block2_dropout": (None, 56, 56, 64),

        "block3_conv1": (None, 56, 56, 128),
        "block3_bn1": (None, 56, 56, 128),
        "block3_relu1": (None, 56, 56, 128),
        "block3_conv2": (None, 56, 56, 128),
        "block3_bn2": (None, 56, 56, 128),
        "block3_relu2": (None, 56, 56, 128),
        "block3_pool": (None, 28, 28, 128),
        "block3_dropout": (None, 28, 28, 128),

        "block4_conv1": (None, 28, 28, 256),
        "block4_bn1": (None, 28, 28, 256),
        "block4_relu1": (None, 28, 28, 256),
        "block4_conv2": (None, 28, 28, 256),
        "block4_bn2": (None, 28, 28, 256),
        "block4_relu2": (None, 28, 28, 256),
        "block4_pool": (None, 14, 14, 256),
        "block4_dropout": (None, 14, 14, 256),

        "global_average_pooling": (None, 256),
        "dense_128": (None, 128),
        "dense_dropout": (None, 128),
        "output_probability": (None, 1),
    }

    print("\n" + "=" * 75)
    print("TENSOR SHAPE VERIFICATION")
    print("=" * 75)

    print(f"{'Layer':<27}{'Expected shape':<25}{'Result':>12}")

    print("-" * 66)

    # Verify model input.
    if tuple(model.input_shape) != (
        None,
        IMAGE_SIZE,
        IMAGE_SIZE,
        INPUT_CHANNELS,
    ):
        raise AssertionError(
            f"Incorrect model input shape: {model.input_shape}"
        )

    print(
        f"{'image_input':<27}"
        f"{str(model.input_shape):<25}"
        f"{'PASS':>12}"
    )

    for layer_name, expected_shape in expected_shapes.items():

        actual_shape = tuple(
            model.get_layer(layer_name).output.shape
        )

        matched = actual_shape == expected_shape

        print(
            f"{layer_name:<27}"
            f"{str(actual_shape):<25}"
            f"{'PASS' if matched else 'FAIL':>12}"
        )

        if not matched:
            raise AssertionError(
                f"Shape mismatch at {layer_name}: "
                f"expected {expected_shape}, got {actual_shape}"
            )

    if tuple(model.output_shape) != (None, 1):
        raise AssertionError(
            f"Incorrect final output shape: {model.output_shape}"
        )

    print("\nPASS: All intermediate tensor shapes are correct.")

    print("=" * 75)


# ============================================================
# 8. MAIN ARCHITECTURE VERIFICATION
# ============================================================

def main():

    print("=" * 75)
    print("DL-05 CNN FROM SCRATCH – MODEL V1")
    print("=" * 75)

    print("\nTensorFlow version:", tf.__version__)

    print(
        "Detected GPU devices:",
        tf.config.list_physical_devices("GPU")
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\n[1] BUILDING CNN ARCHITECTURE")

    model = build_cnn_from_scratch()

    print("\nArchitecture construction completed.")

    # --------------------------------------------------------
    # Keras model summary
    # --------------------------------------------------------

    print("\n[2] KERAS MODEL SUMMARY")

    model.summary()

    # --------------------------------------------------------
    # Manual parameter calculation
    # --------------------------------------------------------

    print("\n[3] MANUAL PARAMETER CALCULATION")

    manual = calculate_manual_parameters()

    print_manual_parameters(manual)

    # --------------------------------------------------------
    # Compare manual parameters with Keras
    # --------------------------------------------------------

    print("\n[4] PARAMETER COUNT VERIFICATION")

    parameter_results = verify_parameter_counts(
        model,
        manual
    )

    # --------------------------------------------------------
    # Verify tensor shapes
    # --------------------------------------------------------

    print("\n[5] TENSOR SHAPE VERIFICATION")

    verify_tensor_shapes(model)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("CNN FROM SCRATCH V1 – FINAL VERIFICATION")
    print("=" * 75)

    print("Model name:", model.name)

    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    print(
        "Total parameters:",
        f"{parameter_results['total']:,}"
    )

    print(
        "Trainable parameters:",
        f"{parameter_results['trainable']:,}"
    )

    print(
        "Non-trainable parameters:",
        f"{parameter_results['non_trainable']:,}"
    )

    print("\n[FINAL RESULT]")
    print("PASS: CNN architecture built successfully.")
    print("PASS: All tensor shapes verified.")
    print("PASS: Manual parameter count matches Keras.")
    print("PASS: Expected parameter totals confirmed.")

    print("\nNOTE:")
    print("Model architecture only.")
    print("Training has NOT started.")

    print("=" * 75)


# ============================================================
# 9. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
