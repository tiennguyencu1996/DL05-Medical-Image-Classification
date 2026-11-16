import platform
import tensorflow as tf

print("=" * 60)
print("DL-05: TENSORFLOW AND APPLE GPU VERIFICATION")
print("=" * 60)

# 1. Environment information
print("\n[1] Environment")
print("Architecture:", platform.machine())
print("TensorFlow:", tf.__version__)

# 2. List available devices
print("\n[2] Available devices")

for device in tf.config.list_physical_devices():
    print(device)

# 3. Detect GPU
print("\n[3] GPU detection")

gpus = tf.config.list_physical_devices("GPU")

if not gpus:
    print("GPU NOT DETECTED")
    print("Please investigate TensorFlow Metal configuration.")

else:
    print("GPU detected:", len(gpus))

    # 4. Run matrix multiplication on GPU
    print("\n[4] GPU computation test")

    with tf.device("/GPU:0"):
        a = tf.random.normal((1024, 1024))
        b = tf.random.normal((1024, 1024))
        result = tf.matmul(a, b)

        # Force execution and retrieve the result
        output = result.numpy()

    print("Requested device: /GPU:0")
    print("Actual tensor device:", result.device)
    print("Output shape:", output.shape)

    if "GPU:0" in result.device.upper():
        print("\nSUCCESS: GPU computation verified!")
    else:
        print("\nWARNING: Result was not placed on GPU.")

print("\n" + "=" * 60)