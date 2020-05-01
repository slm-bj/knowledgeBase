# Setup Keras Model Training Env on Distributed GPUs

## Hardware Setup

NVidia RTX 2060

## Install Tensorflow 2.0

```
pip install --upgrade pip
pip install tensorflow
```

Verify:

> Use tf.config.experimental.list_physical_devices('GPU') to confirm that TensorFlow is using the GPU.

Check if CUDA is needed here.

Ref:

* https://www.tensorflow.org/install/gpu

* [Use a GPU](https://www.tensorflow.org/guide/gpu)

## Install Keras

```
pip install keras
```

Ref:

* https://keras.io/#installation

* [Keras has strong multi-GPU support and distributed training support](https://keras.io/utils/#multi_gpu_model)

## Code running on multiple GPUs

[How can I run a Keras model on multiple GPUs?](https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus)

