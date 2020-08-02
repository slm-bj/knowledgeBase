# Tensorflow 分布式训练总结

本文档由两部分组成，第一部分包含配置多 GPU 深度学习环境，并运行验证代码。

第二部分说明目前 Tensorflow Kares API 支持的几种分布式训练策略，
目前的支持程度等。

## 单机软件安装与验证

首先创建用户，安装必要的软件：
```
useradd -m -s /bin/bash -G wheel avatar
passwd avatar
su - avatar
sudo yum install git wget
```

### 安装 GPU 驱动

```
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/415.27/NVIDIA-Linux-x86_64-415.27.run
chmod 755 NVIDIA-Linux-x86_64-415.27.run
sudo ./NVIDIA-Linux-x86_64-415.27.run
```

安装程序提示更高版本驱动已安装，此过程终止，
检查驱动安装效果（smi means system management interface）：
```
$ nvidia-smi
Wed Jun  3 07:18:08 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 207...  Off  | 00000000:01:00.0 Off |                  N/A |
| 28%   43C    P0    25W / 215W |      0MiB /  7979MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

输出包含两张表格，第一张主要是汇总信息，第二张主要是实时使用情况，
下面分别说明。

#### 检查 GPU 内存容量

```
$ pip install gpustat   # or: conda install -c conda-forge gpustat
$ gpustat
centos7                    Wed Jun  3 08:34:47 2020  440.33.01
[0] GeForce RTX 2070 SUPER | 50'C,   0 % |     0 /  7979 MB |
```

与上面 `nvidia-smi` 报告的 *Memory-Usage* 部分一致，都是 7979MB。

专门查询内存部分：
```
$ nvidia-smi -q -d memory

==============NVSMI LOG==============

Timestamp                           : Wed Jun  3 07:50:00 2020
Driver Version                      : 440.33.01
CUDA Version                        : 10.2

Attached GPUs                       : 1
GPU 00000000:01:00.0
    FB Memory Usage
        Total                       : 7979 MiB
        Used                        : 0 MiB
        Free                        : 7979 MiB
    BAR1 Memory Usage
        Total                       : 256 MiB
        Used                        : 2 MiB
        Free                        : 254 MiB


$ lspci | grep NVIDIA    # get the device ID is `01:00.0`
01:00.0 VGA compatible controller: NVIDIA Corporation Device 1e84 (rev a1)
...

$ lspci -v -s 01:00.0 | grep Memory
        Memory at de000000 (32-bit, non-prefetchable) [size=16M]
        Memory at c0000000 (64-bit, prefetchable) [size=256M]
        Memory at d0000000 (64-bit, prefetchable) [size=32M]
```

一个命令的 FB memory 显示设备内容容量，也是 7979 MB，即 8GB，
BAR1 memory 为 256MB，是CPU或者其他应用可以使用的内存，与 `lspci` 给出的结果一致。
具体含义参考 `man nvidia-smi` 中相关章节的说明。

#### 监控 GPU 内存实时使用情况

每两秒刷新一下监控状态：
```
nvidia-smi -l 2        # by -l option
gpustat -cp -i 2       # by -i option
watch -n 2 nvidia-smi  # by -n option of watch
```

如果 `watch` 后面的命令有颜色，
可以为 `watch` 命令添加 `-c` 选项使其正确解析颜色编码。

可以指定输出内容和格式（仍然是输出到屏幕）：
```
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### 安装 CUDA 10.2 和 CUPTI

根据 [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html),
Cuda 10.2 支持 49 服务器的 驱动版本：440.33.
```
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
# make sure CUPTI in the installation list (in cuda command line tools menu)


#: add '/usr/local/cuda-10.2/bin' into PATH
sudo echo 'export PATH=$PATH:/usr/local/cuda-10.2/bin' > /etc/profile.d/cuda.sh
sudo echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64' >> /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

#: add /usr/local/cuda-10.2/lib64 to /etc/ld.so.conf and run ldconfig as root
sudo echo '/usr/local/cuda-10.2/lib64` >> /etc/ld.so.conf.d/cuda-10-2.conf
sudo ldocnfig
```

### 安装 cuDNN SDK

注册 NVidia developer 并下载下面的文件：

* cuDNN Runtime Library for RedHat/Centos 7.3 (RPM)
* cuDNN Developer Library for RedHat/Centos 7.3 (RPM)
* cuDNN Code Samples and User Guide for RedHat/Centos 7.3 (RPM)

安装文件：
```
sudo rpm -ivh libcudnn7-7.6.5.33-1.cuda10.2.x86_64.rpm
sudo rpm -ivh libcudnn7-devel-7.6.5.33-1.cuda10.2.x86_64.rpm
sudo rpm -ivh libcudnn7-doc-7.6.5.33-1.cuda10.2.x86_64.rpm
```

### 安装 MiniConda, TensorFlow 并运行样例

```
conda create -n tfgpu
conda activate tfgpu
conda install tensorflow-gpu ipython

ipython

import tensorflow as tf

#: GPU verification

tf.test.is_built_with_cuda()  # True

#: at least one GPU working
tf.test.is_gpu_available()    # True

#: the first GPU name, where operations will run
tf.test.gpu_device_name()  # '/device:GPU:0'

#: print the list of all available GPU devices
tf.config.experimental.list_physical_devices('GPU')
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

from tensorflow.python.client import device_lib
res = device_lib.list_local_devices()
len(res)

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```


## 分布式验证

### 单机多 GPU 场景

本场景对应的分布式训练策略是 `tf.distribute.MirroredStrategy`。

验证代码：
```
import tensorflow as tf
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope(): 
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))]) 
    model.compile(loss='mse', optimizer='sgd')
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
model.evaluate(dataset)

import numpy as np
inputs, targets = np.ones((100,1)), np.ones((100, 1))
model.fit(inputs, targets, epochs=2, batch_size=10)
```

第二个验证用例（2020.6.14 于 50 服务器上验证通过）：
```
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

### 多主机多 GPU 场景

#### MultiWorkerMirroredStrategy

##### 单节点验证

首先在每个节点上验证：
```
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import numpy as np
import os
import json

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

per_worker_batch_size = 64

single_worker_dataset = mnist_dataset(per_worker_batch_size)
single_worker_model = build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)
```

##### 单主机上多节点场景

不论多个 worker 是不是在一台主机上，每个 worker 对应一个模型训练脚本，
这些脚本除了 TF_CONFIG.task.index 不同，其他完全一样。

依次执行这些脚本，先启动的会等待后启动的，直到最后一个 worker 启动后开始训练。

下面的脚本复制一份并将 TF_CONFIG.task.index 改为 1 后，
在 50 上运行成功：
```
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import numpy as np
import os
import json

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

num_workers = 4
per_worker_batch_size = 64

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
```

##### 多主机多节点场景

运行脚本与上节相同，但 49 上的配置为：
```
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["192.168.2.49:12345", "192.168.2.50:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
```

50 的配置为：
```
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["192.168.2.49:12345", "192.168.2.50:23456"]
    },
    'task': {'type': 'worker', 'index': 1}
})
```

分别在两台主机上启动脚本，由于 tensorfow Keras API 的 bug "MultiWorkerMirroredStrategy does not work with Keras + accuracy metric #33531"，
在计算第一个 epoch 时卡死，社区目前还没有 fix 这个 bug，
基于 Keras API 的多主机多节点的分布式训练目前不可用。


#### 非镜像策略

本节使用 ParameterServerStrategy 运行分布式训练。

根据 [Distributed training with TensorFlow](https://www.tensorflow.org/guide/distributed_training#types_of_strategies)，
TF.Keras 在 2.3 之后支持 ParameterServerStrategy，
目前的 Keras 分布式教程 [Multi-worker training with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)
只使用 MultiWorkerMirroredStrategy，
所以等后续 Keras 支持 ParameterServerStrategy 后再验证。

## 参考文献

* Chapter 19 of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd edition.

* [Distributed training with TensorFlow](https://www.tensorflow.org/guide/distributed_training)

* [Distributed training with Keras](https://www.tensorflow.org/tutorials/distribute/keras)

