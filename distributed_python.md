# Python 并行和分布式计算实施方案概述

Python 分布式计算的三个主要领域：

* 数据表格和统计学习：主要数据类型：Series, Dataframe，计算框架：numpy, pandas, scikit-learn
* 通用分布式计算：不针对特定的数据结构。
* 深度学习：主要数据类型：Tensor，计算框架：Tensorflow, Keras

下面分别予以说明。

## 数据表格和统计学习

这类框架的核心是实现分布式数据存储和计算，
计算规模不受单点内存大小限制，只受集群总内存限制（可以通过动态增加节点调整上限），
从而实现算子级别的分布式计算。

与 Spark 类似，这些框架都使用了分布式数据结构，
与单点计算框架的数据结构不完全兼容（虽然兼容度已经相当高了）。

### Dask

采用 Dask 自己的数据类型（与 Pandas 十分类似，大部分是一一对应的）实现分布式计算。
用户数量大，社区活跃。

优点：社区庞大，开发活跃，用户数量大，代码迁移难度不大，
与 pandas, numpy, scikit-learn 良好的兼容性，支持 Hadoop 文件系统。
由于采用了新的数据结构，分布式计算不支持的函数在开发阶段就会被发现。

缺点：需要手工更改数据类型，pandas dataframe 改为 dask dataframe，

### Ray

采用标记方法将本地类和函数变为分布式函数。

优点：用户数量大，开发活跃，代码迁移难度不大，分布式计算采用 actor 结构，性能比较高。

缺点：分布式版本的兼容性问题检查比 Dask 难度略高，尚不清楚对 scikit-learn 的支持程度。

### modin

用户友好的分布式计算方案，pandas 代码本地到分布式改写非常容易，
底层可以自由选择 Dask 或者 Ray 作为计算引擎。

优点：降低了本地到分布式算子转换难度。

缺点：增加了一层抽象，稳定性和兼容性需要验证。

### pandarallel

pandas 的分布式计算版本，社区比较活跃。

优点：本地到分布式改写方便，函数一对一映射。

缺点：只适用于 pandas 对象，对 scikit-learn 支持有限。

### sparkit-learn

为 PySpark 添加 Scikit-learn 支持。

近两年开发不活跃。

### PySpark

优点：较好的分布式基础设施。

缺点：Spark API + pandas 函数，库支持有限。

## 通用分布式计算

通用分布式计算框架提供分布式通讯基础设施，类似于 Spark 使用 Akka 作为分布式通信框架。
使用这类工具，可以实现更灵活的分布式计算策略。
比如对于规模不超过单个服务器内存的计算任务，可以采用单进程方式运行，
避免分布式计算带来的网络传输开销，
平台根据服务器的负载情况将同时运行的多个任务分配给资源充足的服务器，
从而实现平台级别的分布式计算。

这类工具一般没有对 pandas, numpy 的开箱支持，不太适合作为算子级别的分布式框架。

目前应用比较广泛的方案有：

* Celery: 任务调度系统，相当于 Python 的 Zookeeper。Python 社区用户最多的任务调度系统。

* fabric: 配置管理框架，方便地管理大量服务器，监控状态，分发任务。

* dispy: Python 通用分布式计算框架，社区活跃程度一般。

* Fiber: 使用 Python multiprocessing 相同的 API 提供分布式计算功能，易于部署。

## 深度学习

除了 Tensorflow 的 Keras API，还有其他解决方案，但功能和稳定性有待验证。

### Elephas

使用 Keras API，底层计算交给 Spark 完成，社区和活跃程度一般。

优点：Keras 代码适用于单机和分布式环境。

缺点：只适用于 Keras，不支持 pandas，scikit-learn

## 小结

针对不同规模的算子（包括统计学习和深度学习）采用不同的计算策略，
优先使用平台级别的分布式计算框架，如果规模超过了单点内存限制，
则采用算子级别的分布式计算框架。

## 参考文献

* [modin](https://github.com/modin-project/modin)
* [dask](https://github.com/dask/dask)
* [elephas](https://github.com/maxpumperla/elephas)
* [ray](https://github.com/ray-project/ray)
* [fiber](https://github.com/uber/fiber)
* [celery](https://github.com/celery/celery)
* [pandarallel](https://github.com/nalepae/pandarallel)
* [sparkit-learn](https://github.com/lensacom/sparkit-learn)
* [dispy](https://github.com/pgiri/dispy)

