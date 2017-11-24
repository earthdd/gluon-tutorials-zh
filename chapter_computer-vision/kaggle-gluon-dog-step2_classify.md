### Demo与否相关参数设定


```{.python .input  n=1}
demo = False
data_dir = '../data/kaggle_dog'

if demo:
    # 注意：此处使用小数据集为便于网页编译。
    input_dir = 'train_valid_test_tiny'
    # 注意：此处相应使用小批量。对Kaggle的完整数据集可设较大的整数，例如128。
    batch_size = 2
else:
    label_file = 'labels.csv'
    input_dir = 'train_valid_test'
    batch_size = 128
    
```

## read proprocessed data from disk

读取用pretrained net处理过的数据

接下来，我们可以使用`Gluon`中的`ImageFolderDataset`类来读取整理后的数据集。

```{.python .input  n=2}
from mxnet import nd
from mxnet import gluon

input_str = data_dir + '/' + input_dir + '/'
proprocessed_dir = input_str + '/proprocessed'
# proprocessed_dir = data_dir + '/proprocessed'

train_nd = nd.load(proprocessed_dir + '/train_data.nd')
valid_nd = nd.load(proprocessed_dir + '/valid_data.nd')
train_valid_nd = nd.load(proprocessed_dir + '/train_valid_data.nd')
test_nd = nd.load(proprocessed_dir + '/test_data.nd')

loader = gluon.data.DataLoader
arrayds = gluon.data.ArrayDataset

train_data = loader(arrayds(train_nd[0], train_nd[1]), batch_size, shuffle=True, last_batch='keep')
valid_data = loader(arrayds(valid_nd[0], valid_nd[1]), batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(arrayds(train_valid_nd[0], train_valid_nd[1]), batch_size, shuffle=True, last_batch='keep')
test_data = loader(arrayds(test_nd[0], test_nd[1]), batch_size, shuffle=False, last_batch='keep')

# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.json .output n=2}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\Users\\husu\\AppData\\Local\\Continuum\\anaconda3\\envs\\gluon\\lib\\site-packages\\urllib3\\contrib\\pyopenssl.py:46: DeprecationWarning: OpenSSL.rand is deprecated - you should use os.urandom instead\n  import OpenSSL.SSL\n"
 }
]
```

## 设计模型

我们这里使用了[ResNet-18](resnet-gluon.md)模型。我们使用[hybridizing](../chapter_gluon-advances/hybridize.md)来提升执行效率。

请注意：模型可以重新设计，参数也可以重新调整。

```{.python .input  n=3}
from mxnet.gluon import nn
from mxnet import init

def get_net(ctx):
    num_outputs = 120
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Flatten())
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(num_outputs))
    net.initialize(ctx=ctx, init=init.Xavier())
    return net
```

## 训练模型并调参

在[过拟合](../chapter_supervised-learning/underfit-overfit.md)中我们讲过，过度依赖训练数据集的误差来推断测试数据集的误差容易导致过拟合。由于图像分类训练时间可能较长，为了方便，我们这里不再使用K折交叉验证，而是依赖验证集的结果来调参。

我们定义损失函数以便于计算验证集上的损失函数值。我们也定义了模型训练函数，其中的优化算法和参数都是可以调的。

```{.python .input  n=4}
from mxnet import autograd
import datetime
import sys
sys.path.append('..')
import utils

def get_loss(data, net, ctx):
    loss = 0.0
    for feas, label in data:
        label = label.as_in_context(ctx)
        output = net(feas.as_in_context(ctx))
        cross_entropy = softmax_cross_entropy(output, label)
        loss += nd.mean(cross_entropy).asscalar()
    return loss / len(data)

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, 
          lr_decay):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 
                                      'wd': wd})
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:  
            valid_loss = get_loss(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Train loss: %f, Valid loss %f, "
                         % (epoch, train_loss / len(train_data), valid_loss))
        else:
            epoch_str = ("Epoch %d. Train loss: %f, "
                         % (epoch, train_loss / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
```

以下定义训练参数并训练模型。这些参数均可调。为了使网页编译快一点，我们这里将epoch数量有意设为1。事实上，epoch一般可以调大些。

我们将依据验证集的结果不断优化模型设计和调整参数。依据下面的参数设置，优化算法的学习率将在每80个epoch自乘0.1。

```{.python .input  n=5}
ctx = utils.try_gpu()
num_epochs = 30
learning_rate = 0.01
weight_decay = 5e-4
lr_period = 10
lr_decay = 0.1

net = get_net(ctx)
net.hybridize()
train(net, train_data, valid_data, num_epochs, learning_rate, 
      weight_decay, ctx, lr_period, lr_decay)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Train loss: 2.917160, Valid loss 1.016714, Time 00:00:29, lr 0.01\nEpoch 1. Train loss: 1.138208, Valid loss 0.705681, Time 00:00:27, lr 0.01\nEpoch 2. Train loss: 0.782233, Valid loss 0.632612, Time 00:00:22, lr 0.01\nEpoch 3. Train loss: 0.641772, Valid loss 0.592312, Time 00:00:22, lr 0.01\nEpoch 4. Train loss: 0.517954, Valid loss 0.551354, Time 00:00:22, lr 0.01\nEpoch 5. Train loss: 0.438300, Valid loss 0.572900, Time 00:00:22, lr 0.01\nEpoch 6. Train loss: 0.372386, Valid loss 0.540294, Time 00:00:22, lr 0.01\nEpoch 7. Train loss: 0.327992, Valid loss 0.546717, Time 00:00:23, lr 0.01\nEpoch 8. Train loss: 0.289893, Valid loss 0.543142, Time 00:00:22, lr 0.01\nEpoch 9. Train loss: 0.260809, Valid loss 0.534260, Time 00:00:23, lr 0.01\nEpoch 10. Train loss: 0.232882, Valid loss 0.525505, Time 00:00:22, lr 0.001\nEpoch 11. Train loss: 0.196240, Valid loss 0.518909, Time 00:00:22, lr 0.001\nEpoch 12. Train loss: 0.195859, Valid loss 0.520569, Time 00:00:22, lr 0.001\nEpoch 13. Train loss: 0.191452, Valid loss 0.524706, Time 00:00:22, lr 0.001\nEpoch 14. Train loss: 0.191333, Valid loss 0.522117, Time 00:00:22, lr 0.001\nEpoch 15. Train loss: 0.184009, Valid loss 0.515059, Time 00:00:26, lr 0.001\nEpoch 16. Train loss: 0.186567, Valid loss 0.515835, Time 00:00:22, lr 0.001\nEpoch 17. Train loss: 0.185304, Valid loss 0.524362, Time 00:00:22, lr 0.001\nEpoch 18. Train loss: 0.182711, Valid loss 0.537020, Time 00:00:22, lr 0.001\nEpoch 19. Train loss: 0.177968, Valid loss 0.527751, Time 00:00:22, lr 0.001\nEpoch 20. Train loss: 0.174044, Valid loss 0.515304, Time 00:00:22, lr 0.0001\nEpoch 21. Train loss: 0.174670, Valid loss 0.518902, Time 00:00:23, lr 0.0001\nEpoch 22. Train loss: 0.171055, Valid loss 0.524707, Time 00:00:22, lr 0.0001\nEpoch 23. Train loss: 0.169959, Valid loss 0.522188, Time 00:00:22, lr 0.0001\nEpoch 24. Train loss: 0.173385, Valid loss 0.510974, Time 00:00:21, lr 0.0001\nEpoch 25. Train loss: 0.174661, Valid loss 0.513951, Time 00:00:23, lr 0.0001\nEpoch 26. Train loss: 0.166021, Valid loss 0.523753, Time 00:00:22, lr 0.0001\nEpoch 27. Train loss: 0.170792, Valid loss 0.507991, Time 00:00:22, lr 0.0001\nEpoch 28. Train loss: 0.167269, Valid loss 0.519596, Time 00:00:22, lr 0.0001\nEpoch 29. Train loss: 0.170182, Valid loss 0.518296, Time 00:00:22, lr 0.0001\n"
 }
]
```

## 对测试集分类

当得到一组满意的模型设计和参数后，我们使用全部训练数据集（含验证集）重新训练模型，并对测试集分类。

```{.python .input  n=6}
from mxnet.gluon.data import vision

# get list of class name
def get_synsets():
    input_str = data_dir + '/' + input_dir + '/'
    train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid')
    class_li = train_valid_ds.synsets
    return class_li
    
```

```{.python .input  n=7}
import numpy as np
import os

net = get_net(ctx)
net.hybridize()
train(net, train_valid_data, None, num_epochs, learning_rate, weight_decay, 
      ctx, lr_period, lr_decay)

outputs = []
for data, label in test_data:
    output = nd.softmax(net(data.as_in_context(ctx)))
    outputs.extend(output.asnumpy())
ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
with open('submission.csv', 'w') as f:
    # f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    f.write('id,' + ','.join(get_synsets()) + '\n')
    for i, output in zip(ids, outputs):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Train loss: 2.832714, Time 00:00:25, lr 0.01\nEpoch 1. Train loss: 1.109276, Time 00:00:22, lr 0.01\nEpoch 2. Train loss: 0.791688, Time 00:00:22, lr 0.01\nEpoch 3. Train loss: 0.633717, Time 00:00:22, lr 0.01\nEpoch 4. Train loss: 0.509352, Time 00:00:22, lr 0.01\nEpoch 5. Train loss: 0.450789, Time 00:00:23, lr 0.01\nEpoch 6. Train loss: 0.386631, Time 00:00:22, lr 0.01\nEpoch 7. Train loss: 0.333381, Time 00:00:24, lr 0.01\nEpoch 8. Train loss: 0.301623, Time 00:00:23, lr 0.01\nEpoch 9. Train loss: 0.269193, Time 00:00:22, lr 0.01\nEpoch 10. Train loss: 0.237022, Time 00:00:22, lr 0.001\nEpoch 11. Train loss: 0.217101, Time 00:00:22, lr 0.001\nEpoch 12. Train loss: 0.206596, Time 00:00:22, lr 0.001\nEpoch 13. Train loss: 0.197277, Time 00:00:22, lr 0.001\nEpoch 14. Train loss: 0.194637, Time 00:00:22, lr 0.001\nEpoch 15. Train loss: 0.189385, Time 00:00:22, lr 0.001\nEpoch 16. Train loss: 0.191711, Time 00:00:23, lr 0.001\nEpoch 17. Train loss: 0.182127, Time 00:00:22, lr 0.001\nEpoch 18. Train loss: 0.189352, Time 00:00:22, lr 0.001\nEpoch 19. Train loss: 0.182687, Time 00:00:22, lr 0.001\nEpoch 20. Train loss: 0.182256, Time 00:00:21, lr 0.0001\nEpoch 21. Train loss: 0.176891, Time 00:00:23, lr 0.0001\nEpoch 22. Train loss: 0.176145, Time 00:00:22, lr 0.0001\nEpoch 23. Train loss: 0.178180, Time 00:00:22, lr 0.0001\nEpoch 24. Train loss: 0.178318, Time 00:00:22, lr 0.0001\nEpoch 25. Train loss: 0.175926, Time 00:00:22, lr 0.0001\nEpoch 26. Train loss: 0.179215, Time 00:00:22, lr 0.0001\nEpoch 27. Train loss: 0.168955, Time 00:00:21, lr 0.0001\nEpoch 28. Train loss: 0.176770, Time 00:00:22, lr 0.0001\nEpoch 29. Train loss: 0.174846, Time 00:00:22, lr 0.0001\n"
 }
]
```

上述代码执行完会生成一个`submission.csv`的文件用于在Kaggle上提交。这是Kaggle要求的提交格式。这时我们可以在Kaggle上把对测试集分类的结果提交并查看分类准确率。你需要登录Kaggle网站，打开[120种狗类识别问题](https://www.kaggle.com/c/dog-breed-identification)，并点击下方右侧`Submit Predictions`按钮。
温馨提醒，目前**Kaggle仅限每个账号一天以内5次提交结果的机会**。所以提交结果前务必三思。

## 作业：

* 使用Kaggle完整数据集，把batch_size和num_epochs分别调大些，可以在Kaggle上拿到什么样的准确率和名次？
* 你还有什么其他办法可以继续改进模型和参数？小伙伴们都期待你的分享。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/2399)
