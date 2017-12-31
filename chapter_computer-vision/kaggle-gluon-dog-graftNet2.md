```{.python .input  n=11}
# 如果训练下载的Kaggle的完整数据集，把demo改为False。
demo = False
data_dir = '../data/kaggle_dog'

if demo:
    zipfiles= ['train_valid_test_tiny.zip']
else:
    zipfiles= ['train.zip', 'test.zip', 'labels.csv.zip']

import zipfile
for fin in zipfiles:
    with zipfile.ZipFile(data_dir + '/' + fin, 'r') as zin:
        zin.extractall(data_dir)
```

### 整理数据集

对于Kaggle的完整数据集，我们需要定义下面的reorg_dog_data函数来整理一下。整理后，同一类狗的图片将出现在在同一个文件夹下，便于`Gluon`稍后读取。

函数中的参数如data_dir、train_dir和test_dir对应上述数据存放路径及原始训练和测试的图片集文件夹名称。参数label_file为训练数据标签的文件名称。参数input_dir是整理后数据集文件夹名称。参数valid_ratio是验证集中每类狗的数量占原始训练集中数量最少一类的狗的数量（66）的比重。

```{.python .input  n=12}
import math
import os
import shutil
from collections import Counter

def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, 
                   valid_ratio):
    # 读取训练数据标签。
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）。
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))
    labels = set(idx_label.values())

    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    # 训练集中数量最少一类的狗的数量。
    min_num_train_per_label = (
        Counter(idx_label.values()).most_common()[:-2:-1][0][1])
    # 验证集中每类狗的数量。
    num_valid_per_label = math.floor(min_num_train_per_label * valid_ratio)
    label_count = dict()

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练和验证集。
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = train_file.split('.')[0]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < num_valid_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))

    # 整理测试集。
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))
```

再次强调，为了使网页编译快一点，我们在这里仅仅使用小数据样本。相应地，我们仅将批量大小设为2。实际训练和测试时应使用Kaggle的完整数据集并调用reorg_dog_data函数整理便于`Gluon`读取的格式。由于数据集较大，批量大小batch_size大小可设为一个较大的整数，例如128。

```{.python .input  n=13}
if demo:
    # 注意：此处使用小数据集为便于网页编译。
    input_dir = 'train_valid_test_tiny'
    # 注意：此处相应使用小批量。对Kaggle的完整数据集可设较大的整数，例如128。
    batch_size = 2
else:
    label_file = 'labels.csv'
    train_dir = 'train'
    test_dir = 'test'
    input_dir = 'train_valid_test'
    batch_size = 128
    valid_ratio = 0.1 
    reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, 
                   valid_ratio)
```

## 使用Gluon读取整理后的数据集

为避免过拟合，我们在这里使用`image.CreateAugmenter`来增广数据集。例如我们设`rand_mirror=True`即可随机对每张图片做镜面反转。以下我们列举了该函数里的所有参数，这些参数都是可以调的。

```{.python .input  n=14}
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np

def transform_train(data, label):
    im = image.imresize(data.astype('float32') / 255, 96, 96)
    auglist = image.CreateAugmenter(data_shape=(3, 96, 96), resize=0, 
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=None, std=None, 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

def transform_test(data, label):
    im = image.imresize(data.astype('float32') / 255, 96, 96)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))
```

接下来，我们可以使用`Gluon`中的`ImageFolderDataset`类来读取整理后的数据集。

```{.python .input  n=15}
input_str = data_dir + '/' + input_dir + '/'

# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1, 
                                     transform=transform_train)
valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1, 
                                     transform=transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid', 
                                           flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1, 
                                     transform=transform_test)

loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, 
                          last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')

# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 设计模型

我们这里使用了[ResNet-18](resnet-gluon.md)模型。我们使用[hybridizing](../chapter_gluon-advances/hybridize.md)来提升执行效率。

请注意：模型可以重新设计，参数也可以重新调整。

```{.python .input  n=16}
"""
My implementation: 
   Simple Net
"""

from mxnet.gluon import nn
from mxnet import nd

class Net_covden(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(Net_covden, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # 模块1
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            net.add(nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1))
            #net.add(nn.Conv2D(channels=1024, kernel_size=1, strides=2))
            # 模块2
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            net.add(nn.AvgPool2D(pool_size=1, strides=1))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out

class Net_den(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(Net_den, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            #net.add(nn.AvgPool2D(pool_size=1, strides=1))
            net.add(nn.Flatten())
            net.add(nn.Dropout(0.6))
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out
    
class Net_denpure(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(Net_denpure, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            net.add(nn.BatchNorm())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out
    


def get_net(ctx):
    num_outputs = 120
    net = Net_den(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net
```

## 训练模型并调参

在[过拟合](../chapter_supervised-learning/underfit-overfit.md)中我们讲过，过度依赖训练数据集的误差来推断测试数据集的误差容易导致过拟合。由于图像分类训练时间可能较长，为了方便，我们这里不再使用K折交叉验证，而是依赖验证集的结果来调参。

我们定义损失函数以便于计算验证集上的损失函数值。我们也定义了模型训练函数，其中的优化算法和参数都是可以调的。

### 用pretrained net转换样本数据

用prenet做一preprocess,以转换样本数据


```{.python .input  n=23}
from mxnet.gluon.model_zoo import vision as models
from mxnet import init

def get_preprocess_net(pretrained_net = models.resnet50_v2(pretrained=True)):
    pretrained_net.collect_params().reset_ctx(ctx)
    prenet = pretrained_net.features
    return prenet


def transform_samples_by_pretrained_net(data, label):
    d = prenet(data.as_in_context(ctx))
    l = label.as_in_context(ctx)
    return d, l
    
    """
    prenet = nn.HybridSequential()
    with prenet.name_scope():
        for i, b in enumerate(pretrained_net.features):
            prenet.add(pretrained_net.features[i])
            if (i == len(pretrained_net.features) - 1):
                break;
    """
    """
    labels, features = None, None
    for data, label in loaded_data:
        feature = prenet(data.as_in_context(ctx))
        #print(feature)
        #print('feature shape ' , feature.shape)
        
        # flattern to 2D
        feature = feature.reshape((feature.shape[0], feature.shape[1] * feature.shape[2] * feature.shape[3]) )
        
        labels = label if labels is None else nd.concat(labels, label, dim = 0)
        features = feature if features is None else nd.concat(features, feature, dim = 0)
        
    return features, labels
    """
        
    
```

```{.json .output n=23}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Model file is not found. Downloading.\nDownloading C:\\Users\\husu/.mxnet/models\\resnet50_v2-eb7a3687.zip from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/resnet50_v2-eb7a3687.zip...\n"
 }
]
```

```{.python .input  n=24}
import datetime
import sys
sys.path.append('..')
import utils

def get_loss(data, net, ctx):
    loss = 0.0
    for feas, label in data:
        # insert to transform samples
        feas, label = transform_samples_by_pretrained_net(feas, label)
        
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
            #insert to transform samples
            data, label = transform_samples_by_pretrained_net(data, label)
            
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

```{.python .input  n=9}
ctx = utils.try_gpu()
num_epochs = 10
learning_rate = 0.01
weight_decay = 5e-4
lr_period = 80
lr_decay = 0.1

prenet = get_preprocess_net()
#prenet.hybridize()

net = get_net(ctx)
net.hybridize()
train(net, train_data, valid_data, num_epochs, learning_rate, 
      weight_decay, ctx, lr_period, lr_decay)
```

```{.json .output n=None}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Train loss: 5.063927, Valid loss 4.676566, Time 00:11:36, lr 0.01\nEpoch 1. Train loss: 4.115519, Valid loss 4.495497, Time 00:12:47, lr 0.01\nEpoch 2. Train loss: 3.459512, Valid loss 4.583109, Time 00:12:56, lr 0.01\nEpoch 3. Train loss: 2.976993, Valid loss 4.241439, Time 00:12:52, lr 0.01\nEpoch 4. Train loss: 2.612582, Valid loss 4.454577, Time 00:12:51, lr 0.01\nEpoch 5. Train loss: 2.381614, Valid loss 4.415094, Time 00:12:30, lr 0.01\nEpoch 6. Train loss: 2.109349, Valid loss 4.370974, Time 00:12:40, lr 0.01\nEpoch 7. Train loss: 1.892061, Valid loss 4.303561, Time 00:12:37, lr 0.01\nEpoch 8. Train loss: 1.746286, Valid loss 4.229083, Time 00:12:37, lr 0.01\nEpoch 9. Train loss: 1.640426, Valid loss 4.299583, Time 00:12:24, lr 0.01\n"
 }
]
```

## 对测试集分类

当得到一组满意的模型设计和参数后，我们使用全部训练数据集（含验证集）重新训练模型，并对测试集分类。

```{.python .input  n=8}
import numpy as np

net = get_net(ctx)
net.hybridize()
train(net, train_valid_data, None, num_epochs, learning_rate, weight_decay, 
      ctx, lr_period, lr_decay)

outputs = []
for data, label in test_data:
    #insert to transform samples
    data, label = transform_samples_by_pretrained_net(data, label)
    
    output = nd.softmax(net(data.as_in_context(ctx)))
    outputs.extend(output.asnumpy())
ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, outputs):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.json .output n=None}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Train loss: 5.180043, Time 00:12:33, lr 0.01\n"
 }
]
```
