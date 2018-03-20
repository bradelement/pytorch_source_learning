# torch.nn

torch.nn定义了神经网络需要的各种结构。如各种layer的functional实现和module实现。

常用的conv, pooling, batchnorm, activation等均在此处定义。

## modules
所有layer均继承自`nn.Module`。该模块定型了整个神经网络前向传播的流程。
其他具体实现只需关注前向传播的具体运算即可。

```python
### nn.Module 的 __call__ 函数， 有删减
def __call__(self, *input, **kwargs):
    #... 省略了一些 _forward_pre_hooks 函数流程

    # 真正的前向传播
    result = self.forward(*input, **kwargs)
    
    #... 省略了一些 _forward_hooks 函数流程 
    #... 省略了一些 _backward_hooks 函数流程
    return result
```

如常用的`Conv2d`，定义如下：
```python
class _ConvNd(Module):
    # _ConvNd定义了卷积层共有的一些函数，继承自nn.Module

class Conv2d(_ConvNd):
    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
```

常用的`MaxPool2d`，定义如下：
```python
class MaxPool2d(Module):
    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride,
            self.padding, self.dilation, self.ceil_mode,
            self.return_indices)
```

可以看出这些具体的层只是简单的继承了`nn.Module`，重写了`forward`方法，调用对应的functional实现即可。


## functional
真正完成各层前向传播功能的函数，定义在nn下面的functional.py

可以看到上述的`F.conv2d`, `F.max_pool2d`
```python
_ConvNd = torch._C._functions.ConvNd

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    
    f = _ConvNd(_pair(stride), _pair(padding), _pair(dilation), False,
                _pair(0), groups, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)

    return f(input, weight, bias)


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):

    ret = torch._C._nn.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    return ret if return_indices else ret[0]
```

最终其实调用了_C.so中的各个函数。

## train && eval
一般训练一个神经网络的时候，需要区分train_net和validate_net。

caffe里一般对应两个不同的prototxt。 网络结构一般基本一样，某些层如batchNormalization, dropout等行为会有不同。

pytorch的`nn.Module`提供了两个函数`train`和`eval`，可以无缝的切换两种模式。
```python
# nn.Module
class Module(object):
    def __init__(self):
        self.training = True
        
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)


# for example
net = xxxNet()
net.train()
# do some training
net.eval()
# do some validation
```

可以看到batch norm层在前向传播的时候会判断`self.training`做不同的操作。

因为最终的BatchNorm操作是在c模块中完成的，我们此处只能看到函数把`self.training`作为参数进行了传递。

使用方法可以参考batchnorm.py中`BatchNorm2d`的注释。
> During training, this layer keeps a running estimate of its computed mean 
and variance. The running sum is kept with a default momentum of 0.1.

>During evaluation, this running mean/variance is used for normalization.

```python
class _BatchNorm(Module):
    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)

class BatchNorm2d(_BatchNorm):
    #... 
    pass


# functional.py
def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    if training:
        size = list(input.size())
        if reduce(mul, size[2:], size[0]) == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
            
    f = torch._C._functions.BatchNorm(running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled)
    return f(input, weight, bias)
```

## Serialization semantics
```python
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))
```

## DataParallel layers
```python
net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
output = net(input_var)
```