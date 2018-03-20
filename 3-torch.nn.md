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