# torch.nn

torch.nn定义了神经网络需要的各种结构。如各种layer的functional实现和module实现。

常用的conv, pooling, batchnorm, activation等均在此处定义。

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