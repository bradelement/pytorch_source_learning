# torch.optim

torch.optim主要定义了各种优化算法，sgd, RMSprop, Adam等等。即后向传播更新了grad后怎样更新各层weight。

## optimizer
所有优化器均继承自`Optimizer`
```python
class Optimizer(object):
    def __init__(self, params, defaults):
        # ... 省略

    def step(self, closure):
        raise NotImplementedError
```

各个优化器需要实现自己的`step()`函数。 如sgd.py
```python
class SGD(Optimizer):
    def step(self, closure=None):
        for group in self.param_groups:
            # ... 省略若干行
            for p in group['params']:           
                # ... 省略若干行
                d_p = p.grad.data
                # important, update param here~
                p.data.add_(-group['lr'], d_p)


# 使用
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer.step()

# optimizer第一个参数可以传一个数组，用来实现不同的lr等操作。数组里每一个dict为一个param_group
optim.SGD([
    {'params': model.base.parameters()},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)
```

可以看到，sgd的最终实现即直接减去 learning_rate * grad

`model.parameters()`返回了整个网络所有层的weight，类型为`Variable`
```python
## nn.Parameter
class Parameter(Variable):
    # ...省略
```

当`loss.backward()`完成后向传播时，Parameter中各个层的权重的grad已经计算完毕。

optimizer此处直接根据learning_rate和计算好的梯度直接更新weight即可。

## lr_scheduler
`torch.optim.lr_scheduler`里含有几个简单的修改学习率的类

可以使用`StepLR`或者`ReduceLROnPlateau`等

```python
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)
```