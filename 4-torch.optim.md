# torch.optim

torch.optim主要定义了各种优化算法，sgd, RMSprop, Adam等等。即后向传播更新了grad后怎样更新各层weight。

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
```

可以看到，sgd的最终实现即直接减去 learning_rate * grad

model.parameters()即整个网络所有层的weight， 类型为`Variable`
```python
## nn.Parameter
class Parameter(Variable):
    # ...省略
```

当`loss.backward()`完成后向传播时，Parameter中各个层的权重的grad已经计算完毕。

optimizer此处直接根据learning_rate和计算好的梯度直接更新weight即可。