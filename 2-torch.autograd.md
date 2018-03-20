# torch.autograd

torch包里的各种Tensor可以看作numpy的pytorch具体实现。可以理解为一个多维矩阵。

每种tensor只能包含了一种数据类型(如float, double, int, long等)，并有cpu和cuda两种版本。

autograd包则在tensor的基础上，定义了重要的`Variable`类型。封装了自动求导的相关逻辑，简化了后向传播的实现。

相关逻辑封装在`torch/autograd/variable.py`

每一个`Variable`包含了`data`, `grad`, `grad_fn`等成员。

`data`即为封装的tensor，参与前向传播。

当`requires_grad == True`时，`grad_fn`记录了computational graph中的梯度函数，`grad`记录计算出来的梯度

```python
class Variable(_C._VariableBase):
    #后向传播
    def backward(self, gradient=None, retain_graph=None, 
        create_graph=None, retain_variables=None):

        torch.autograd.backward(self, gradient, retain_graph, create_graph, retain_variables)  
```

可以看到`Variable`定义了`backward()`来完成后向传播。