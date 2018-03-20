# torch.cuda

torch.cuda主要定义的是各种cuda类型的tensor和相关操作。

平时使用的`torch.cuda.FloatTensor`等gpu类型的tensor都在此定义。
```python
class FloatTensor(_CudaBase, torch._C.CudaFloatTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return FloatStorage
```

可以看到这里的FloatTensor和torch.FloatTensor定义非常类似，简单封装了`_C.so`里对应的cuda数据类型。

### 常用函数
```python
# 判断cuda是否可用
torch.cuda.is_available()
```