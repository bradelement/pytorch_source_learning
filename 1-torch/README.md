# torch

torch内文件夹较多，先从`__init__.py`看起

开头判断了一下动态库的加载方式等等，之后加载了`torch._C`模块。

看起来是pytorch的C模块，以动态库的形式对python暴露接口。源文件为同级目录下的_C.so

```python
from torch import _C
dir(_C)
# 可以看到有很多常用函数和类。包含各种tensor的基类，数学运算等。
```

之后定义了若干python中常用的storage和tensor类，包装了与之对应的c接口
```python
class FloatTensor(_C.FloatTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return FloatStorage
```

之后import了其他重要模块，autograd，nn，optim等等。