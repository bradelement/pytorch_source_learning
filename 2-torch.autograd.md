# torch.autograd

torch包里的各种Tensor可以看作numpy的pytorch具体实现。

每种tensor只能包含了一种数据类型(如float, double, int, long等)，并有cpu和gpu两种实现。

autograd包则在tensor的基础上，定义了重要的`Variable`类型。封装了自动求导的相关逻辑，简化了后向传播的实现。