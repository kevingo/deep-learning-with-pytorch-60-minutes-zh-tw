# 什麼是 PyTorch？

官方頁面：[https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)

PyTorch 是一個用於科學計算的套件，它的目標族群為：

1. 想要使用 GPU 運算能力的 NumPy 使用者
2. 提供深度學習研究者一個具有彈性及運算速度的套件

## 開始

### Tensors

Tensors 不僅跟 NumPy 的 ndarrays 很像，還可以在 GPU 上進行加速。

```python
from __future__ import print_function
import torch
```

```
* 注意

一個未經過初始化的矩陣被宣告時，不會包含任何已知的值在其中，任何曾經被分配到記憶體的值都會被當成初始化的值。
```

建構一個 5x3 的矩陣, 未初始化:

```python
x = torch.empty(5, 3)
print(x)
```

輸出：

```python
tensor([[6.8947e-31, 4.5761e-41, 6.3417e-31],
        [4.5761e-41, 7.1335e-39, 4.5761e-41],
        [2.0526e+19, 4.5761e-41, 7.1747e-39],
        [4.5761e-41, 7.1748e-39, 4.5761e-41],
        [7.1709e-39, 4.5761e-41, 7.1291e-39]])
```

建構一個隨機初始化的矩陣：

```python
x = torch.rand(5, 3)
print(x)
```

輸出：

```python
tensor([[0.4510, 0.9623, 0.2794],
        [0.8717, 0.5863, 0.8304],
        [0.4561, 0.8626, 0.7028],
        [0.8319, 0.0965, 0.4028],
        [0.9683, 0.3564, 0.8721]])
```

建構一個值皆為零的矩陣，型別為 long：

```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

輸出：

```python
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```

從既有的資料建立 tensor：

```python
x = torch.tensor([5.5, 3])
print(x)
```

輸出：

```python
tensor([5.5000, 3.0000])
```

或是根據一個既有的 tensor 來建立另外一個 tensor，這些方法會重用原本 tensor 的屬性，除非你提供新的值 (例如：dtype)

```python
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)
```
```python
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size
```

輸出：

```python
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[-4.6591e-01, -2.6125e-02,  1.8899e-01],
        [-1.0497e-03,  1.4154e+00,  6.7342e-01],
        [-1.2954e+00,  5.1293e-01,  4.7079e-01],
        [-1.7283e-01, -5.7510e-01, -5.4671e-01],
        [-1.0818e+00, -7.2011e-01, -1.0395e+00]])
```

取得 tensor 的大小：

```python
print(x.size())
```

輸出：

```python
torch.Size([5, 3])
```

注意

`torch.Size` 是一個 tuple，所以它支援任何 tuple 的操作。

### 運算

運算操作有很多語法，在下面的範例中，我們會看看加法操作。

加法：第一種形式

```python
y = torch.rand(5, 3)
print(x + y)
```

輸出：

```python
tensor([[-0.1254,  0.2268,  0.5978],
        [ 0.6447,  2.1564,  1.4111],
        [-0.6157,  1.0004,  1.1581],
        [-0.0860,  0.0310,  0.2498],
        [-0.2172,  0.1825, -0.3939]])
```

加法：第二種形式

```python
print(torch.add(x, y))
```

輸出：

```python
tensor([[-0.1254,  0.2268,  0.5978],
        [ 0.6447,  2.1564,  1.4111],
        [-0.6157,  1.0004,  1.1581],
        [-0.0860,  0.0310,  0.2498],
        [-0.2172,  0.1825, -0.3939]])
```

加法：透過參數提供另外一個輸出的 tensor

```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

輸出：

```python
tensor([[-0.1254,  0.2268,  0.5978],
        [ 0.6447,  2.1564,  1.4111],
        [-0.6157,  1.0004,  1.1581],
        [-0.0860,  0.0310,  0.2498],
        [-0.2172,  0.1825, -0.3939]])
```

加法：原地進行加法

```python
# adds x to y
y.add_(x)
print(y)
```

輸出：

```python
tensor([[-0.1254,  0.2268,  0.5978],
        [ 0.6447,  2.1564,  1.4111],
        [-0.6157,  1.0004,  1.1581],
        [-0.0860,  0.0310,  0.2498],
        [-0.2172,  0.1825, -0.3939]])
```

注意

Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.

You can use standard NumPy-like indexing with all bells and whistles!

```python
print(x[:, 1])
```

輸出：

```python
tensor([-0.0261,  1.4154,  0.5129, -0.5751, -0.7201])
```

Resizing: If you want to resize/reshape tensor, you can use `torch.view`:

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
```

輸出：

```python
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

If you have a one element tensor, use .item() to get the value as a Python number

```python
x = torch.randn(1)
print(x)
print(x.item())
```

輸出：

```python
tensor([-1.0562])
-1.056159257888794
```

Read later:

100+ Tensor operations, including transposing, indexing, slicing, mathematical operations, linear algebra, random numbers, etc., are described here.
NumPy Bridge
Converting a Torch Tensor to a NumPy array and vice versa is a breeze.

The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.

Converting a Torch Tensor to a NumPy Array

```python
a = torch.ones(5)
print(a)
```

輸出：

```python
tensor([1., 1., 1., 1., 1.])
b = a.numpy()
print(b)
```

輸出：

```python
[1. 1. 1. 1. 1.]
```

See how the numpy array changed in value.

```python
a.add_(1)
print(a)
print(b)
```

輸出：

```python
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```

Converting NumPy Array to Torch Tensor
See how changing the np array changed the Torch Tensor automatically

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

輸出：

```python
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

CUDA Tensors
Tensors can be moved onto any device using the .to method.

let us run this cell only if CUDA is available
We will use ``torch.device`` objects to move tensors in and out of GPU

if torch.cuda.is_available():

```python
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
```

輸出：

```python
tensor([-0.0562], device='cuda:0')
tensor([-0.0562], dtype=torch.float64)
```
