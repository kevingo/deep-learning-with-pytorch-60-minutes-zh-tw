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

調整大小：如果你想要 tensor 的形狀或大小，你可以使用 `torch.view`：

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

如果你的 tensor 只有一個元素，可以使用 `.item()` 來取得它的值：

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

稍後閱讀：

超過 100 個以上關於 tensor 的操作，包含了轉置、索引、切片、數學操作、線性代數、隨機數等，可以參考[這份文件](https://pytorch.org/docs/torch)。

## NumPy 的橋樑

將 torch 的 tensor 專換為 numpy 陣列是輕而易舉的，反之亦然。

torch tensor 和 numpy 陣列會共享記憶體位置 (當 torch tensor 是在 CPU 時)，當你改變其中一個內容，另一個也會被改變。

### 將 torch tensor 轉換為 numpy 陣列

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

查看 numpy 陣列的值是如何變化的。

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

## 轉換 numpy array 到 torch tensor

來看看修改 numpy 陣列是如何自動改變 torch tensor：

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

除了 CharTensor 之外，所有運行在 CPU 的 tensor 都支援轉換成 numpy 陣列

## CUDA Tensors

Tensor 可以透過 `.to` 方法移動到任何的計算裝置上。

```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
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
