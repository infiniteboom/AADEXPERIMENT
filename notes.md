简单介绍下自动微分(Algorithmic Adjoint Differentiation)在衍生品定价里的应用，相较于其他微分方法，AAD的计算复杂度与参数数量无关,能够在常数时间内计算出所有的Greeks，以下大部分内容出自Antoine Savine的**[Modern Computational Finance](https://github.com/asavine/CompFinance)**。

### 背景

我们的目的是寻找一种能够快速计算Greeks，也即期权价格对各个要素的偏导数的方法。

以看涨期权为例，在无套利和常数波动率条件下，由BS公式，看涨期权的价格表示为

$$
C(S_0, r, y, \sigma, K, T) = DF [ F N(d_1) - K N(d_2) ]
$$

其中：
*   $S_0$ ：当前标的资产的价格
*   $r$ ：无风险利率
*   $y$ ：股息率
*   $\sigma$ ：波动率
*   $K$ ：期权的行权价
*   $T$ ：期权的到期时间

计算时会先求出下面的中间变量：

*   折现因子： $DF = \exp(-rT)$
*   远期价格： $F = S_0 \exp[(r - y)T]$
*   到期收益的标准差： $std = \sigma \sqrt{T}$ （注：即总波动率）
*   中间变量：
    $$d = \frac{\log(\frac{F}{K})}{std}, \quad d_1 = d + \frac{std}{2}, \quad d_2 = d - \frac{std}{2}$$

*   $N(d_1)$ 代表在现货测度下期权最终为实值的概率；
*   $N(d_2)$ 代表在风险中性测度下的概率，其中 $N$ 是标准正态分布的累积分布函数。

这个式子可以写成如下的纯函数

```cpp
double BlackScholes(
    const double S0,
    const double r,
    const double y,
    const double sigma,
    const double K,
    const double T)
{
    // 1
    const double df = exp(-r*T);
    // 2
    const double F = S0 * exp((r-y)*T);
    // 3
    const double std = sigma * sqrt(T);
    // 4
    const double d = log(F/K) / std;
    // 5, 6
    const double d1 = d + 0.5 * std, d2 = d1 - std;
    // 7, 8
    const double nd1 = normalCdf(d1), nd2 = normalCdf(d2);
    
    // 9
    const double c = df * (F * nd1 - K * nd2);

    return c;
}
```

根据上式，计算一次看涨期权价格的开销是：两次正态分布运算，两次指数运算，一次开方，十次乘法和除法运算。在实际应用中，一次期权定价不仅需要计算期权的价格，还需要返回价格对各个入参的偏导数。

例如，我们称Delta是期权价格 ($C$) 相对于标的资产现货价格 ($S_0$) 变动的敏感度，在数学上，它是期权价格对标的资产价格的一阶偏导数：

$$
\Delta = \frac{\partial C}{\partial S_0}
$$

计算偏导数有多种方法，一般来说，期权的价格计算比较复杂，没法写出显式的表达式，因此在一个通用的计算框架里，我们常常用有限差分法计算偏导数，在衍生品定价领域也称Bumping Method。

### 1. 有限差分

回顾导数的定义，导数是下列过程的极限

$$
\frac{f(x + \varepsilon) - f(x)}{\varepsilon}
$$

我们称该种形式的差分为前向差分，它遵循以下过程：给定其他要素，对某个变量施加微小的扰动$\varepsilon$，分别对函数$f(x+\varepsilon)$与$f(x)$求值，两式相减再除以扰动项$\varepsilon$，即可得到对应输入的偏导数。

将前向差分应用于$S_0$和$\sigma$，有

$$
\frac{\partial C}{\partial S_0} \approx \frac{C(S_0 + \varepsilon, r, y, \sigma, K, T) - C(S_0, r, y, \sigma, K, T)}{\varepsilon}
$$

$$
\frac{\partial C}{\partial \sigma} \approx \frac{C(S_0, r, y, \sigma + \varepsilon, K, T) - C(S_0, r, y, \sigma, K, T)}{\varepsilon}
$$

对香草看涨期权，`BlackScholes`定价函数有6个输入项，在不考虑精度的情况下，用前向差分法计算所有的一阶偏导数需要进行额外的6次`BlackScholes`计算，因此前向差分的计算复杂度与入参个数呈线性关系。

### 2. 公式法

在一些简单的场景，比如前述的看涨期权，我们可以写出偏导数的闭式解

$$
\begin{aligned}
\frac{\partial C}{\partial S_0} &= \frac{DF \cdot F}{S_0} \cdot N(d_1) \\[8pt]
\frac{\partial C}{\partial r} &= DF \cdot K T \cdot N(d_2) \\[8pt]
\frac{\partial C}{\partial y} &= -DF \cdot F T \cdot N(d_1) \\[8pt]
\frac{\partial C}{\partial \sigma} &= DF \cdot F \cdot n(d_1) \cdot \sqrt{T} \\[8pt]
\frac{\partial C}{\partial K} &= -DF \cdot N(d_2) \\[8pt]
\frac{\partial C}{\partial T} &= DF \cdot F \cdot n(d_1) \cdot \frac{\sigma}{2\sqrt{T}} + r K \cdot DF \cdot N(d_2) - y \cdot DF \cdot F \cdot N(d_1)
\end{aligned}
$$

其中 $n$ 是标准正态分布的概率密度函数：

$$
n(x) = \frac{\partial N(x)}{\partial x} = \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{x^2}{2} \right)
$$

然而，简单的应用公式法也并不便宜，注意对每个偏导数的计算，我们仍然需要一次与`BlackScholes`函数相当的开销，并且偏导数计算的次数相对于入参个数仍然是线性的。

以下我们称该种方法为`Naive Formula Method`，因为不难观察到这些偏导数公式中有许多公共的部分，只要经过合理的编排，将公共的子表达式提取出来，计算的开销就能大幅降低。

例如在计算$\frac{\partial C}{\partial S_0}$时，我们可以把$N(d_1)$的结果保存下来，之后只需额外的几次乘法就可以得到$\frac{\partial C}{\partial y}$；同样的，$N(d_2)$可以在 $\frac{\partial C}{\partial r}$和$\frac{\partial C}{\partial K}$中复用。（这里不太能指望CSE帮助我们）

经过一些努力，我们得到了下面更聪明更快的计算方法`Clever Formula`：

```cpp
struct BSGreeks {
    double price;   // C
    double dS0;     // ∂C/∂S0
    double dr;      // ∂C/∂r
    double dy;      // ∂C/∂y
    double dSigma;  // ∂C/∂σ
    double dK;      // ∂C/∂K
    double dT;      // ∂C/∂T
};

inline double normal_pdf(double x)
{
    static const double inv_sqrt_2pi = 0.39894228040143267794; // 1 / sqrt(2π)
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

inline double normal_cdf(double x)
{
    // Φ(x) = 0.5 * erfc(-x / sqrt(2))
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

BSGreeks BlackFormula(
    double S0,     // spot
    double r,      // risk-free rate
    double y,      // dividend yield
    double sigma,  // volatility
    double K,      // strike
    double T       // maturity
)
{
    BSGreeks g{};

    const double DF     = std::exp(-r * T);            
    const double F      = S0 * std::exp((r - y) * T);  
    const double sqrtT  = std::sqrt(T);
    const double stdDev = sigma * sqrtT;               
    const double d      = std::log(F / K) / stdDev;
    const double d1     = d + 0.5 * stdDev;
    const double d2     = d - 0.5 * stdDev;

    const double N1 = normal_cdf(d1);
    const double N2 = normal_cdf(d2);
    const double n1 = normal_pdf(d1);

    const double DF_F = DF * F;  // DF * F
    const double DF_K = DF * K;  // DF * K

    g.price = DF * (F * N1 - K * N2);

    g.dS0    = DF_F / S0 * N1;                  // ∂C/∂S0
    g.dr     = DF_K * T * N2;                   // ∂C/∂r
    g.dy     = -DF_F * T * N1;                  // ∂C/∂y
    g.dSigma = DF_F * n1 * sqrtT;               // ∂C/∂σ
    g.dK     = -DF * N2;                      // ∂C/∂K
    g.dT     = DF_F * n1 * sigma / (2.0*sqrtT)  // ∂C/∂T
               + r * DF_K * N2
               - y * DF_F * N1;

    return g;
}
```

我们自然会想，对`Clever Formula Method`来说，计算复杂度是什么。要回答这个问题，我们必须弄明白我们是如何提取公共子表达式的。注意到在 BS 模型中，期权价格 $C$ 是远期价格 $F$ 的函数，而 $F$ 是 $S_0$ 和 $y$（以及 $r$ 和 $T$）的函数，根据链式法则：

$$
\frac{\partial C}{\partial S_0} = \frac{\partial C}{\partial F} \frac{\partial F}{\partial S_0}, \quad \frac{\partial C}{\partial y} = \frac{\partial C}{\partial F} \frac{\partial F}{\partial y}
$$

在`Clever Formula Method`中，我们看到$\frac{\partial C}{\partial S_0}$和$\frac{\partial C}{\partial y}$的公共子表达式$N(d_1)$，正是$\frac{\partial C}{\partial F}$。

所有变量的依赖关系如下图所示，它可以启发我们如何提取各个表达式中的公共子表达式。

![计算图](picture/computation_graph.png)

由上图我们可以看到$S_0$如何影响最终结果$C$，它先参与了$F$的计算，$F$决定了$d$和 $v$的数值，$d$进入$d_1,d_2$，$d_1,d_2$变换到$nd_1,nd_2$，$nd_1,nd_2$传导到$v$，最后是一个恒等映射$C = v$。

$$S_0 \longrightarrow F \longrightarrow \{d, v\} \longrightarrow \{d_1, d_2\} \longrightarrow \{nd_1, nd_2\} \longrightarrow v \xrightarrow{\text{恒等映射}} C$$

为方便描述，我们定义最终结果$C$对变量$x$的导数为$x$的伴随，记作$\overline{x}$。

例如，$S_0$的伴随可以表示为

$$
\overline{S_0} = \frac{\partial C}{\partial S_0} =\frac{\partial C}{\partial F} \frac{\partial F}{\partial S_0} = \frac{\partial F}{\partial S_0} \overline{F} = \frac{F}{S_0} \overline{F}
$$

注意$F$是$S_0$的函数，而用伴随时，$\overline{S_0}$是$\overline{F}$的函数，这是一种普适的关系。

接下来，注意到$F$指向了$d$和$v$，根据链式法则，它的伴随是两条路径的梯度之和。

$$
\begin{aligned}
\overline{F} &= \underbrace{\frac{\partial d}{\partial F} \cdot \overline{d}}_{\text{路径1： via } d} + \underbrace{\frac{\partial v}{\partial F} \cdot \overline{v}}_{\text{路径2： via } v} \\
&= \frac{\overline{d}}{F \cdot std} + DF \cdot nd_1 \cdot \overline{v} \\
&= \frac{\overline{d}}{F \cdot std} + DF \cdot nd_1 \cdot 1
\end{aligned}
$$

$d$分别流向$d_1$和$d_2$，于是

$$
\begin{cases}
d_1 = d + \frac{std}{2} \\
d_2 = d - \frac{std}{2}
\end{cases}
\Rightarrow \overline{d} = \overline{d_1} + \overline{d_2}
$$

$d_1$和$d_2$分别指向$N(d_1)$和$N(d_2)$

$$
\begin{aligned}
nd_1 = N(d_1) &\Rightarrow \overline{d_1} = n(d_1)\overline{nd_1} \\
nd_2 = N(d_2) &\Rightarrow \overline{d_2} = n(d_2)\overline{nd_2}
\end{aligned}
$$

最后我们有

$$
v = DF[F nd_1 - K nd_2] \Rightarrow
\begin{cases}
\overline{nd_1} = DF \cdot F \cdot \overline{v} = DF \cdot F \\
\overline{nd_2} = -DF \cdot K \cdot \overline{v} = -DF \cdot K
\end{cases}
$$

总结如下

$$
\begin{aligned}
\overline{S_0} &= \frac{F}{S_0} \overline{F} \\
&= \frac{F}{S_0} \left( \frac{\overline{d}}{F \cdot std} + DF \cdot nd_1 \right) \\
&= \frac{F}{S_0} \left( \frac{\overline{d_1} + \overline{d_2}}{F \cdot std} + DF \cdot nd_1 \right) \\
&= \frac{F}{S_0} \left( \frac{n(d_1)\overline{nd_1} + n(d_2)\overline{nd_2}}{F \cdot std} + DF \cdot nd_1 \right) \\
&= \frac{F}{S_0} \left( \frac{n(d_1)DF \cdot F - n(d_2)DF \cdot K}{F \cdot std} + DF \cdot nd_1 \right) \\
&= DF \frac{F}{S_0} \left[ N(d_1) + \frac{n(d_1) - \frac{K}{F}n(d_2)}{std} \right] \\
&= \frac{DF \cdot F}{S_0} N(d_1)
\end{aligned}
$$

这里的关键在于，在计算时我们并不直接使用公式$\overline{S_0} = \frac{DF \cdot F}{S_0} N(d_1)$计算偏导数，而遵循如下的顺序

$$
\overline{v} \longrightarrow \{\overline{nd_1}, \overline{nd_2}\} \longrightarrow \{\overline{d_1}, \overline{d_2}\} \longrightarrow \overline{d} \longrightarrow \overline{F} \longrightarrow \overline{S_0}
$$

也即从最终结果的伴随开始，沿着与函数计算相反的顺序进行伴随的求解，最终正好每个变量被计算了一次，这给出了我们合并子表达式的标准过程。

对其他变量的进行相同的过程，最终得到
![计算图](picture/backpropagate_cg.png)

下面是利用伴随微分方法求解Greeks的代码。

```cpp
#include <cmath>
#include <algorithm>

double normalDens(double x) {
    static const double inv_sqrt_2pi = 0.39894228040143267794;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

BSGreeks BlackAdjoint(
    const double S0,
    const double r,
    const double y,
    const double sig,
    const double K,
    const double T)
{
    const double C_ = 1.0;

    // --- Forward Pass (Evaluation) ---

    const double sqrtT = std::sqrt(T);

    const double df  = std::exp(-r * T);
    const double F   = S0 * std::exp((r - y) * T);
    const double std = sig * sqrtT;
    const double d   = std::log(F / K) / std;
    const double d1  = d + 0.5 * std;
    const double d2  = d - 0.5 * std;

    const double nd1 = 0.5 * std::erfc(-d1 / std::sqrt(2.0));
    const double nd2 = 0.5 * std::erfc(-d2 / std::sqrt(2.0));

    const double v = df * (F * nd1 - K * nd2);

    // --- Reverse Pass (Adjoint Calculation) ---

    double S0_  = 0.0;
    double r_   = 0.0;
    double y_   = 0.0;
    double sig_ = 0.0;
    double K_   = 0.0;
    double T_   = 0.0;

    double df_    = C_ * (F * nd1 - K * nd2);
    double F_     = C_ * df * nd1;
    double K_loc_ = C_ * df * (-nd2);
    double nd1_   = C_ * df * F;
    double nd2_   = C_ * df * (-K);

    double d1_ = nd1_ * normalDens(d1);
    double d2_ = nd2_ * normalDens(d2);

    double d_   = d1_ + d2_;
    double std_ = 0.5 * d1_ - 0.5 * d2_;

    F_     += d_ / (F * std);
    K_loc_ -= d_ / (K * std);
    std_   -= d_ * d / std;

    K_ += K_loc_;

    sig_ += std_ * sqrtT;
    T_   += std_ * sig / (2.0 * sqrtT);

    const double dF_dS0 = F / S0;
    const double dF_dr  = T * F;
    const double dF_dy  = -T * F;
    const double dF_dT  = (r - y) * F;

    S0_ += F_ * dF_dS0;
    r_  += F_ * dF_dr;
    y_  += F_ * dF_dy;
    T_  += F_ * dF_dT;

    const double ddf_dr = -T * df;
    const double ddf_dT = -r * df;

    r_ += df_ * ddf_dr;
    T_ += df_ * ddf_dT;

    BSGreeks g{};
    g.price  = v;
    g.dS0    = S0_;
    g.dr     = r_;
    g.dy     = y_;
    g.dSigma = sig_;
    g.dK     = K_;
    g.dT     = T_;

    return g;
}
```
### 自动微分

在之前的求解过程中，我们手动引入了诸如 $DF, F, std $等中间变量，从更一般的视角来看，Black-Scholes 公式本质上由加减乘除和正态分布等基本函数组合而成，将这些更基本的运算视为节点，我们会得到一张更细粒度、包含大量中间变量的计算图。随着模型复杂度的增加，手动实现伴随微分变得几乎不可行。此时需要借助自动微分框架：只需编写前向传播的代码，框架会自动构建计算图并跟踪梯度。下面是Pytorch的版本。

```python
import torch
import math
from torchviz import make_dot

INV_SQRT_2 = 1.0 / math.sqrt(2.0)

def BlackTorch(
      S0: float,
      r: float,
      y: float,
      sigma: float,
      K: float,
      T: float,
  ):
    dtype = torch.float64

    S0_t    = torch.tensor(S0,    dtype=dtype, requires_grad=True)
    r_t     = torch.tensor(r,     dtype=dtype, requires_grad=True)
    y_t     = torch.tensor(y,     dtype=dtype, requires_grad=True)
    sigma_t = torch.tensor(sigma, dtype=dtype, requires_grad=True)
    K_t     = torch.tensor(K,     dtype=dtype, requires_grad=True)
    T_t     = torch.tensor(T,     dtype=dtype, requires_grad=True)

    DF      = torch.exp(-r_t * T_t)
    F       = S0_t * torch.exp((r_t - y_t) * T_t)
    sqrtT   = torch.sqrt(T_t)
    std_dev = sigma_t * sqrtT

    d  = torch.log(F / K_t) / std_dev
    d1 = d + 0.5 * std_dev
    d2 = d - 0.5 * std_dev

    N1 = 0.5 * torch.erfc(-d1 * INV_SQRT_2)
    N2 = 0.5 * torch.erfc(-d2 * INV_SQRT_2)

    price = DF * (F * N1 - K_t * N2)
    price.backward()

    return {
        "price":  float(price.item()),
        "dS0":    float(S0_t.grad.item()),    # ∂C/∂S0 
        "dr":     float(r_t.grad.item()),     # ∂C/∂r   
        "dy":     float(y_t.grad.item()),     # ∂C/∂y
        "dSigma": float(sigma_t.grad.item()), # ∂C/∂σ   
        "dK":     float(K_t.grad.item()),     # ∂C/∂K
        "dT":     float(T_t.grad.item()),     # ∂C/∂T   
    }
```
Torch在背后为我们构建了下面的计算图，它完全对应着BS公式，以最左边的一条线为例，`r` 和 `T` 进入 `MulBackward0` ($r \times T$) $\rightarrow$ `NegBackward0` ($-rT$) $\rightarrow$ `ExpBackward0` ($e^{-rT}$)，其他的诸如此类。

![计算图](picture/bs_computation_graph.png)

### 性能测试

在正式讨论复杂度之前，我们先来进行性能测试，在同一组参数下进行了 1,000,000 次定价计算。测试环境使用单线程 `-O3` 编译优化，参数如下：
> $S_0 = 100, r = 1\%, y = 0, \sigma = 20\%, K = 100, T = 1$

测试结果汇总如下表所示：

| 方法名称 | 耗时 (1M次) | 相对前向计算倍数 |
| :--- | :--- | :--- |
| **Finite Difference** (有限差分) | ~ 222.8 ms | ≈ 10.5 × |
| **Naive Formulas** (朴素公式法) | ~ 188.2 ms | ≈ 8.9 × |
| **Adjoint (AAD)** (自动微分/伴随) | ~ 54.5 ms | ≈ 2.6 × |
| **Optimized Formulas** (优化公式法) | ~ 48.1 ms | ≈ 2.3 × |

#### 结果分析与小结

1.  **AAD vs. 有限差分**：
    AAD 的优势是压倒性的。有限差分每增加一个 Greek 就需要多运行一次完整的定价函数；而 AAD 通过一次反向传播即可获得所有参数的梯度。在本例中只有 6 个参数，差距已达 4 倍以上；如果在包含数百个风险因子的复杂模型（如 xVA 或 Local Volatility Calibration）中，这一差距将扩大到数百倍。

2.  **为什么 AAD 比“优化公式法”略慢？**
    你可能会注意到，通用性更强的 Adjoint (50.3 ms) 略慢于手动优化的公式法 (43.9 ms)。这是完全合理的：
    *   **代数化简 vs. 机械求导**：优化公式法利用了 BS 模型的数学特性（如 Gamma 的解析解极其简洁）进行了代数层面的化简。
    *   **计算开销**：AAD 是“机械地”对每一步运算应用链式法则。虽然它避免了重复计算，但在反向累加梯度时，仍然包含了一些加法和乘法操作，而这些操作在解析解的最终形态中可能已经被数学消去了。

### 复杂度分析
为方便分析，我们限定在$\odot \in \{+, \cdot, /\}$ 的域$K$上讨论，设 $F(x_1, ..., x_n)$ 是一个变量为 $x_1, ..., x_n \in K$ 的有理函数

**定义** 我们称计算 $F$ 的运算序列 $\{g_1, ..., g_s\}$ 是一个**直线程序（straight-line programs，简称SLP）**，如果每个 $g_i$ 都属于以下形式之一：  
(1) $g_i \leftarrow g_j \odot g_k$，其中 $j, k < i$；  
(2) $g_i \leftarrow g_j \odot c$，其中 $j < i$ 且 $c \in K$；  
(3) $g_i \leftarrow x_k \odot x_l$；  
(4) $g_i \leftarrow x_k \odot c$，其中 $c \in K$；或  
(5) $g_i \leftarrow g_j \odot x_k$，其中 $j < i$。  

规定运算序列的最后一项$g_s$ 给出 $F(x_1, ..., x_n)$的结果。

举例来说，我们希望求值 $F(x_1, x_2, x_3) = 5 - x_1 x_2^2 / x_3$，它可由以下运算序列计算：

$$
\begin{aligned}
g_1 &\leftarrow x_2 \cdot x_2 \\
g_2 &\leftarrow x_1 \cdot g_1 \\
g_3 &\leftarrow g_2 / x_3 \\
g_4 &\leftarrow -1 \cdot g_3 \\
g_5 &\leftarrow 5 + g_4
\end{aligned}
$$

那么上述序列就是一个计算 $F$ 的 SLP，显然 SLP 并不唯一，我们称 $T(F)$是计算$F$的最小操作数，即

$$
T(F) = \min_{\Gamma} \{ s \mid \Gamma = (g_1, \dots, g_s) \text{ computes } F \}
$$

我们已经知道前向差分法（Forward Differentiation）的复杂度与入参的个数$x_1,x_2,...x_n$成正比，它的算数复杂度可以表示为

$$
T_{\text{fwd}}(\nabla F(x_1, \dots, x_n)) \in O\left( n \cdot T(F(x_1, \dots, x_n)) \right)
$$

下面我们将证明，应用反向传播（Reverse Mode）计算导数时，其计算复杂度与输入维度 $n$ 无关。事实上，Baur-Strassen 定理给出了如下的复杂度界限：

**Theorem (Baur-Strassen):**

$$
T(\nabla F(x_1, \dots, x_n)) \leq 6 \cdot T(F)
$$

其中 $T(F)$ 表示计算函数 $F$ 的最小算术运算次数。

证明并不复杂，但也没什么必要，如果你相信我的话可以略过...

证明:


假设 $\{g_1, \dots, g_s\}$ 是计算 $F(x_1, \dots, x_n)$ 的最小长度 SLP，定义 $F^{(i)}$ 为由 $g_i, \dots, g_s$ 计算的函数。

以之前的 SLP 为例
$$
\begin{aligned}
g_1 &\leftarrow x_2 \cdot x_2 \\
g_2 &\leftarrow x_1 \cdot g_1 \\
g_3 &\leftarrow g_2 / x_3 \\
g_4 &\leftarrow -1 \cdot g_3 \\
g_5 &\leftarrow 5 + g_4
\end{aligned}
$$

对应着函数

$$ F(x_1, x_2, x_3) = 5 - \frac{x_1 x_2^2}{x_3} $$

$F^{(i)}$ 的定义可以理解为：前 $i-1$ 步已经计算完成，由剩下的步骤所组成的函数。

当$i=1$，运算还未开始，
$$F^{(1)}(x_1,x_2,x_3) = F = 5 - \frac{x_1 x_2^2}{x_3}$$

当$i=3$，$g_1$ ($x_2^2$) 和 $g_2$ ($x_1 g_1$) 已经计算完成，他们的结果变为已知量，则输入变量变为$\{x_1, x_2, x_3, \mathbf{g_1}, \mathbf{g_2}\}$，剩余的步骤是$\{g_3, g_4, g_5\}$

$$ F^{(3)}(x_1, x_2, x_3, g_1, g_2) = 5 - \frac{g_2}{x_3} $$

当$i=5$，运算序列只差最后一步加法
$$
F^{(5)}(x_1, x_2, x_3, g_1, g_2, g_3, g_4) = 5 + g_4
$$

当$i=6$，所有的计算已经完成，$F^{(6)}$是恒等映射：
$$
F^{(6)}(x_1, x_2, x_3, g_1, g_2, g_3, g_4, g_5) = g_5
$$

一般的，$F^{(i)}$的变量集合为 $\{x_1, \dots, x_n, g_1, \dots, g_{i-1}\}$，为了便于说明，我们将变量重命名为 $\{z_1, \dots, z_{n+s}\}$，当 $j \leq n$ 时 $z_j = x_j$，当 $j > n$ 时 $z_j = g_{j-n}$。

证明的思路是归纳法,假设 $F^{(i+1)}$ 的导数已经计算完成，成本是至多 $5(s - i)$ 次运算，需要证明至多 $5(s - i + 1)$ 次运算可以计算出 $F^{(i)}$ 的所有导数。

**Base case：** $i = s$。  

当$i = s$， $F^{(i+1)} = F^{(s+1)}(x_1, \dots, x_n, g_1, \dots, g_s) = g_s$是恒等映射，于是，

$$
\frac{\partial F^{(s+1)}}{\partial z_j} =
\begin{cases}
1, & \text{if } z_j = g_s \\
0, & \text{if } z_j \neq g_s
\end{cases}
$$。这需要 0 次运算。

**Induction：**  
假设我们用至多 $5(s - i)$ 次运算计算出了所有的$\frac{\partial F^{(i+1)}}{\partial z_j}$，需要证明只需额外的 5 次运算即可计算出所有的 $\frac{\partial F^{(i)}}{\partial z_j}$。

注意到$F^{(i)}$ 可通过$F^{(i+1)}$和$g_i$计算得到，具体来说，假设 $g_i = s_b \odot s_\ell$，则：

$$
F^{(i)}(z_1, \dots, z_{n+i-1}) = F^{(i+1)}(z_1, \dots, z_{n+i-1}, \underbrace{s_b \odot s_\ell}_{g_i})
$$

应用链式法则：
$$
\frac{\partial F^{(i)}}{\partial z_j} = \frac{\partial F^{(i+1)}}{\partial z_j} + \frac{\partial F^{(i+1)}}{\partial g_i} \cdot \frac{\partial g_i}{\partial z_j}
$$

由于 $g_i$ 的定义仅涉及 $s_b$ 和 $s_\ell$，我们只需更新与操作数 $s_b$ 和 $s_\ell$ 相关的导数项：

$$
\frac{\partial F^{(i)}}{\partial s_k} = \frac{\partial F^{(i+1)}}{\partial s_k} + \frac{\partial F^{(i+1)}}{\partial g_i} \cdot \frac{\partial g_i}{\partial s_k}, \quad \text{for } k \in \{b, \ell\}
$$

只需逐个讨论即可

**case 1：** $g_i = z_t + z_{t'}$  
此时
$$
\begin{aligned}
\frac{\partial F^{(i)}}{\partial z_t} &= \frac{\partial F^{(i+1)}}{\partial z_t} + \frac{\partial F^{(i+1)}}{\partial g_i} \\
\frac{\partial F^{(i)}}{\partial z_{t'}} &= \frac{\partial F^{(i+1)}}{\partial z_{t'}} + \frac{\partial F^{(i+1)}}{\partial g_i}
\end{aligned}
$$
注意我们假设$F^{(i+1)}$的导数都已计算完成，因此这两个更新只需 1 次加法运算，总共需要 **2 次运算**。

**case 2：** $g_i = z_t \cdot z_{t'}$

$$
\begin{aligned}
\frac{\partial F^{(i)}}{\partial z_t} &= \frac{\partial F^{(i+1)}}{\partial z_t} + \frac{\partial F^{(i+1)}}{\partial g_i} \cdot z_{t'} \\
\frac{\partial F^{(i)}}{\partial z_{t'}} &= \frac{\partial F^{(i+1)}}{\partial z_{t'}} + \frac{\partial F^{(i+1)}}{\partial g_i} \cdot z_t
\end{aligned}
$$
每个梯度的更新需要 2 次运算（1 次乘法，1 次加法），总计 **4 次运算**。

**case 3：** $g_i = z_t / z_{t'}$
$$
\begin{aligned}
\frac{\partial F^{(i)}}{\partial z_t} &= \frac{\partial F^{(i+1)}}{\partial z_t} + \frac{\partial F^{(i+1)}}{\partial g_i} / z_{t'} \\
\frac{\partial F^{(i)}}{\partial z_{t'}} &= \frac{\partial F^{(i+1)}}{\partial z_{t'}} + \frac{\partial F^{(i+1)}}{\partial g_i} \cdot \left( \frac{-z_t}{z_{t'}^2} \right)
\end{aligned}
$$

这是最为复杂的情况，我们可以通过如下步骤更新梯度

1. $a \leftarrow \frac{\partial F^{(i+1)}}{\partial g_i} / z_{t'}$
2. $b \leftarrow a \cdot (-1)$
3. $c \leftarrow b \cdot g_i$
4. $\frac{\partial F^{(i)}}{\partial z_t} + a$
5. $\frac{\partial F^{(i)}}{\partial z_{t'}} + c$

总计**5 次运算**。
这三种情况也覆盖了 $s_b$ 或 $s_\ell$ 是标量的情况，证明就此完成。


