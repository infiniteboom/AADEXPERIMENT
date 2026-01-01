这篇文章介绍下自动微分方法在衍生品定价里的应用，大部分内容出自Antoine Savine 的 
**[Modern Computational Finance](https://github.com/asavine/CompFinance)**。

早就听闻AAD方法能够在常数时间内一次性计算出所有的greeks，我一直觉得是什么黑魔法，
最近终于有时间了解一下，发现它是Algorithmic Adjoint Differentiation的缩写......鉴于这个时代已经没有人不懂AI了，所以这应该是非常简单的一篇介绍。

### 背景

我们的目的是寻找一种能够快速计算期权Greeks的方法，也即期权价格对各个要素的偏导数。

以看涨期权为例，根据BS公式，在无套利和常数波动率的条件下，看涨期权的价格可以表示为

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

通常我们会先计算出下面的中间变量，再代入得到最终的结果：

*   折现因子： $DF = \exp(-rT)$
*   远期价格： $F = S_0 \exp[(r - y)T]$
*   到期收益的标准差： $std = \sigma \sqrt{T}$ （注：即总波动率）
*   中间变量：
    $$d = \frac{\log(\frac{F}{K})}{std}, \quad d_1 = d + \frac{std}{2}, \quad d_2 = d - \frac{std}{2}$$

*   $N(d_1)$ 代表在现货测度下期权最终为实值的概率；
*   $N(d_2)$ 代表在风险中性测度下的概率。其中 $N$ 为标准正态分布的累积分布函数。

用代码表示的话，看涨期权的价格是一个纯函数

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

可以看到，计算一次期权的价格的成本是：两次正态分布计算，两次指数运算，一次开方，以及10次乘法和除法运算。在实际应用中，一次期权定价不仅需要返回期权的价格，还需要返回相应的Greeks，即价格对各种输入要素的导数(敏感度)。

例如，我们称Delta是期权价格 ($C$) 相对于标的资产现货价格 ($S_0$) 变动的敏感度，在数学上，它是期权价格对标的资产价格的一阶偏导数：

$$
\Delta = \frac{\partial C}{\partial S_0}
$$

### 1. Finite Difference

不同于上述的看涨期权，一般而言，期权的价格计算常常比较复杂，无法写出显式的表达式。因此在一个通用的定价框架里，我们通常用有限差分，或者说Bumping Method计算偏导数。

回顾导数的定义，导数是下列过程的极限

$$
\frac{f(x + \varepsilon) - f(x)}{\varepsilon}
$$

我们称这种形式为前向差分，遵循如下的过程：给定其他要素，对某个输入增加一个微小的扰动$\varepsilon$，评估函数$f(x+\varepsilon)$，与$f(x)$相减再除以扰动$\varepsilon$，即可得到对应输入的偏导数。

如果将前向差分应用于$S_0$和$\sigma$，则有

$$
\frac{\partial C}{\partial S_0} \approx \frac{C(S_0 + \varepsilon, r, y, \sigma, K, T) - C(S_0, r, y, \sigma, K, T)}{\varepsilon}
$$

$$
\frac{\partial C}{\partial \sigma} \approx \frac{C(S_0, r, y, \sigma + \varepsilon, K, T) - C(S_0, r, y, \sigma, K, T)}{\varepsilon}
$$

注意`BlackScholes`函数有6个输入，在不考虑精度的情况下，用前向差分计算所有的一阶偏导数需要进行额外的6次运算，前向差分的复杂度与入参个数呈线性关系。

### 2. Analytical Differentiation

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

可能会有些出人意料，简单的应用公式法也并不便宜，注意对于每个Greek，我们仍然需要进行一次与`BlackScholes`函数相当的运算，并且复杂度相对于入参个数仍然是线性的。

但是我们可以观察到，这些偏导数公式中有许多公共的部分，如果将这些计算合并，复杂度就会大幅降低。例如在计算$\frac{\partial C}{\partial S_0}$时，我们可以把$N(d_1)$的结果保存下来，这样只需要额外的几次乘法就可以得到$\frac{\partial C}{\partial y}$的结果；同样的，$\frac{\partial C}{\partial K}$和$\frac{\partial C}{\partial K}$ 的表达式都有$N(d_2)$，只需要计算一次。

经过一些观察整理，我们得到了下述更聪明更快的计算方法：

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

BSGreeks bs_call_all_greeks(
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

### 基准测试

我们可以对比一下三种方法的性能，在单线程的情况下：

- CPU：Intel i5-12490
- 调用次数：`N = 1,000,000`

结果如下：

| 方法 | 复杂度| 耗时 (ms) |
| :--- | :--- | :--- |
| **bumping** | $O(N)$ | 220.4 |
| **formulas (naive)** | $O(N)$ | 174.2 |
| **clever formula** | **$O(1)$** | **43.9** |

这里的复杂度是相对于参数个数的复杂度，**clever formula**方法通过复用中间结果，相对简单的公式法提升了四倍左右的性能，相对于bumping提升了5倍左右的性能。

现在重新审视公式法，冗余计算的问题并非偶然，在 BS 模型中，期权价格 $C$ 是远期价格 $F$ 的函数，而 $F$ 是 $S_0$ 和 $y$（以及 $r$ 和 $T$）的函数，根据链式法则：

$$
\frac{\partial C}{\partial S_0} = \frac{\partial C}{\partial F} \frac{\partial F}{\partial S_0}, \quad \frac{\partial C}{\partial y} = \frac{\partial C}{\partial F} \frac{\partial F}{\partial y}
$$

它们的偏导数有着相同的因子 $\frac{\partial C}{\partial F} = N(d_1)$，对自动求导有所了解的朋友应该会意识到，之前我们把公共的表达式提出来的过程，本质上是手动构建了计算图进行微分。

![计算图](./computation graph.png)

上图描述了BS公式中变量的依赖关系和，从上到下分别是入参，中间变量以及最终的结果$C$。为了方便叙述，我们引入伴随微分的概念，我们称最终结果对$x$的导数为$x$的伴随，记作$\overline{x}$，例如，在BS公式中，$C$是最终的结果，则$S$的伴随可以表示为

$$
\overline{S_0} = \frac{\partial C}{\partial S_0} =\frac{\partial C}{\partial F} \frac{\partial F}{\partial S_0} = \frac{\partial F}{\partial S_0} \overline{F} = \frac{F}{S_0} \overline{F}
$$

注意到$F$是$S_0$的函数，但是$\overline{S_0}$是$\overline{F}$的函数，只需知道$\overline{F}$的值就可以求出$\overline{S_0}$的值，至于$\overline{F}$，从图上可以看到它指向了$v$和$d$，根据链式法则，它的梯度是两个的相加

$$
\begin{aligned}
\overline{F} &= \underbrace{\frac{\partial d}{\partial F} \cdot \overline{d}}_{\text{路径1： via } d} + \underbrace{\frac{\partial v}{\partial F} \cdot \overline{v}}_{\text{路径2： via } v} \\
&= \frac{\overline{d}}{F \cdot std} + DF \cdot nd_1 \cdot \overline{v} \\
&= \frac{\overline{d}}{F \cdot std} + DF \cdot nd_1 \cdot 1
\end{aligned}
$$

$d$流向$d_1$和$d_2$两条路径，于是

$$
\begin{cases}
d_1 = d + \frac{std}{2} \\
d_2 = d - \frac{std}{2}
\end{cases}
\Rightarrow \overline{d} = \overline{d_1} + \overline{d_2}
$$

$d_1$和$d_2$分别流向了$N(d_1)$和$N(d_2)$

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

回顾以上过程，为了计算我们依次计算$\overline{S_0}$，我们需要从下往上依次计算 

$$
\overline{v} \longrightarrow \{\overline{nd_1}, \overline{nd_2}\} \longrightarrow \{\overline{d_1}, \overline{d_2}\} \longrightarrow \overline{d} \longrightarrow \overline{F} \longrightarrow \overline{S_0}
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


对其他变量的伴随微分我们可以用完全相同的方式进行，最终得到
![计算图](./computation graph.png)

可以看到计算伴随微分的顺序正好与函数计算的顺序相反，并且每个节点刚好计算一次，这给出了上面合并公共子表达式的标准方案，只需要按照伴随微分的顺序计算即可。利用伴随微分实现啊的代码实现如下

```cpp
#include <cmath>
#include <algorithm>

double normalDens(double x) {
    static const double inv_sqrt_2pi = 0.39894228040143267794;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

BSGreeks C_adjoint(
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

实际上我们可以使用自动微分框架来简化手动提取公共表达式的过程，只需写出前向传播的实现，框架会自动为我们构建计算图，记录各个梯度，下面是用Torch实现的版本。

```python
import torch
import math
from torchviz import make_dot

INV_SQRT_2 = 1.0 / math.sqrt(2.0)

def bs_call_all_greeks_torch(
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

    price = bs_call_price_torch(S0_t, r_t, y_t, sigma_t, K_t, T_t)
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
