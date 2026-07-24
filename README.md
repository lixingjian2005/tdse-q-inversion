# v1.0 Q-Inversion: 从 P(k,τ) 反演量子光场 Q 分布

## 项目目标

从 attosecond streaking 实验（或 TDSE 模拟）测得的光电子动量分布 $P(k, \tau)$ 反演出驱动 NIR 脉冲的 Husimi-Q 分布 $Q(\alpha)$。

## 正向模型（v5.1-release）

v5.1-release 实现了正向流程：

$$
P(k, \tau) = \sum_{i} Q_i \cdot |A_{\alpha_i}(k, \tau)|^2
$$

其中：
- $\alpha_i$ 为 Husimi-Q 函数采样得到的相干态振幅
- $Q_i$ 为对应采样点的 Q 函数权重，$\sum Q_i = 1$
- $A_{\alpha_i}(k, \tau)$ 为 TDSE 计算出的光电离跃迁振幅

## 逆问题

给定多个延迟 $\tau_j$ 和动量 $k_m$ 处的测量值 $P^{\text{meas}}(k_m, \tau_j)$，求解 $Q(\alpha)$ 满足：

$$
P(k, \tau) = \int Q(\alpha) \cdot |A_\alpha(k, \tau)|^2 \, d^2\alpha
$$

这是一个 **Fredholm 第一类积分方程**，属于不适定（ill-posed）逆问题，需要正则化。

## 候选方法

1. **最大似然估计（MLE）** + 平滑先验
2. **Tikhonov 正则化**
3. **最大熵方法（MEM）**

## 目录结构

```
v1.0-q-inversion/
├── README.md                 # 本文件
├── docs/                     # 理论推导文档
├── src/                      # 反演核心代码（Fortran/Python）
├── tools/                    # Python 工具脚本
├── examples/                 # 示例配置和测试数据
└── tests/                    # 单元测试
```

## 与 v5.1-release 的关系

本子项目与 [v5.1-release](../v5.1-release/) 平行，可引用其：
- TDSE 正向求解器（生成 $A_\alpha(k, \tau)$）
- Husimi-Q 采样工具（`tools/sample_husimi_qfunc.py`）
- 分析工具（`tools/analyze_tdse.py`）

## 状态

**项目初始化阶段** — 目录结构已建立，待实现。
