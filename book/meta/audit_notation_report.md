# 符号/记法一致性审计报告（初版）

> 用法：先看『章内混用』（阅读最受伤），再定『全书统一目标』写入 `book/meta/notation.md`，
> Phase 2 逐章把少数派改为主流记法。注意：正则统计有噪声，改前需人工确认上下文。

## 一、章内混用（同一章出现 ≥2 种互斥记法）

### ε（噪声）

- ch01: `\\varepsilon`×36, `\\epsilon`×2
- ch02: `\\varepsilon`×26, `\\epsilon`×17
- ch03: `\\varepsilon`×11, `\\epsilon`×4
- ch04: `\\epsilon`×21, `\\varepsilon`×15
- ch06: `\\varepsilon`×21, `\\epsilon`×1
- ch07: `\\epsilon`×26, `\\varepsilon`×20
- ch10: `\\varepsilon`×62, `\\epsilon`×7
- ch14: `\\varepsilon`×55, `\\epsilon`×14
- ch18: `\\epsilon`×18, `\\varepsilon`×9

### ᾱ_t（累积信号保留率）

- ch10: `\\bar{\\alpha}`×136, `\\bar\\alpha`×15
- ch11: `\\bar\\alpha`×117, `\\bar{\\alpha}`×2, `\\barα（Unicode）`×2
- ch12: `\\bar\\alpha`×214, `\\barα（Unicode）`×27
- ch13: `\\barα（Unicode）`×29, `\\bar\\alpha`×25

### x 的粗体

- ch00绪论: `纯 x`×58, `\\mathbf{x}`×38
- ch01: `纯 x`×445, `\\mathbf{x}`×55
- ch10: `纯 x`×352, `\\mathbf{x}`×310
- ch16: `纯 x`×161, `\\mathbf{x}`×8

### ε_θ 的 hat

- ch12: `\\epsilon_\\theta（无hat）`×82, `\\hat\\epsilon_\\theta`×17
- ch15: `\\epsilon_\\theta（无hat）`×4, `\\hat{\\epsilon}_\\theta`×1

### 高斯记号

- ch03: `\\mathcal{N}`×15, `N(0,I)（裸N）`×3
- ch04: `\\mathcal{N}`×42, `N(0,I)（裸N）`×30
- ch05: `\\mathcal{N}`×48, `N(0,I)（裸N）`×5
- ch07: `\\mathcal{N}`×71, `N(0,I)（裸N）`×1
- ch08: `\\mathcal{N}`×81, `N(0,I)（裸N）`×2
- ch09: `\\mathcal{N}`×64, `N(0,I)（裸N）`×6
- ch10: `\\mathcal{N}`×49, `N(0,I)（裸N）`×7
- ch11: `\\mathcal{N}`×37, `N(0,I)（裸N）`×2
- ch12: `\\mathcal{N}`×16, `N(0,I)（裸N）`×3
- ch13: `\\mathcal{N}`×21, `N(0,I)（裸N）`×21
- ch14: `\\mathcal{N}`×13, `N(0,I)（裸N）`×2
- ch18: `\\mathcal{N}`×8, `N(0,I)（裸N）`×1

### KL 散度

- ch11: `D_\\text{KL}`×12, `\\mathrm{KL}（无D前缀）`×1

### 期望算子

- ch01: `\\mathbb{E}`×4, `E[（裸E）`×1, `\\mathbb E（无括号）`×1
- ch02: `\\mathbb{E}`×41, `E[（裸E）`×4
- ch03: `\\mathbb{E}`×9, `\\mathbb E（无括号）`×1
- ch06: `\\mathbb{E}`×60, `E[（裸E）`×2, `\\mathbb E（无括号）`×1
- ch07: `\\mathbb{E}`×19, `E[（裸E）`×2, `\\mathbb E（无括号）`×1
- ch09: `\\mathbb{E}`×63, `E[（裸E）`×4
- ch10: `\\mathbb{E}`×31, `E[（裸E）`×2, `\\mathbb E（无括号）`×1
- ch12: `\\mathbb{E}`×66, `E[（裸E）`×2
- ch13: `\\mathbb{E}`×23, `E[（裸E）`×6

### 范数记号

- ch05: `\\|（双竖线）`×86, `‖（Unicode）`×6
- ch12: `\\|（双竖线）`×152, `‖（Unicode）`×4
- ch13: `‖（Unicode）`×79, `\\|（双竖线）`×68

### 行内公式定界

- ch15: `$...$（美元）`×360, `\\( \\)（圆括号）`×5

### 下标 0（x_0）

- ch10: `x_0`×87, `\\mathbf{x}_0`×63

### σ（噪声水平）

- ch01: `\\sigma`×107, `\\sigma_y`×3
- ch13: `\\sigma_y`×28, `\\sigma`×4

## 二、全书记法分布（决定统一目标）

| 记法组 | 记法 | 全书总次数 | 使用章数 |
|---|---|---|---|
| ε（噪声） | `\\epsilon` | 507 | 16 |
| ε（噪声） | `\\varepsilon` | 767 | 12 |
| ᾱ_t（累积信号保留率） | `\\bar\\alpha` | 451 | 8 |
| ᾱ_t（累积信号保留率） | `\\bar{\\alpha}` | 140 | 3 |
| ᾱ_t（累积信号保留率） | `\\barα（Unicode）` | 58 | 3 |
| x 的粗体 | `\\mathbf{x}` | 411 | 4 |
| x 的粗体 | `纯 x` | 6484 | 19 |
| ε_θ 的 hat | `\\hat\\epsilon_\\theta` | 17 | 1 |
| ε_θ 的 hat | `\\epsilon_\\theta（无hat）` | 155 | 6 |
| ε_θ 的 hat | `\\hat{\\epsilon}_\\theta` | 1 | 1 |
| 高斯记号 | `\\mathcal{N}` | 544 | 16 |
| 高斯记号 | `N(0,I)（裸N）` | 93 | 13 |
| KL 散度 | `D_\\text{KL}` | 22 | 2 |
| KL 散度 | `\\mathrm{KL}（无D前缀）` | 11 | 4 |
| KL 散度 | `D_KL（裸下标）` | 1 | 1 |
| 期望算子 | `\\mathbb{E}` | 682 | 17 |
| 期望算子 | `E[（裸E）` | 23 | 8 |
| 期望算子 | `\\mathbb E（无括号）` | 5 | 5 |
| 范数记号 | `\\|（双竖线）` | 2129 | 18 |
| 范数记号 | `‖（Unicode）` | 89 | 3 |
| 行内公式定界 | `$...$（美元）` | 15555 | 19 |
| 行内公式定界 | `\\( \\)（圆括号）` | 5 | 1 |
| 显示公式定界 | `$$（双美元）` | 2811 | 19 |
| 下标 0（x_0） | `x_0` | 1182 | 14 |
| 下标 0（x_0） | `\\mathbf{x}_0` | 63 | 1 |
| σ（噪声水平） | `\\sigma` | 1537 | 18 |
| σ（噪声水平） | `\\sigma_y` | 31 | 2 |

**章内混用组合共 48 处（组×章）。**
