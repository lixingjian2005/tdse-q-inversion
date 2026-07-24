# v1.0 Q-Inversion — 开发进度

## 当前状态

**阶段：v1.0 最小可行实现**

## 开发日志

### 2026-07-24

- [x] 项目目录结构建立
- [x] README.md 完成
- [x] 理论推导文档（Q函数重建_v3_完整论证）收录
- [x] 工程实施方案文档（inversion_engineering.tex/pdf）完成
- [x] Python venv 创建，numpy/scipy/matplotlib 安装
- [x] requirements.txt 创建
- [x] config.py — namelist 参数系统 (IO 组引号解析有已知小 bug)
- [ ] forward_model.py — 解析核正向模型
- [ ] transform.py — 1D FFT + 去模糊 + 截断
- [ ] gridding.py — Fourier 空间网格化
- [ ] reconstruct.py — 2D IFFT → Q(α)
- [ ] regularize.py — 截断/Tikhonov 正则化
- [ ] io_data.py — 数据读写
- [ ] diagnostics.py — 诊断报告
- [ ] invert_q.py — 主入口 + CLI
- [ ] closed-loop 测试通过

## 设计决策记录

1. Python 优先（numpy/scipy），不做 Fortran 扩展
2. 参数系统用 namelist 风格（与 v5.1 一致）
3. v1.0 仅做对角近似 + 截断正则化
4. 先 closed-loop 验证，再对接 v5.1 真实数据
