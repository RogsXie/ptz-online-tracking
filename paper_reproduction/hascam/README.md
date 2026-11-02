# HAS-CAM 算法实现与审计约定

本目录实现论文所用的 parameterised DQN，接 UE 替代平台。它不是作者的 Unity CamSim，也不是先前七动作固定 0.2 秒 PPO。源码依据为本项目 `reference/HAS-CAM` 的固定快照；更完整的原文/官方源码冲突记录见上级 `docs/paper_audit.md`。

## 模块

| 论文部分 | 文件 | 实际实现与检查 |
| --- | --- | --- |
| §3.1、Table 3、作者 model.py/agent.py | `agent.py`、`config.py` | Q_Actor 输入 15（8 状态+7 时长）、ParamNet 输入 8，两个网络隐藏层均为 128/256/128/64、输出 7，ReLU 隐层、线性输出。贪心用 Q 选择动作类型并执行其连续时长。 |
| Eq 1–3 | `config.py` | Stay/Down/Up/Right/Left/ZoomIn/ZoomOut；每种动作一个时长候选，执行时长 0.04–1 秒。禁止把持续时间离散成 bins。 |
| Eq 4、Table 1 | `detection.py`、`state_reward.py` | 纯渲染检测 bbox/confidence + 前一动作产生 8 维状态。offset/trend 用半图像大小归一化；area 为面积比例；action/7；duration 秒；reset 四项历史为 0；丢失为 [-2,-2,-1,0,0,0,0,0]。不读取相机内参、真值坐标或投影框。 |
| Eq 5–12 | `state_reward.py` | 保留 `paper_literal` 和 `author_code` 两种明确 profile，详见下文。终止奖励另由环境在最终 transition 添加。 |
| Algorithm 1 / §3.2 | `cfff.py` | 先按置信度排序，以 cosine 保持当前目标，再排除已完成的外观记忆。阈值 .525，当前/历史融合 .85/.15；连续 10 个动作后检测 confidence≥.9 才完成，完成外观为该段特征均值。多目标任务是顺序视觉增强；不是有真值 ID 的实例追踪。 |
| §3.2.1、作者 FastReID config | `reid.py` | 官方 VeRi SBS R50-IBN checkpoint；ResNet50+IBN+5 nonlocal+GeM+BN neck，256×256 RGB crop，真实 2048 维 L2 特征。权重及 SHA-256 在 `weights/*.provenance.json`。全部推理 tensor 严格匹配，唯一旧 BN counter 重命名记录在 metadata；训练分类头不参与推理。 |
| §4.2 / Eq 13 | `agent.py`、`config.py` | Adam Q lr=.001、参数 lr=.0001、γ=.90、容量 20000、batch512，均匀无放回回放；每32步由运行器请求更新；ε 从 completed episodes 的 .9 线性到300时 .1，之后 .1。 |
| 作者公开优化步骤 | `agent.py` | Huber Q-loss，Q 梯度逐元素[-1,1]；参数目标 mean(sum Q)，inverting gradients 后上升，参数梯度范数 clip10；每100次优化硬同步两个 target。checkpoint 保存四网络、两优化器、回放、计数与本地随机数状态。 |

## 必须显式记录的论文/源码差异与本次选择

- PDF Table 3 的 8/15 标签与作者代码网络名称似乎对调。采用实际可运行的 Q(s,all_parameters) 和 Param(s) 结构。
- PDF 正文称外观 4096 维，作者 FastReID YAML 及官方 checkpoint 均为 2048。输出真实 2048，metadata 同时写明论文数字；不拼接、补零或冒称 4096。
- 作者仓库没有打包其重新训练的检测器与论文实验精确 ReID 权重。本次使用 FastReID 官方 VeRi 公开权重，不能称为论文作者训练权重。
- PDF 式 7 为 `1-.5(x²+y²)`，式 9 是框内1/框外0，在总负号下会反向惩罚居中。小面积分支式8也与作者运行源码相反且在 .05 有跳变。`paper_literal` 如印刷公式执行这些项；`author_code` 采用作者环境实际执行的有效 cost：position=.5(x²+y²)，小面积 area=((a-.05)/.05)²、大面积 area=1-exp(-(a-.05)/.1)，inbox 为内0/外1。两种都保留 lost=-4；各项改善加0.5采用可累计解释。
- `author_code` 名称指本次选择的源码 per-frame cost 和改善阈值/guard；状态仍以论文 Table 1 为准。原源码 reset 前罚项全0、跨实例历史未正确隔离。本实现只在上一帧确有捕获时比较改善；运行器在 CFFF `new_target/acquired` 时清趋势和奖励历史，避免两个实例之间算“运动/改善”。这项修正应记入实验配置。
- 作者环境持续参数边界 .3–5，日志除5得 .06–1；论文要求 .04–1。本实现直接以秒使用论文边界。作者线性参数输出没有执行硬保护；本实现将候选投影到合法时长后选择/执行/写回放，以防不可行控制。
- 参数网末层可行初始值 .52、微小随机权重是论文未指定的初始化选择（`feasible_param_init=True`）。其余层保持 PyTorch 默认初始化；参数输出仍线性并用 inverting gradients 更新。
- 原源码回放默认100k覆盖了 memory.py 的20k；本实现选论文20k。原文“每episode初始化网络”若逐字执行将遗忘学习，本实现如官方源码在构造时初始化一次。
- 本实现 `update()` 为一次优化，由调用方按每episode step_index%32==0调度。源码在第1/33/65/97步尝试更新，且回放不足512时不更新。
- 原算法伪码初次选择时即加入 Fmon（line9）会与“只排除已完成目标”冲突；按作者运行代码只在任务完成后加入。原文 L、均值权重未完整指定，本实现容量100 FIFO（配置可改）、完成融合采用作者均值。`choose(...,count_step=False)` 用于 reset 取初始框，初始帧不计入10个动作。
- 完成个数是外观簇计数。完全相似车辆可能被合并；TC 必须用记录的完成框在离线评价中与真实实例匹配，不可拿记忆长度当覆盖率。CFFF 失检会清当前外观，但保留已完成记忆，不含真值恢复捷径。
- 128步超时优先失败；`terminal_adjustment(success,steps)` 只在 success 且steps<128时+40，否则-40。多目标全任务完成/每目标事件由环境层处理；此模块不接收 GT 目标身份或位置。

## 验证

在既有 `D:\EI\ptz_ue\python_env\Scripts\python.exe` 运行 `verify_math.py`，输出 `verification/math-results.json`。检查覆盖 Table 1、哨兵、动作/时长、ε、VPE 两种读取、CFFF 连续帧/切换/FIFO/同外观合并局限、独立 Bellman 数值 oracle、640 合成 transition 上两个网络实际更新与100次硬同步、checkpoint 完整续训一致。

ReID 还以同一 checkpoint 和随机 256×256 输入，直接加载固定作者源码的 ResNet/IBN/nonlocal 原始类，核对 adapter 前向最大绝对误差。严格加载成功不是唯一证据，需该数值对照也通过。

这些检查只验证数学和实现机制。合成 transition 不是 UE 任务训练、收敛结果或论文性能复现；实际训练、独立测试、视频和场景配置由上级运行器记录。
