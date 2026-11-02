# 作者源码对应训练（2026-10-31）

用户要求重新按原文源码训练。`learned_control_v2` 已停止在 66 个完整回合，未达标；不再作为当前复现方案。未实施 MP-DQN、新奖励或额外训练技巧。旧 320 回合失败权重及 v2 失败记录保留。

当前入口：`D:\EI\ptz_ue\scripts\train_author_source_50mamba.py`。核心固定为作者仓库 commit `9173676fb636140ffa4145e101631d3f30634267`。每次启动校验原文件 SHA-256。

## 对应关系

| 部分 | 本次执行方式 | 作者文件 |
| --- | --- | --- |
| Q / 参数网络 | 直接导入原类；128/256/128/64，原始 Linear 初始化 | `model.py` |
| 选择动作、TD、Huber、反转梯度、优化器 | 直接调用原 PDQNAgent；不另写公式 | `agent.py` |
| 回放 | 原 ReplayBuffer，均匀无放回；保留全部 7 个原始参数 | `memory.py` |
| 超参数 | batch512，gamma.9，Qlr.001，参数lr.0001，回放100000，target每100次更新 | `main.py` / `agent.py`；源码回放100000与论文20000有差异，按用户要求采用源码 |
| 调度 | 每回合零基步数0/32/64/96更新；epsilon(回合序号)，第1回合.897333；每10回合保存 | `main.py` |
| 状态 / 奖励 / CFFF / 结束条件 | 编译并执行原环境类完整、未改写的 AST，仅绑定 I/O 依赖 | `rl_control/UnityCameraEnvironment_exp.py` |
| 损失播报 | 读取原update返回帧中的实际loss_actor、param_loss；不改变反传 | 外部观测器 `source.py` |

源码环境保留面积×100、动作/6、丢失sentinel、迟一拍的动作历史及原时长索引行为。原奖励为VPE成本和阈值改善奖；单车连续10次置信度严格>.9完成（+4并触发+40），128步超时-40。这是视觉增强任务，不能把置信度完成直接解释成持续精准跟踪。

## 必要适配及证据边界

1. UE替代Unity。用户明确确认物理时长=原始参数/5；作者声明Box[.3,5]对应.06–1秒。仓库没有Unity执行端，因此不宣称此映射已由Unity源码证明。
2. 原参数网络线性输出可越界。用户于2026-10-31明确授权仅执行端裁剪，因此训练入口默认`--execution-boundary clip_box`：执行端裁剪到[.3,5]再/5；Q、回放、目标Q、参数梯度仍使用原始参数，每步保存原值、执行值、裁剪标志和实际时间。该规则是UE适配，非作者算法。保留`reject`模式用于检查越界。
3. 保持用户冻结YOLO26s权重、detector阈值.15、960输入、half=False和已验证FastReID。原环境额外的`score_thr=.3`过滤保持源码原值，未调参；与检测器自身阈值分开记录。当前检测器与作者MMDetection模型不同。
4. Gymnasium只提供Env/space容器；当前SciPy的cosine输入将(1,2048)展平成2048维。ReID预处理保留原BGR→RGB、256×256、bicubic。
5. 使用内存中的真实UE RGB替代Unity截图文件；关闭cv2 GUI，无录像或离线重录。
6. 原main.py默认恢复缺失的epoch_4060；当前从头初始化，不加载旧失败策略。为可重复性显式设置torch和全局numpy种子，原随机采样算法不变。
7. 原`choose_action(train=False)`分支有错误；验证用原训练分支epsilon=0得到同一个贪心选择路径，禁用更新并在验证后恢复RNG/计数。验证前后逐张量检查四个网络一致。

## 当前训练范围和结果口径

先做静态训练审计批次，默认32回合；原预算4060回合写入配置，不把短批次冒充完成原实验。场景直接使用历史`scenarios/static_near.json`：静止单车、25度/秒、变焦5度/秒、全部7动作，不导入v2的8度/秒、动作屏蔽和固定FOV。没有原作者Unity场景源码，场景不能称为完全一致。

新训练seed700000起；可选40例冻结验证seed800000–800039。未使用旧40000诊断集或最终测试。不会自动切换动态/复杂场景。旧严格持续控制门槛保持为独立尚未通过的验收，不能拿源码confidence-only完成替代。

每轮`round-metrics.log/.csv`、`episodes.jsonl`及控制台都记录实际Q损失、参数目标（最大化）、梯度范数、奖励、epsilon、步数、回放大小、更新次数、学习率、batch、中心误差、源码完成与质量连续10帧。质量旁路沿用已有中心≤.15、面积[.02,.08]、置信度≥.9规则，仅统计，不改奖励或终止。

`initial.pt`、每10轮checkpoint及`last.pt`含4网络、2优化器、完整回放、随机状态与来源元数据，拒绝加载旧profile。每步原始/执行动作可审计。运行目录放置`STOP_REQUEST`会在下一动作边界退出并保存中断记录。

验证命令：

```powershell
$env:PYTHONPATH='D:\EI\ptz_ue\paper_reproduction'
& 'C:\Users\12879\.conda\envs\50mamba\python.exe' -m author_aligned.verify --device cuda
```

合成一致性检查不算真实训练成绩；实际训练、验证成绩只以新run的逐步日志和检查点为准。
