# 分阶段 RGB 云台控制协议

本目录是新的 `learned_control_v2` 实验协议。失败的320回合模型、原论文配置和五类场景均保留；检测器与ReID冻结。这里不声称与作者代码逐行等价。

原P-DQN核心之外，新控制配置明确使用物理秒 `.04–.25`、水平/垂直8度/秒、变焦8 FOV度/秒，单次最大转动2度。第一阶段只允许Stay/上下左右，FOV固定60度；其后增加变焦，再依次加入匀速、变速、转弯、遮挡、低照度。每个阶段的门槛在 `protocol.json` 中记录。增加难度只由实际验证门槛通过触发；训练回合预算耗尽不会触发升级。

第一阶段场景版本 `static-center-local-limits-v2` 将机械范围限定为pitch[-12°,0°]、yaw[-10°,10°]，保持原有种子初始偏差不变，避免局部基础任务探索到大面积空天空。该调整发生在4回合启动检查之后、正式训练和验证之前；旧启动配置及源码快照保留。范围是固定场景参数，不读取真值来瞄准。其他阶段的范围和全部数值验收门槛不变，protocol SHA仍为 `c7ad22074859972a15e7dcdef1978d4c78b023e94a450d80e0104d8420923ede`。

第一阶段验证固定40个新种子，四种水平/垂直初始偏差符号各10例。至少38/40通过，而且每一象限至少9/10通过。每例必须完成10模拟秒真实RGB闭环，4秒内收敛，然后持续至少5秒稳定：中心径向误差均值≤.025、p95≤.05，95%以上时间在.05内；全程选中框可用率≥99%、离线正确目标匹配覆盖≥98%，最长非预期丢失≤.25秒，快速且增大误差的无谓反向最多2次；水平、垂直与变焦两两相反指令合计频率≤4次/10模拟秒；FOV保持60度。静态变焦阶段也应用此反向频率门槛。验证不以训练奖励或置信度单独判成功，也不允许只有终点好看的短片通过。

中心误差使用原始归一化图像偏移，`sqrt((2cx-1)^2+(2cy-1)^2)`。门槛计算检查它和x/y一致，避免将网络内部放大的特征当作原始误差。稳定指标按动作末RGB样本和实际模拟时长加权；不能称为动作间像素级连续真值。真实动作正负、单位、相机位移和采集顺序还需实际UE校准。静态阶段的固定相机名义方向加独立种子偏差，从不读取真值来瞄准。

训练种子每阶段从`200000+阶段号×10000`开始；验证从`500000+阶段号×10000`开始固定40个。保留的最终测试从1000000以上的独立区间分配，默认禁止生成，只有全部阶段和回归门槛通过后才能显式释放。历史40000段测试已用于诊断，不能重新称为全新未见测试。

API：

```python
from control_curriculum import cases, materialize, stage_spec, assess_episode, gate_aggregate
plan = cases("static_center", "validation")  # seeds510000..510039
path = materialize("static_center", "validation", plan[0]["seed"], output_dir)
result = assess_episode("static_center", rows)
result["seed"] = plan[0]["seed"]
# Collect all40 cases; incomplete/duplicate/bad-quadrant sets fail.
gate = gate_aggregate("static_center", episode_results)
```

行格式：`sim_time_s, center_error, x, y, area_ratio, visible, matched, action_type, duration_s, fov`，初始行`t=0,duration=0`。`matched`仅供离线门槛核验，不得送入策略或奖励。`assess_episode`返回明确的passed/失败原因、收敛时间、末段均值/p95、可用率、丢失、反向和动作幅度；`gate_aggregate`检查完整固定种子和protocol SHA，不会自行训练或升级。

后续阶段也预先声明在protocol中；遮挡阶段在本版门槛实现中保守地拒绝通过，直到补齐实际RGB遮挡/恢复事件证据接口，不能用一条中心射线代替可见性。这不阻碍先完成基础静态阶段。

纯机制验证：`verification/gated-control-20261031/protocol/check_protocol.py`。它不启动UE、不加载模型、不训练、不生成最终测试配置。运行成功只证明协议代码的机制，不证明真实控制已达标。每次真实训练仍必须保存其协议副本、源码hash、种子、实际场景、冻结前端hash、动作/相机/RGB日志及独立验证结果。
