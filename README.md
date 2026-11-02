# PTZ 在线单目标跟踪

## 在线演示

**当前版本：前段双轴居中修复 + 急弯身份接续。** 保留原十字路口、货车遮挡、同色跟车和转弯后不停加速上高架的场景，巡航速度 **30 m/s**。下方为本次接受的完整 28 秒真实在线实录，点击即可播放。

https://github.com/user-attachments/assets/8d53e993-9220-40ba-84dd-28050ceb2ccc

[下载本版完整 MP4](pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/front-centering-fix-20261103/final-original-scene/online-output/true-online-output.mp4) · [分段对照结果](pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/front-centering-fix-20261103/comparison.json) · [修复与回退说明](pdqn_static_20261101/speed_revision_20261101/FRONT_CENTERING_FIX.md)

## 当前版本

2026-11-03，`v2026.11.03-front-centering`。默认入口为 `pdqn_static_20261101/speed_revision_20261101/live_fast_online_results.py`；版本、权重和视频 SHA-256 记录在 `current-version.json`。

- 修复水平动作连续占用反馈、纵向偏差长期得不到纠正的问题；优先较大误差轴，相近误差交替服务。
- 放大前居中容差收紧到 12 px；使用有上限的 RGB 运动与反馈时延补偿，初次捕获脉冲仍限制在 100 ms 内。
- 保留已修复的局部多帧急弯重关联、轨迹归属保护、重复框假设以及独立更新的几何尺寸历史。
- 初始选择最高置信度对象；短时丢失保持身份并留空，连续丢失 3 秒后才遗忘并重选。约 1.6% 目标尺度、2% 框面积上限、居中黄框和黄色距离线保留。
- 场景与上一轮原场景 JSON 完全相同：弯后 12 米连接段、80 米上桥坡道、420 米高架、三车并行，主车转弯后加速至 30 m/s，不停车等车。
- 相比上次已发布版本，本次采用从头训练后继续急弯训练的 precision15 策略权重；累计 11,008 条真实交互、4,992 次原作者 P-DQN 更新。检测与 ReID 权重未变。**本次居中修复本身只改控制规则，没有再次训练网络。**

## 本次实录结果

| 项目 | 实测 |
| --- | --- |
| 真实在线处理输出 | **23.62 FPS**，662 次实际更新 |
| 可见目标框 | 651 个，全部对应原红车 |
| 空框结果 | 11 个；转弯 9–12 秒无空框 |
| 目标身份 | 保持原 ID，本段未观察到串车或遗忘重选 |
| 源帧到发布延迟中位数 | **94.64 ms** |
| 初次连续进入中心 20 px 内 | **0.63 → 0.53 秒** |
| 前段 0.75–6 秒居中误差中位数 / P95 | **28.7 / 53.6 → 11.1 / 34.6 px** |
| 4–6 秒居中误差中位数 | **48.0 → 13.4 px** |

居中误差由所有原生输入图像事后评分，包含无框和遮挡时刻，不只统计成功检测。初始目标仍从原来的角落开始，未隐藏起始画面或预先居中；前 0.75 秒 P95 仍约 336 px，尚非全程像素级居中。另一次急弯回归保持原 ID，267 次出框、2 次确认留空，未出视野。

结果来自有限布局，不代表跨场景全面可靠。实际画框在 UE 运行期间直接使用对应 RGB 帧的在线输出；结束后只编码保存的像素，不重跑推理、补框或插帧。相机采集目标频率与实际输出频率分别记录。控制器不读取目标真值、路线或场景 ID，目标真值只用于结束后的诊断。

本版录像、逐次决策、原生帧元数据、已绘制图像及录制时源码快照：

`pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/front-centering-fix-20261103/final-original-scene/`

进入版本目录后，可重新核对随仓库保存的原始记录：

```powershell
python audit_front_centering.py --baseline unified_motion_20261101/user-original-scene-turn-fix-20261103 --candidate unified_motion_20261101/front-centering-fix-20261103/final-original-scene --output centering-comparison-local.json
```

上一版 Release 和旧策略保留。使用 `--no-centering-v4` 可仅回退到上一轮摄像机控制；要回到上次发布的完整版本，请检出旧标签 `v2026.11.02-viaduct30-nonstop`。大型 ReID/FFmpeg 依赖仍由原有 `release-assets.json` 恢复；新 Release 另附本次 MP4、部署权重和结果清单。

## 环境与模型

运行环境是 Windows、Unreal Engine 5.7、NVIDIA CUDA GPU。实测环境见 `environment-observed.json`。当前机器使用已有的 `50mamba` Python 环境与 CUDA 12.8 Torch；不要直接用旧实验的 requirements 文件覆盖这个环境。

已将当前策略和整车检测权重一并纳入代码快照：

- 当前策略：`pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/control-v3-turn-training-20261102/deploy.pt`
- 旧策略保留回退：`pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/pdqn_training_v1/last.pt`
- `pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/whole_vehicle_detector_v1/whole_vehicle.pt`

较大的 FastReID 权重和原版 FFmpeg 二进制位于本私人仓库的 Release。使用有仓库读取权限的 GitHub Git 凭据，或者在环境变量 `GH_TOKEN` / `GITHUB_TOKEN` 中提供令牌后运行：

```powershell
python tools/restore_release_assets.py
```

脚本会恢复到原目录并验证 SHA-256；不会输出或保存令牌。清单见 `release-assets.json`。

Python 依赖至少包括 `torch`、`torchvision`、`numpy`、`opencv-python`、`gymnasium`、`ultralytics`、`UnrealCV`、`threadpoolctl`、`scipy`、`Pillow`。准确实测版本记录在 `environment-observed.json`；源码中保留原依赖布局。

## 编译 UE 项目

项目位于 `pdqn_static_20261101/speed_revision_20261101/UE/PTZSim/PTZSim.uproject`。仓库包含项目源码、配置、地图与 UnrealCV 插件源码。编译产物和 UE 引擎本身不在代码仓库中。

需要 Unreal Engine 5.7 与对应的 Visual Studio C++ 编译工具链。可用引擎的 Build.bat 编译：

```powershell
$project = Join-Path $PWD 'pdqn_static_20261101/speed_revision_20261101/UE/PTZSim/PTZSim.uproject'
& 'D:/UE/UE_5.7/Engine/Build/BatchFiles/Build.bat' PTZSimEditor Win64 Development "-Project=$project" -WaitMutex -NoHotReloadFromIDE
```

当前源码在 `scripts/run_in_pycharm.py` 的 `UE_EDITOR` 使用本机引擎路径 `D:/UE/UE_5.7/Engine/Binaries/Win64/UnrealEditor.exe`。其他机器需按实际安装位置调整。启动程序会使用此快照中的独立 UE 项目。

## 运行

从仓库根目录进入当前版本目录，使用已准备好的 CUDA Python 环境：

```powershell
Set-Location pdqn_static_20261101/speed_revision_20261101
python live_fast_online_results.py --scene viaduct30 --output unified_motion_20261101/live-local-001 --seconds 28 --fps 30 --perception-fps 30
```

输出目录必须是新目录。同一时间只运行一个 UE/GPU 实验。`--fps` 是相机采集目标频率，实际处理输出频率以 `online-output/online-recording-audit.json` 为准。

部署时所有网络权重冻结。当前使用新的 precision15 训练权重，并包含明确的 RGB 关联和摄像机控制规则；本次前段居中修复本身没有再次训练网络。

针对新增流程的检查：

```powershell
python check_balanced_centering.py
python check_robust_tracker.py
python check_turn_reassociation.py
python check_turn_view_identity.py
```

## 快照与第三方来源

`source-snapshot-manifest.json` 记录从工作目录复制的文件及原始字节 SHA-256；Git 对文本采用 LF 规范化，因此工作树行尾可能不同。首次提交的示例保留用于历史对照；其他试错录像、训练数据集、无关训练检查点、缓存和机器凭据未包含在本快照中。

作者原始 P-DQN 来源为 `Vampon/HAS-CAM`，固定提交 `9173676fb636140ffa4145e101631d3f30634267`，完整来源声明保留在 `paper_reproduction/reference/HAS-CAM` 与 `paper_reproduction/docs/source_manifest.json`。

FastReID 的 Apache-2.0 声明保留在 `paper_reproduction/hascam/LICENSE_FastReID.txt`。UnrealCV、原始 HAS-CAM 及其附带组件的条款保留在各自目录中。此快照未给第三方源码重新授予许可证；具体权利以对应原始条款为准。
