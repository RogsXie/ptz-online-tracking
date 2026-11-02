# PTZ 在线单目标跟踪

## 在线演示

**当前版本：转弯后不停加速上桥，主车与三车并行巡航均为 30 m/s。** 下方是最新 28 秒真实在线结果，点击即可播放。

https://github.com/user-attachments/assets/af4d80f7-1cc0-4fc4-9f4e-2e66fa291591

[下载本版完整 MP4](pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/user-nonstop-viaduct30-v1-video/online-output/true-online-output.mp4) · [查看本版结果与核对记录](pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/user-nonstop-viaduct30-v1-video/nonstop-30mps-result.json)

## 当前版本

2026-11-02，`v2026.11.02-viaduct30-nonstop`。当前入口为 `pdqn_static_20261101/speed_revision_20261101/live_fast_online_results.py`，运行参数见下文；`current-version.json` 记录当前版本和示例路径。

- 保留原十字路口、接近段、转弯及蓝车/货车安排；弯后连接段为 **12 米**。
- 移除桥上和上桥前的停车、等待。主车约 **10.95 秒**完成转弯，在 **1.1 秒**内按 12 → 21 → 30 m/s 提速，然后保持 30 m/s。
- 三车沿 **80 米上桥坡道**进入高 **8 米**的桥面。桥面长 **420 米**，三条车道各宽 4 米。
- 后方红车、紫车短时分别约 35.25、42.75 m/s 追齐，约 **24 秒**后与主车并排，统一保持 **30 m/s**。主车不会为了等车减速或停车；录像结束时车辆仍在行驶。
- 原有启动主动放大、约 1.6% 目标尺度、2% 框面积上限、居中黄框、黄色距离线保留。
- 初始选择最高置信度目标，后续保持身份。短时丢失清框搜索，连续丢失 3 秒才遗忘并重新选择。采用联合匹配、持续的轨迹归属保护和可信特征更新，避免同色车抢框。
- 检测、ReID、P-DQN 权重保持冻结；当前控制包含明确的 RGB 运动估计、限幅、搜索缩放等工程规则。

## 本次实录结果

| 项目 | 实测 |
| --- | --- |
| 真实在线处理输出 | **23.57 FPS**，28 秒内 660 次实际更新 |
| 主车巡航 / 并行速度 | **30.00 / 30.00 m/s** |
| 可见目标框 | 648 个，全部匹配原红车 |
| 空框结果 | 12 个；转弯 9–12 秒无掉框 |
| 目标身份 | 始终为首次锁定对象，本段未观察到串车 |
| 原始采集到在线输出延迟中位数 | 约 **97.3 ms** |

这些数据来自这一段实录，并非跨场景可靠性保证。画框在 UE 运行时直接根据对应 RGB 帧的实际在线结果绘制；结束后只压缩已保存像素，不做离线重检测、补帧或重画框。相机目标采集频率 30 FPS 与实际处理输出频率分别记录。

本段场景、逐次决策、原生帧诊断、在线绘制图像与源文件快照位于：

`pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/user-nonstop-viaduct30-v1-video/`

进入当前版本目录后，可用 `python audit_nonstop_viaduct30.py` 读取随仓库保存的记录核对速度、并行位置及结果完整性；该脚本不运行模型、不修改录像。运行期间控制器和画框不读取目标真值，原生目标位置仅供事后核对。

大型 ReID/FFmpeg 依赖与模型权重未变，下载脚本继续使用已校验的依赖 Release；本版 Release 另附最新演示视频。旧示例与旧 Release 保留，README 默认展示本版。

## 环境与模型

运行环境是 Windows、Unreal Engine 5.7、NVIDIA CUDA GPU。实测环境见 `environment-observed.json`。当前机器使用已有的 `50mamba` Python 环境与 CUDA 12.8 Torch；不要直接用旧实验的 requirements 文件覆盖这个环境。

已将当前策略和整车检测权重一并纳入代码快照：

- `pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/pdqn_training_v1/last.pt`
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

控制器权重保持冻结；该版本包含明确的工程控制规则，不是重新训练后的纯 P-DQN。

针对新增流程的检查：

```powershell
python check_startup_scale.py
python check_robust_tracker.py
python check_continuous_online.py
```

## 快照与第三方来源

`source-snapshot-manifest.json` 记录从工作目录复制的文件及原始字节 SHA-256；Git 对文本采用 LF 规范化，因此工作树行尾可能不同。首次提交的示例保留用于历史对照；其他试错录像、训练数据集、无关训练检查点、缓存和机器凭据未包含在本快照中。

作者原始 P-DQN 来源为 `Vampon/HAS-CAM`，固定提交 `9173676fb636140ffa4145e101631d3f30634267`，完整来源声明保留在 `paper_reproduction/reference/HAS-CAM` 与 `paper_reproduction/docs/source_manifest.json`。

FastReID 的 Apache-2.0 声明保留在 `paper_reproduction/hascam/LICENSE_FastReID.txt`。UnrealCV、原始 HAS-CAM 及其附带组件的条款保留在各自目录中。此快照未给第三方源码重新授予许可证；具体权利以对应原始条款为准。
