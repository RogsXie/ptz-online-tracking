# PTZ 在线单目标跟踪

2026-11-02 当前版本的源码快照，包含 UE 5.7 场景、RGB 整车检测、FastReID、身份保持、冻结的 P-DQN 控制器、真实在线画框与录像。

当前入口：`pdqn_static_20261101/speed_revision_20261101/live_fast_online_results.py`。

## 当前行为

- 初始选择置信度最高的目标，随后保持同一身份；持续丢失 3 秒才遗忘并重新选择。
- 首次锁定后先大致居中，再主动放大到约 1.6% 框面积，保留 2% 面积上限。启动缩放每次最多改变 6° FOV，每次都等待真实新画面反馈。
- 遮挡时清除目标框、逐步扩大视野搜索；不会给其他车临时套上目标框。
- 居中时框变黄，显示目标到画面中心的黄色距离线。
- 感知与电机等待分开运行；电机仍执行互斥动作。录像中的画面与框是在 UE 运行时根据当前实际处理结果绘制，导出只压缩已保存像素，不做离线重检测或补帧。

本次 30 秒示例：真实在线输出约 **23.65 FPS**；约 **1.0 秒**开始主动放大，**2.42 秒**达到启动放大完成阈值。699 个可见结果均匹配原红车，11 个结果在遮挡期间没有框，转弯 9–12 秒无掉框。该数据仅来自本次场景，不代表广泛场景可靠性。像素居中误差与单帧延迟仍有优化空间。

视频和记录位于：

`pdqn_static_20261101/speed_revision_20261101/unified_motion_20261101/user-startup-active-zoom-v1-video/`

其中 `online-output/true-online-output.mp4` 是实际在线结果；`startup-scale-result.json` 和 `online-output/online-recording-audit.json` 是对应记录。

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
python live_fast_online_results.py --output unified_motion_20261101/live-local-001 --seconds 30 --fps 30 --perception-fps 30
```

输出目录必须是新目录。同一时间只运行一个 UE/GPU 实验。`--fps` 是相机采集目标频率，实际处理输出频率以 `online-output/online-recording-audit.json` 为准。

控制器权重保持冻结；该版本包含明确的工程控制规则，不是重新训练后的纯 P-DQN。

针对新增流程的检查：

```powershell
python check_startup_scale.py
python check_continuous_online.py
```

## 快照与第三方来源

`source-snapshot-manifest.json` 记录从工作目录复制的文件及原始字节 SHA-256；Git 对文本采用 LF 规范化，因此工作树行尾可能不同。历史实验视频、训练数据集、其他训练检查点、缓存和机器凭据未包含在本快照中。

作者原始 P-DQN 来源为 `Vampon/HAS-CAM`，固定提交 `9173676fb636140ffa4145e101631d3f30634267`，完整来源声明保留在 `paper_reproduction/reference/HAS-CAM` 与 `paper_reproduction/docs/source_manifest.json`。

FastReID 的 Apache-2.0 声明保留在 `paper_reproduction/hascam/LICENSE_FastReID.txt`。UnrealCV、原始 HAS-CAM 及其附带组件的条款保留在各自目录中。此快照未给第三方源码重新授予许可证；具体权利以对应原始条款为准。
