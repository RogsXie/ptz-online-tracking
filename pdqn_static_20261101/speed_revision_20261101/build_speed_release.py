"""Build a separate speed release only from complete actual evidence."""
import hashlib,json
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    qualification=load(ROOT/'qualification-v5-status.json');live=load(ROOT/'speed-live-v5-status.json')
    assert qualification['phase']=='independent_tests_complete' and qualification['qualified']
    assert live['phase']=='paired_realtime_complete' and live['qualified']
    cp=Path(qualification['selected']['checkpoint']);digest=sha(cp);assert digest==live['checkpoint_sha256']
    development=load(qualification['selected']['development_report']);corners=load(qualification['selected']['corner_report'])
    validation=load(qualification['validation_report']);final=load(qualification['final_report'])
    for s,n in ((development,80),(corners,32),(validation,160),(final,160)):
        assert s['qualification']['qualified'] and len(s['episodes'])==n and s['configuration']['checkpoint_sha256']==digest
    for split in ('validation','final_test'):assert load(ROOT/f'v5-{split}-speed-comparison.json')['qualified']
    comparison=load(live['comparison']);assert comparison['qualified'] and comparison['all16_quality_pass']
    cpu=load(live['cpu_audit']);assert cpu['passed']
    protocol=load(ROOT/'source-protocol-paper25/protocol-diagnostic-note.json');assert protocol['completed']
    protocol_rows=load(protocol['summary']);gpu=load(ROOT/'training-v5/gpu-update-proof.json');training=load(ROOT/'training-v5/status.json')
    for p,h in load(ROOT/'source-freeze-training-v5/manifest.json').items():assert sha(p)==h,p
    import torch
    checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
    assert checkpoint['completed_episodes']>0 and checkpoint['update_count']>0
    profile=load(ROOT/'training_profile_v5.json')
    files=load(OLD/'release-runtime-manifest.json')['files']
    paths=[*ROOT.glob('*.py'),*ROOT.glob('training_profile*.json'),ROOT/'speed-realtime-plan.json']
    paths += [p for p in (ROOT/'UE/PTZSim').rglob('*') if p.is_file() and not any(part in ('Saved','Intermediate','DerivedDataCache','.vs','Backup') for part in p.relative_to(ROOT/'UE/PTZSim').parts) and p.suffix.lower() not in ('.pdb','.log')]
    for p in paths:files[str(p)]=sha(p)
    for p,h in files.items():assert sha(p)==h,p
    manifest=ROOT/'speed-runtime-manifest.json';write(manifest,dict(files=files,scope='Preserved release imports plus isolated speed source/native runtime/frozen perception; no old release modified'))
    default=ROOT/'v5-paired-realtime/wide-outer-BR/speed/detection-overlay/live-detection-vfr.mp4';assert default.is_file()
    release=dict(qualified=True,created=datetime.now().astimezone().isoformat(),checkpoint=str(cp),checkpoint_sha256=digest,
        profile=profile,live_entry='live_speed_v5.py',runtime_manifest=str(manifest),validation_report=qualification['validation_report'],
        final_test_report=qualification['final_report'],development_report=qualification['selected']['development_report'],corner_report=qualification['selected']['corner_report'],
        paired_realtime_report=live['comparison'],cpu_decision_motion_audit=live['cpu_audit'],default_boxed_video=str(default),
        original_release=str(OLD/'qualified-release.json'),paper_fidelity='Unmodified pinned P-DQN update math/four networks; explicit UE,perception,reward,duration,andfine-tuning extensions; not full paper-protocol replication')
    write(ROOT/'qualified-speed-release.json',release)
    rows=['固定俯视静态车辆 PTZ：速度修订独立验收报告',f'生成：{release["created"]}','',
        '新入口：'+str(ROOT/'run_speed_in_pycharm.py'),'使用现有50mamba解释器；运行真实UE闭环，50秒原速录制，结束后自动逐帧导出检测框。',
        '旧模型、旧入口、旧验收报告和所有失败分支完整保留。','',
        '选定权重：'+str(cp),'SHA256：'+digest,
        f'选定速度分支：{checkpoint["completed_episodes"]}个新真实控制训练回合、{checkpoint["environment_steps"]}条新真实转移、{checkpoint["update_count"]}次原作者更新。',
        '新转移含64个8步均匀随机片段的512条转移；片段不作为完整控制回合。新分支缓存从空开始，无V1/V2/V3/V4缓存混入。',
        f'另完整训练至{training["completed"]}回合，实际累计{training["steps"]}条转移、{training["updates"]}次更新；这些总数与选定检查点分开报告。',
        '继承已训练480回合、48,402条真实转移、12,262次更新的原合格策略；四个网络初始化时逐位相同。',
        '四网络、两个在线网络梯度、两个Adam的动量均验证位于cuda:0；原作者梯度反演的CPU边界保留。','',
        '完整质量验收（均为同一冻结检查点）',
        f'开发：{development["qualification"]["passed"]}/80；已知真角落：{corners["qualification"]["passed"]}/32；独立验证：{validation["qualification"]["passed"]}/160；首次最终测试：{final["qualification"]["passed"]}/160。',
        '独立验证种子40,000,000段，最终测试41,000,000段；最终测试在验证合格后执行。开发/训练种子与其分离。',
        '最后5秒全部为策略Stay，中心归一化误差均值≤.025、P95≤.05，面积.025..055、平均abs(log(area/.04))≤.25；45秒内停稳。',
        '仍检查初始完整框、决策可见率≥.99、时间可见率≥.99、正确目标匹配≥.98、最长丢失≤.25秒、换向≤4次/10秒、快速不当换向≤2。',
        '总体/象限/尺度≥95%、每偏移层≥90%，160例的80个单元每格至少1/2合格；没有放宽质量门槛。','',
        '相同初始场景的同步UE时间对照（不是实际墙钟速度）']
    for split in ('validation','final_test'):
        s=load(ROOT/f'v5-{split}-speed-comparison.json')
        rows.append(f'{split}：旧/新停稳中位数{s["old_median_settle_sim_s"]:.3f}/{s["median_settle_sim_s"]:.3f}秒，比例{s["median_ratio"]:.3f}；旧/新P95 {s["old_p95_settle_sim_s"]:.3f}/{s["p95_settle_sim_s"]:.3f}秒，比例{s["p95_ratio"]:.3f}；需要变焦例首次变焦中位数{s["need_zoom_first_median_sim_s"]:.3f}秒，外层{s["outer_need_zoom_first_median_sim_s"]:.3f}秒。')
    rows+=['','8个预先固定实例的真正实时对照',
        f'16次新录制均通过质量核查。旧/新实际墙钟停稳中位数{comparison["median_wall_settle_old_s"]:.3f}/{comparison["median_wall_settle_new_s"]:.3f}秒，比例{comparison["median_ratio"]:.3f}。',
        f'外层四例旧/新墙钟停稳中位数{comparison["outer_median_old_s"]:.3f}/{comparison["outer_median_new_s"]:.3f}秒。',
        '每次从重置开始实际运行/ptz/play；原RGB帧和墙钟时标保留，VFR原速编码，无插值重放、端点复制或视频加速。',
        '这8例属于已知开发实例，不作为新增独立统计测试。感知在决策末帧核查，选定对照视频的框另外由同一冻结RGB检测器逐帧重算。',
        '默认带框视频：'+str(default),'',
        f'独立CPU复核{cpu["total_decisions"]}次新验证/测试决策，离散不匹配{cpu["discrete_action_mismatches"]}，连续输出最大差{cpu["continuous_parameter_max_abs_cpu_gpu_difference"]:.9g}，UE姿态/FOV最大偏差{cpu["max_physical_pose_fov_error_deg"]:.9g}度。','',
        '与原论文/代码的对应关系和边界',
        '直接使用只读作者仓库commit9173676fb636140ffa4145e101631d3f30634267的agent.py/model.py/memory.py，四网络架构、sum-Q参数梯度、梯度反演、均匀回放、gamma.9、batch512、目标100次硬拷贝保留。',
        '8维状态保留area*100、延迟动作与原时长索引行为；动作和所有7个连续参数由网络输出。没有GT方向、PID/MPC、老师动作或强制先变焦规则。',
        '论文Table2转速25°/秒、变焦5°/秒、960×540已采用，并按真实UE姿态变化独立核查。',
        '原始参数Box.3..5保留，但执行转换显式改为每参数单位：Stay.1秒、tilt.016秒、pan.032秒、zoom.2秒；不是原代码统一raw/5。',
        '实际时长范围：Stay .03...5秒，tilt .0048...08秒，pan .0096...16秒，zoom .06..1秒。短脉冲只在独立UE工程副本中把两个原生时长下限改为.001秒；旧工程未改。',
        '与旧版每个参数对应的转角一致，角速度提高使转角更快完成。这里的时间校准是工程扩展，不能称作完整原论文协议还原。',
        'Q学习率1e-6、参数学习率1e-7用于保守微调，区别于原作者.001/.0001；更新函数本身不改。',
        '使用已披露的几何/4%面积奖励及5秒Stay终止，区别于原论文/代码的置信度+5%面积奖励、连续10帧置信度>.9终止；V4换向奖励没有进入选定V5分支。',
        'UE5.7.4固定俯视静态单车，FOV14..66透视模拟、yaw±65/pitch-80..-40，与作者平台、15倍物理光学及论文角限不同。',
        '冻结UE适配YOLO26s(full+half)，真实VeRi FastReID2048维；不是原脚本的MMDetection/YOLOX感知配置。GT仅在控制完成后用于独立匹配核查，不进入控制或奖励。','',
        '原协议行为诊断（不是原论文训练性能复现）']
    for mode in ('source_exact','confidence_termination_disabled_only'):
        rr=[x for x in protocol_rows if x['profile']==mode]
        rows.append(f'{mode}：{len(rr)}例，原置信度终止成功{sum(x["source_success"] for x in rr)}，历史较宽几何诊断成功{sum(x["quality_success"] for x in rr)}。')
    rows+=['这里用保留的276回合旧原协议检查点在25°/秒场景做行为对照，它没有在该25°/秒俯视环境重新训练；因此不能据此评价论文方法原始性能。','',
        '失败分支：V1完成40回合后开发失败；V2完成40回合后开发失败；V3完成160回合，四个开发快照均未通过；V4完成22回合，20回合开发达到5次失败后停止。原始记录、检查点和停止原因均保留。',
        '本交付应称为：保留作者P-DQN内核的UE任务适配与实训速度修订；不应称整篇论文端到端原样复现。']
    (ROOT/'REPORT-SPEED.txt').write_text('\n'.join(rows)+'\n',encoding='utf-8-sig')
    print('Separate speed release and evidence report built:',ROOT/'qualified-speed-release.json')
if __name__=='__main__':main()
