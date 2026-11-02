"""Build a human-readable release only from completed independent evidence."""
import csv,hashlib,json,subprocess
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON=Path('C:/Users/12879/.conda/envs/50mamba/python.exe')

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    status=load(ROOT/'exploration-qualification-status.json');live=load(ROOT/'realtime-evidence-status.json')
    assert status['phase']=='independent_tests_complete' and status['final_test_qualified']
    assert live['phase']=='realtime_examples_complete' and live['qualified'] and len(live['results'])==8
    selected=status['selected'];checkpoint=Path(selected['checkpoint']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    paths=dict(corners=Path(selected['corner_report']),development=Path(selected['development_report']),
        validation=Path(status['validation_report']),final_test=Path(status['final_test_report']))
    reports={name:load(p) for name,p in paths.items()}
    for name,r in reports.items():
        assert r['qualification']['qualified'] and r['configuration']['checkpoint_sha256']==digest
        assert load(paths[name].parent/'frozen-verification.json')['networks_targets_optimizers_replay_rng_unchanged']
    assert reports['validation']['qualification']['count']==reports['final_test']['qualification']['count']==160
    assert all(load(r['audit'])['passed'] and load(r['audit'])['checkpoint_sha256']==digest for r in live['results'])
    capture_manifest=load(ROOT/'release-live-evidence/source-freeze.json')
    for path,sha in capture_manifest.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    runtime_files={p:sha for p,sha in capture_manifest.items() if Path(p).name not in
        ('plot_release_live.py','audit_live.py','audit_release_live.py','realtime-evidence-plan.json')}
    for p in [ROOT/'run_in_pycharm.py',ROOT/'live_demo_visible.py',ROOT/'render_detection_video.py',ROOT.parent/'PTZSim/Source/PTZSim/PTZArena.cpp',
              ROOT.parent/'PTZSim/Source/PTZSim/PTZArena.h',ROOT.parent/'PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll',
              Path('D:/UE/UE_5.7/Engine/Build/Build.version')]:
        assert p.is_file();runtime_files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    first_cfg=load(Path(live['results'][0]['folder'])/'configuration.json')
    for path,sha in [(first_cfg['detector']['checkpoint'],first_cfg['detector']['sha256']),
                     (first_cfg['reid']['weights'],first_cfg['reid']['weights_sha256'])]:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha;runtime_files[path]=sha
    runtime_manifest=ROOT/'release-runtime-manifest.json'
    write(runtime_manifest,dict(files=runtime_files,scope='Frozen control/evaluation source, entry, native project source/binary and perception weights'))
    audit=ROOT/'release-original-output-physical-audit.json'
    if not audit.exists():
        subprocess.run([str(PYTHON),'-B',str(ROOT/'audit_policy_decisions.py'),'--checkpoint',str(checkpoint),
            '--folders',*[str(p.parent) for p in paths.values()],'--output',str(audit)],check=True,
            creationflags=subprocess.CREATE_NO_WINDOW)
    decision_audit=load(audit);assert decision_audit['passed'] and decision_audit['checkpoint_sha256']==digest
    summary=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,
        training=dict(completed_episodes=480,new_exploration_fragments=640,fragment_steps=5120,
            selected_lineage_real_transitions=48402,selected_checkpoint_updates=12262,
            note='Fragments are not additional complete control episodes; rejected branches and calibration/evaluation are separate'),
        qualification={name:r['qualification'] for name,r in reports.items()},
        realtime_examples=live['results'],original_decision_audit=str(audit),
        material_scope='One stationary target, fixed overhead mount, UE5.7.4 perspective-FOV simulation; original learner core with declared task/calibration/training extensions')
    write(ROOT/'final-qualification-summary.json',summary)
    groups=[]
    for split in ('validation','final_test'):
        q=reports[split]['qualification'];groups.append(dict(split=split,group='overall',value='all',passed=q['passed'],count=q['count'],rate=q['passed']/q['count']))
        for group,values in q['groups'].items():
            for value,counts in values.items():groups.append(dict(split=split,group=group,value=value,
                passed=counts['passed'],count=counts['count'],rate=counts['passed']/counts['count']))
    with (ROOT/'grouped-qualification.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['split','group','value','passed','count','rate']);writer.writeheader();writer.writerows(groups)
    for split in ('validation','final_test'):
        rows=reports[split]['episodes'];keys=[k for k in rows[0] if k not in ('checks','failures')]
        with (ROOT/f'{split}-per-case.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=keys+['failures']);writer.writeheader()
            writer.writerows({**{k:r[k] for k in keys},'failures':','.join(r['failures'])} for r in rows)
    video_rows=[]
    for item in live['results']:
        a=load(item['audit']);video_rows.append(dict(name=Path(item['folder']).name,split=a['case']['split'],index=a['case']['index'],
            passed=a['passed'],center_mean=a['center_mean'],center_p95=a['center_p95'],area_mean=a['area_mean'],
            settling_s=a['settling_time_s'],wall_s=a['wall_duration_s'],raw_frames=a['raw_captured_frames'],
            encoded_frames=a['encoded_frames'],video=item['video']))
    with (ROOT/'realtime-examples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(video_rows[0]));writer.writeheader();writer.writerows(video_rows)
    lines=[
        '固定俯视 PTZ 静态车辆：作者 P-DQN 内核实训与独立验收',
        f'报告生成：{datetime.now().astimezone().isoformat(timespec="seconds")}',
        '',
        '运行入口',
        f'在 PyCharm 中运行：{ROOT / "run_in_pycharm.py"}',
        f'现有解释器：{PYTHON}',
        '默认无参数会打开真实 UE 摄像机窗口，使用已验收权重控制并录制新的50秒实时视频。',
        '可选 --case 0..159 切换验证场景；--headless 适用于后台录制。输出位于 demo-runs。',
        '本入口不安装依赖、不训练模型；使用当前50mamba环境。',
        '启动时同时核验控制源文件、UE项目模块、感知权重和策略权重的哈希。',
        '可见窗口使用独立HUD适配器：动作开始后恢复最近的RGB检测显示，消除原生状态栏误显preview；控制仍调用同一冻结live_demo。',
        '录制后自动额外导出detection-overlay/live-detection-vfr.mp4：同一冻结检测器/ReID逐帧处理已保存RGB，绿色框、置信度和蓝色中心十字。',
        '带框展示版为录制后的逐帧检测叠加；原始控制录像、决策与验收记录保留。',
        '',
        '选定权重与训练计数',
        str(checkpoint),f'SHA256: {digest}',
        '本任务训练分支从作者默认随机初始化开始；旧276回合作者模型仅用于单独基线对照。',
        '选定分支完成480个控制训练回合、43,282条真实转移；另采集640个真实8步随机探索片段、5,120条转移。',
        '合计48,402条选定分支真实训练转移。短片段不计作完整控制回合。选定权重累计12,262次原版更新。',
        '新片段采集期间四个网络和两个优化器冻结；CPU逐位比较与逐条缓存/日志核对已通过。',
        '真实GPU训练证明：training-source-cadence/gpu-update-proof.json。四网络、梯度、Adam动量位于cuda:0。',
        '',
        '独立验收',
        f'已知32个真实角落：{reports["corners"]["qualification"]["passed"]}/32；已知80个完整开发场景：{reports["development"]["qualification"]["passed"]}/80。',
        f'独立验证：{reports["validation"]["qualification"]["passed"]}/160；首次最终测试：{reports["final_test"]["qualification"]["passed"]}/160。',
        '最终测试仅在完整独立验证通过后启动；同一权重，评估期间四网络、目标网络、优化器、缓存、RNG状态均恢复并核验。',
        '每组含四个象限、五档偏移、四档初始FOV/尺度；每个80格单元各2例。验证与测试使用未训练种子及中间FOV范围。',
        '各象限/尺度组要求至少95%，每偏移层至少90%，总体至少95%，每个单元至少1/2通过。',
        '验收量由独立文件程序从原始框、时钟和单独GT记录重算；GT不进入网络状态、动作、奖励或终止。',
        '居中使用归一化坐标误差sqrt((2cx-1)^2+(2cy-1)^2)。最后5秒全部Stay，均值<=0.025、P95<=0.05。',
        '面积在2.5%..5.5%，平均abs(log(area/0.04))<=0.25；45秒内停稳，最多50秒。',
        '另检查初始完整框、目标保持、最长非预期丢失<=0.25秒、换向频率<=4次/10秒及快速不当换向<=2。',
        '',
        '分组结果（象限0/1/2/3对应左上/右上/左下/右下）'
    ]
    for row in groups:lines.append(f'{row["split"]}: {row["group"]}={row["value"]}: {row["passed"]}/{row["count"]}')
    lines.extend(['','真实实时影像','8个预先固定的实例覆盖四个象限和四档尺度，含四个远端真角落、大目标主动缩小和已居中尺度合适的起点。',
        '使用UE实际实时动作和中间RGB采集；视频来自真实墙钟时标的VFR编码，不是端点插值或轨迹重放。',
        '逐例独立核查静止目标、固定机位、配对时间、最后5秒Stay、正确目标保持、停稳、换向及原版网络输出。',
        '这些是已知场景的新实时演示，不作为额外独立统计测试。原始采集帧数与编码帧数分别列出。'])
    lines.append('感知/身份匹配按决策末帧统计，原始视频每帧另检GT可见性；不把视频帧数当独立场景数。')
    for row in video_rows:lines.extend([f'{row["name"]}: PASS={row["passed"]}, mean={row["center_mean"]:.5f}, P95={row["center_p95"]:.5f}, area={row["area_mean"]:.5f}, settle={row["settling_s"]:.2f}s',row['video']])
    lines.extend(['','作者内核与工程扩展的边界',
        '直接编译只读作者仓库commit9173676fb636140ffa4145e101631d3f30634267的agent.py、model.py和memory.py。',
        'Q网络、参数网络及各自目标网络；原版均匀回放、Adam、gamma0.9、batch512、100次硬拷贝目标网络。',
        '学习率Q=0.001、参数=0.0001；保留原梯度反演及其CPU拷贝，保留原sum-Q参数梯度行为。',
        '保留作者8维状态中的area*100、延迟动作和原时长索引行为。所有7个连续参数由参数网络输出。',
        '执行边界明确clip到[0.3,5]/5，即0.06..1秒；原始输出保存在Q输入及回放中。',
        '已完成100次CUDA更新与只读记录器逐位一致性检查；这是源代码对应性测试，不是控制成功样本。',
        f'独立CPU复核{decision_audit["total_decisions"]}次冻结决策：离散动作不匹配{decision_audit["discrete_action_mismatches"]}次，连续输出最大差{decision_audit["continuous_parameter_max_abs_cpu_gpu_difference"]:.8g}。',
        f'UE实际姿态/FOV变化与所选动作、持续时间及限位最大偏差{decision_audit["max_physical_pose_fov_error_deg"]:.8g}度。',
        '以下属于明确披露的工程任务扩展，不是整篇论文端到端原样复现：',
        'UE5.7.4固定俯视机位、静态单车；平移4度/秒、俯仰2度/秒、FOV5度/秒；透视FOV模拟，不是物理光学镜头。',
        '感知改用适配UE的YOLO26s及既有full+half输入，替换原脚本MMDetection/YOLOX接口；冻结真实VeRi FastReID，实际2048维。',
        '检测与身份匹配均由RGB产生，感知权重在本任务控制训练和评估期间冻结。',
        '作者置信度完成条件另做对照；本任务使用几何/尺度奖励和5秒Stay完成，目标面积从原0.05改为0.04。',
        '更新间隔由原32改为4；后期补足采样覆盖并使用作者epsilon1随机探索分支收集真实短片段，再作原版回放更新。',
        '没有PID、MPC、方向教师、坐标回归或用GT选择动作。已知开发集负责模型选择，最终测试不用于调参。',
        '',
        '保留的失败及对照',
        '原置信度基线12/12完成，但几何验收0/12；仅取消置信度结束仍0/12。',
        '320回合完整开发74/80、角落24/32；480回合完整开发80/80、角落30/32。',
        '旧数据额外2048原版更新导致角落24/32；加入真实新片段后2048更新31/32、1024更新32/32。所有失败均保留。',
        '早期0320开发评估时补过初始GT完整框离线检查；随后统一重算全部80例，仍为74/80，未改变任一通过判定。',
        '之后的冻结评估保存前后源文件哈希；本次验证和最终测试期间控制与验收文件未变。',
        'corner-comparison-after-exploration/corner-cell-comparison.png提供逐单元对照。',
        '',
        '主要证据文件',
        'protocol.json；heldout_coverage_declaration.json；exploration_support_declaration.json；realtime-evidence-plan.json',
        'real-exploration-support-after0480/independent-data-audit.json',
        'release-original-output-physical-audit.json；final-qualification-summary.json；grouped-qualification.csv',
        'validation-per-case.csv；final_test-per-case.csv；realtime-examples.csv',
        *[str(p) for p in paths.values()],
        '',
        '适用边界：已验证静态单车辆和固定俯视机位；移动目标、多目标遮挡及真实硬件迁移尚未验收。'
    ])
    proof_path=ROOT/'pycharm-entry-proof.json'
    if proof_path.exists():
        proof=load(proof_path);assert proof['passed'] and proof['checkpoint_sha256']==digest
        summary['pycharm_entry_proof']=proof
        lines.extend(['','PyCharm实际运行证明',str(proof_path),
            '已从现有IDE默认Python3.12转接50mamba，完成50秒可见UE闭环并以退出代码0结束；独立视频审计通过。',
            str(Path(proof['fresh_capture'])/'live-sensor-vfr.mp4')])
    boxed_path=ROOT/'boxed-video-update/result.json'
    if boxed_path.exists():
        boxed=load(boxed_path);summary['boxed_video_export']=boxed
        lines.extend(['','带检测框的视频',boxed['video'],
            '框由同一冻结RGB检测器逐帧重算，不是GT框；源控制录像与时标保留。'])
    write(ROOT/'final-qualification-summary.json',summary)
    report_text='\n'.join(lines)+'\n'
    (ROOT/'使用说明与实验结果.txt').write_text(report_text,encoding='utf-8-sig')
    (ROOT/'REPORT.txt').write_text(report_text,encoding='utf-8-sig')
    release=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,validation_report=str(paths['validation']),
        final_test_report=str(paths['final_test']),default_demo_case=48,realtime_report=str(ROOT/'realtime-evidence-status.json'),
        runtime_manifest=str(runtime_manifest),
        original_output_physical_audit=str(audit),usage_report=str(ROOT/'使用说明与实验结果.txt'))
    if proof_path.exists():release['entry_run_proof']=str(proof_path)
    if boxed_path.exists():release['boxed_video']=load(boxed_path)['video']
    write(ROOT/'qualified-release.json',release)
    print('QUALIFIED RELEASE',json.dumps(release,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
