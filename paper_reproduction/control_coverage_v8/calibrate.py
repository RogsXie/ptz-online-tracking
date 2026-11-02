"""Real RGB-only frontend check and coverage gallery; no learning/actions."""
from pathlib import Path
import hashlib
import json
import math
import cv2
import numpy as np
import torch
from integration.runtime import StandaloneArena, REPRO
from integration.pretrained_detector import create_detector
from integration.experiment import make_extractor
from integration.metrics import iou
from control_learning.environment import AtomicTransport
from control_corners.run import write_json
from .scenes import cases, materialize, stage_spec, layout, initial_zone, load_protocol
from .environment import CoverageEnvironment

OUT = REPRO/'verification/coverage-static-v8-20261031'


def main():
    torch.set_num_threads(4)
    protocol = load_protocol()
    out = OUT/'initial-rgb-calibration'
    out.mkdir(parents=True, exist_ok=False)
    detector = create_detector(checkpoint=REPRO/protocol['frozen_frontend']['detector'], device='cuda')
    extractor = make_extractor('cuda')
    detector.warmup()
    assert detector.provenance['sha256'] == protocol['frozen_frontend']['detector_sha256']
    assert extractor.metadata()['weights_sha256'] == protocol['frozen_frontend']['reid_sha256']
    protected = [detector.checkpoint, extractor.weights]
    before = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in protected}
    write_json(out/'frozen-before.json', before)
    results = []
    with StandaloneArena(out, headless=True) as arena:
        transport = AtomicTransport(arena.port, hud=False)
        try:
            for case in cases('static_center', 'calibration'):
                seed = case['seed']
                folder = out/f'seed{seed}'
                scene = materialize('static_center', 'calibration', seed, folder)
                env = CoverageEnvironment(transport, detector, extractor, stage_spec('static_center'))
                env.reset(scene, seed)
                row = env.rows[0]
                target = transport.diagnostics['targets'][0]
                truth_box = target.get('gt_bbox_normalized')
                item = layout(seed, 'calibration')
                checks = {
                    'rgb_layout': initial_zone(row, seed, 'calibration'),
                    'correct_target_iou_0_5': bool(row.get('selected_bbox') and truth_box and iou(row['selected_bbox'], truth_box) >= .5),
                    'complete_visible_target': bool(truth_box and all(.005 <= v <= .995 for v in truth_box) and not target['gt_center_occluded']),
                    'single_detection': len(env.detections) == 1,
                    'fixed_installation': transport.diagnostics['camera_location'] == [-2200, 0, 4500],
                    'initial_downward_orientation': max(abs(a-b) for a,b in zip(row['rotation'], [-60,0,0])) < 1e-4,
                    'fov_matches': abs(row['fov']-item['fov_deg']) < 1e-4,
                }
                image = folder/'initial.png'
                assert cv2.imwrite(str(image), cv2.cvtColor(transport.rgb, cv2.COLOR_RGB2BGR))
                result = item | {'passed': all(checks.values()), 'checks': checks, 'image': str(image), 'observation': row,
                                 'gt_box_diagnostic_only': truth_box, 'diagnostics': transport.diagnostics}
                write_json(folder/'report.json', result)
                results.append(result)
                write_json(out/'progress.json', {'completed': len(results), 'total': 126, 'passed': sum(r['passed'] for r in results)})
                print('CALIBRATION', len(results), seed, result['passed'], flush=True)
        finally:
            transport.close()
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    write_json(out/'frozen-after.json', after)
    assert before == after
    report = {'passed': all(r['passed'] for r in results), 'passed_cases': sum(r['passed'] for r in results), 'total': 126,
              'cases': results, 'detector': detector.provenance, 'reid': extractor.metadata(), 'hashes_unchanged': True,
              'training_updates': 0, 'learned_actions': 0,
              'scope': 'Development layout/frontend coverage only; not independent control validation or learned success'}
    write_json(out/'report.json', report)
    render(report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','detector','reid']}), flush=True)


def render(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in report['cases'] if r['observation']['visible']]
    fig, ax = plt.subplots(figsize=(10,6))
    points = ax.scatter([(r['observation']['x']+1)/2 for r in rows], [(r['observation']['y']+1)/2 for r in rows],
        c=[r['observation']['area_ratio']*100 for r in rows], cmap='viridis', s=44, edgecolor='black', linewidth=.35)
    ax.axvline(.5, color='gray', lw=.8); ax.axhline(.5, color='gray', lw=.8)
    ax.set(xlim=(0,1), ylim=(1,0), xlabel='Detected box center x / image width', ylabel='Detected box center y / image height',
           title='126 real UE initial observations: 2D offset and initial scale coverage')
    fig.colorbar(points, ax=ax, label='Detected box area (% of image)')
    fig.tight_layout(); fig.savefig(OUT/'initial-center-distribution.png', dpi=160); plt.close(fig)
    cards = []
    for r in report['cases']:
        row = r['observation']; box = row.get('selected_bbox'); rectangles = ''
        if box:
            x1,y1,x2,y2 = [v*s for v,s in zip(box,[960,540,960,540])]
            rectangles = f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#17ed88" stroke-width="2"/>'
        path = Path(r['image']).relative_to(OUT).as_posix()
        cards.append(f'<figure data-layer="{r["layer"]}" data-scale="{r["scale_group"]}"><svg viewBox="0 0 960 540"><image href="{path}" width="960" height="540"/>{rectangles}</svg><figcaption>#{r["seed"]} · 象限{r["quadrant"]+1} · 距离层{r["layer"]} · FOV {r["fov_deg"]:.1f}° · 框面积{row.get("area_ratio",0)*100:.2f}% · {"通过" if r["passed"] else "失败"}</figcaption></figure>')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>多距离、多尺度场景覆盖</title><style>body{font:16px system-ui;background:#101923;color:#e7eef4;margin:26px}p{line-height:1.8}#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0;background:#243345}svg{width:100%;display:block}figcaption{padding:10px;font-size:13px}select{padding:8px;margin:8px}#plot{max-width:1000px;width:100%}</style><h1>四象限 × 多距离 × 多尺度</h1>'''
    page += f'<p>126 个真实 UE 开发初始条件：{report["passed_cases"]}/126 通过位置、完整可见和检测核验。相机安装位置与初始朝向固定；尺度通过初始 FOV 改变，回合内焦距固定。车辆原生尺寸，无黄色立柱。没有学习更新；这不是控制验证通过。</p><img id="plot" src="initial-center-distribution.png"><p>距离层：0 已居中，1 近中心，2 内圈，3 中圈，4 外圈。散点来自实际检测框，颜色为实际图像面积。</p><label>距离<select id="layer"><option value="all">全部</option>'+''.join(f'<option>{i}</option>' for i in range(5))+'</select></label><label>尺度组<select id="scale"><option value="all">全部</option>'+''.join(f'<option>{i}</option>' for i in range(3))+'</select></label><section id="grid">'+''.join(cards)+'</section>'
    page += '''<script>function filter(){const l=document.querySelector('#layer').value,s=document.querySelector('#scale').value;document.querySelectorAll('figure').forEach(e=>e.hidden=!((l==='all'||e.dataset.layer===l)&&(s==='all'||e.dataset.scale===s)));}document.querySelectorAll('select').forEach(e=>e.onchange=filter);</script></html>'''
    (OUT/'scene-coverage.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
