"""Focused comparison helpers; all geometric scoring is post-run only."""
import argparse,json
from pathlib import Path
from evaluate_precision_policy import postrun_camera,postrun_identity


def score(folder):
    cases={}
    for case in sorted(folder.glob('case-*')):
        records=json.loads((case/'offline-identity-diagnostics.json').read_text())
        identity=postrun_identity(records);camera=postrun_camera(records,identity['initial_selected_identity'])
        cases[case.name]=dict(camera=camera,identity=identity)
    return cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    old,new=score(args.baseline),score(args.candidate)
    result=dict(baseline=old,candidate=new,scoring='All completed native observations including no-box. GT scoring only after the run; perception fixed.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for case in new:
        a,b=old[case]['camera'],new[case]['camera']
        print(case,'out',a['target_out_of_view'],'->',b['target_out_of_view'],'pixel median/P95',a['center_median_p95_px'],'->',b['center_median_p95_px'])


if __name__=='__main__':main()
