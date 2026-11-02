"""Direct original reward/termination class at paper25rate, UE sensor adapter.

This is a protocol diagnostic with the preserved source-trained276checkpoint,
not a claim of a newly trained or complete original-paper simulation result.
"""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import json,hashlib
import baseline as original_runner
from scenes import materialize as original_materialize

def main():
    configure_precision_project();base=REV/'source-protocol-paper25';base.mkdir(exist_ok=False)
    original_runner.HERE=base
    original_runner.materialize=lambda folder,item:original_materialize(folder,item,(25.,25.,5.))
    writer=original_runner.write
    def declared_write(path,value):
        if path.name=='configuration.json':
            value.update(task_adaptation='Fixed overhead static UE, paperTable2rate25/25/5; unchanged originalclass reward andconfidence10completion; sourceRawBox.3..5,raw/5execution',
                scope='Protocol behavior diagnostic, not a source-trained-at25claim or statistical quality qualification',
                reference_quality_note='Historical helper reports looser center.05geometric diagnostic; it is not the newstrict.025independent acceptance')
        return writer(path,value)
    original_runner.write=declared_write
    original_runner.main()
    path=base/'source-baseline/summary.json'
    results=json.loads(path.read_text(encoding='utf-8'))
    (base/'protocol-diagnostic-note.json').write_text(json.dumps(dict(completed=True,cases=len(results),
        original_source_class=True,paper_rate=[25,25,5],raw_box=[.3,5],duration_mapping='raw/5',
        reward='Original confidence+position+area5percent+inbox and improvement bonuses',termination='Original ten consecutiveconfidence>.9; disabledonlyinexplicitsecondcontrast',
        current_perception='Frozen UE-adapted YOLO and genuineFastReID; not originalMMDetection detector',
        caveat='Preserved276checkpoint was not newlytrained for this25deg/s overheadscene. No comparison here proves originalpapertraining performance.',
        summary=str(path)),indent=2),encoding='utf-8')

if __name__=='__main__':main()
