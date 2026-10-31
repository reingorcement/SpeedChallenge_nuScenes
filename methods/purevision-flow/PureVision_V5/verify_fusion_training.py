"""Bounded real trainer interruption/resumption and independent RGB-state evaluation."""
from argparse import ArgumentParser,Namespace
from pathlib import Path
import json
import torch
from purevision_v5.common import ROOT,now,read_json,sha256,write_json
from purevision_v5.datasets import VelocityDataset
from purevision_v5.model import build_model
from purevision_v5.training import train,evaluate_model,model_state_sha256
from startup_watch import check_checkpoint


def verify(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope='smoke_real_trainer_not_formal_results',started_at=now(),rows={})
    for name in getattr(a,'variants',('R02_FLEX_CONT','R03_Frozen','R03_FT')):
        config=ROOT/'configs/next'/f'{name}.json';conf=read_json(config)
        features=a.features if name=='R03_Frozen' else None
        args=Namespace(config=str(config),manifest=a.manifest,flow=a.flow,features=features,selection=None,rgb_frames=getattr(a,'rgb_frames',None),
            output=str(output/name/'interrupted'),device=a.device,workers=0,threads=4,resume=None,smoke=True,
            limit_scenes=1,limit_windows=128,max_batches=2,stop_after_steps=3)
        first=train(args);args.output=str(output/name/'resumed');args.resume=str(first/'last.pt');args.stop_after_steps=5
        resumed=train(args);packet=torch.load(resumed/'last.pt',map_location='cpu',weights_only=True)
        assert packet['processed_batches']==packet['processed_microbatches']==5==packet['optimizer_updates']+packet['amp_skips']
        assert packet['optimizer_updates']>0
        restored=build_model(conf).to(a.device);restored.load_state_dict(packet['model'],strict=True)
        dev=VelocityDataset(a.manifest,'dev',conf['experiment'],features,a.flow,rgb_policy=conf['rgb_policy'],frame_root=getattr(a,'rgb_frames',None),limit_scenes=1,limit_windows=128)
        assert dev.contract==packet['contract']['data'] and dev.selection_hash==packet['contract']['dev_selection']
        summary,predictions=evaluate_model(restored,dev,a.device,batch_size=8,workers=0,max_batches=2,lateral_rule=conf['lateral_rule'])
        assert len(predictions)==16 and torch.isfinite(torch.tensor(summary['overall']['epe']))
        run=read_json(resumed/'run.json');validation=json.loads((resumed/'metrics.jsonl').read_text().splitlines()[0])
        proof=check_checkpoint(resumed/'last.pt',run,validation)
        if name=='R03_Frozen':assert model_state_sha256(restored.rgb_encoder)==packet['initialization']['rgb_state_sha256']
        if name=='R03_FT':assert model_state_sha256(restored.rgb_encoder)!=packet['initialization']['rgb_state_sha256']
        report['rows'][name]=dict(status='passed',initialization=packet['initialization'],
            processed_batches=5,optimizer_updates=packet['optimizer_updates'],amp_skips=packet['amp_skips'],
            processed_microbatches=packet['processed_microbatches'],recovery_proof=proof,
            independent_evaluation_count=len(predictions),independent_evaluation_scope='16 real dev windows, smoke only',
            checkpoint_sha256=sha256(resumed/'last.pt'),saved_rgb_bn_used=True)
        write_json(output/'summary.json',report);del restored,packet,dev
        torch.cuda.empty_cache()
    report.update(status='passed',finished_at=now());write_json(output/'summary.json',report)
    print(dict(status='passed',summary=str(output/'summary.json')),flush=True)


if __name__=='__main__':
    p=ArgumentParser()
    for key in ('manifest','flow','features','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--variants',nargs='+',default=['R02_FLEX_CONT','R03_Frozen','R03_FT'])
    p.add_argument('--rgb-frames')
    p.add_argument('--device',default='cuda:0');verify(p.parse_args())
