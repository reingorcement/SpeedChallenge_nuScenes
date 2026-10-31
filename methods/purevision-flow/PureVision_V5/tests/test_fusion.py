"""Fusion-specific regression tests; real-backbone gradient evidence is verify_fusion.py."""
from argparse import Namespace
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from purevision_v5.common import read_json,write_json
from purevision_v5.model import build_model,FlexFlowEncoder
from purevision_v5.optimization import make_optimizer,make_scheduler
from purevision_v5 import training


class TinyRGB(nn.Module):
    """Cheap stand-in only for trainer recovery mechanics, not architecture verification."""
    def __init__(self,*,trainable=True):
        super().__init__();self.trainable_rgb=trainable
        self.backbone=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.ReLU())
        self.region_projection=nn.Linear(4,384);self.requires_grad_(trainable);self.train(trainable)
    def train(self,mode=True):return super().train(mode if self.trainable_rgb else False)
    def forward(self,x):return self.region_projection(self.backbone(x.mean(1)).mean((2,3)))


@pytest.mark.parametrize('policy',['frozen','finetune'])
def test_fusion_resume_and_independent_evaluation_include_rgb_state(tmp_path,monkeypatch,policy):
    from test_training import make_case,load_state,assert_tree_equal
    import purevision_v5.rgb as rgb
    import evaluate
    monkeypatch.setattr(rgb,'RGBDescriptor',TinyRGB)
    monkeypatch.setattr(training,'code_version',lambda require_pushed=False:dict(commit='test',dirty=False))
    monkeypatch.setattr(training,'environment',lambda:dict(device='cpu_test'))
    args=make_case(tmp_path/'case',experiment='R03',flows=True)
    config=read_json(args.config);config.update(encoder_variant='flex',rgb_policy=policy,rgb_learning_rate=1e-4,
        scheduler_policy='cosine_each_group_to_0.1',micro_batch_size=2,evaluation_batch_size=2)
    write_json(args.config,config)
    if policy=='finetune':
        args.features=None
        for record in read_json(args.manifest)['sequences']:
            for i,f in enumerate(read_json(Path(args.manifest).parent/record['file'])['frames']):
                Image.fromarray(np.full((80,140,3),20+i,dtype=np.uint8)).save(Path(args.manifest).parent/f['filename'])
    args.output=str(tmp_path/'full');full=load_state(training.train(args))
    args.output=str(tmp_path/'stopped');args.stop_after_steps=2;stopped=training.train(args)
    args.output=str(tmp_path/'resumed');args.stop_after_steps=0;args.resume=str(stopped/'last.pt')
    resumed=load_state(training.train(args))
    for key in ('model','optimizer','scheduler','scaler','rng','initialization','processed_batches',
                'processed_microbatches','optimizer_updates','amp_skips','best_epe'):
        assert_tree_equal(full[key],resumed[key])
    assert full['processed_microbatches']==12
    packet=torch.load(tmp_path/'full/best.pt',weights_only=True,map_location='cpu')
    assert any(k.startswith('rgb_encoder.backbone.1.running_') for k in packet['model'])
    eval_args=Namespace(checkpoint=str(tmp_path/'full/best.pt'),manifest=args.manifest,features=args.features,
        flow=args.flow,output=str(tmp_path/'evaluation'),device='cpu',workers=0,threads=1)
    result=evaluate.evaluate(eval_args)
    assert result['overall']['epe']==read_json(tmp_path/'full/best_metrics.json')['overall']['epe']
    assert [g['name'] for g in full['optimizer']['param_groups']]==(['flow_head','rgb'] if policy=='finetune' else ['flow_head'])
    if policy=='frozen':
        assert full['model']['rgb_encoder.backbone.1.num_batches_tracked'].item()==0
    else:assert full['model']['rgb_encoder.backbone.1.num_batches_tracked'].item()==12


def test_group_cosine_decays_each_lr_and_restores(monkeypatch):
    import purevision_v5.rgb as rgb
    monkeypatch.setattr(rgb,'RGBDescriptor',TinyRGB)
    config=dict(experiment='R03',encoder_variant='flex',rgb_policy='finetune',learning_rate=1e-3,
        rgb_learning_rate=1e-4,weight_decay=1e-4,epochs=40,scheduler_policy='cosine_each_group_to_0.1')
    model=build_model(config);assert isinstance(model.flow_encoder,FlexFlowEncoder)
    opt=make_optimizer(model,config);sched=make_scheduler(opt,config)
    for _ in range(20):opt.step();sched.step()
    state=sched.state_dict();groups=opt.state_dict()
    other=make_optimizer(model,config);other.load_state_dict(groups);resumed=make_scheduler(other,config);resumed.load_state_dict(state)
    for _ in range(20):opt.step();sched.step();other.step();resumed.step()
    assert [g['lr'] for g in opt.param_groups]==pytest.approx([1e-4,1e-5])
    assert [g['lr'] for g in other.param_groups]==[g['lr'] for g in opt.param_groups]


def test_ft_rejects_cached_descriptor_and_future_fields(monkeypatch):
    import purevision_v5.rgb as rgb
    monkeypatch.setattr(rgb,'RGBDescriptor',TinyRGB)
    model=build_model(dict(experiment='R03',encoder_variant='flex',rgb_policy='finetune'))
    with pytest.raises(ValueError,match='never frozen'):model(dict(descriptor=torch.zeros(1,384)))
    with pytest.raises(ValueError,match='Forbidden'):model(dict(future_pose=torch.zeros(1)))


def test_uint8_views_preserve_pixels_and_bind_tokens(tmp_path):
    from test_training import make_case
    from prepare_rgb_frames import prepare
    from purevision_v5.rgb_frames import RGBFrames
    from purevision_v5.rgb import preprocess_rgb
    from purevision_v5.common import sha256
    args=make_case(tmp_path/'case');manifest=Path(args.manifest);source=manifest.parent
    generator=np.random.default_rng(42)
    for record in read_json(manifest)['sequences']:
        for f in read_json(source/record['file'])['frames']:
            Image.fromarray(generator.integers(0,256,(79,141,3),dtype=np.uint8)).save(source/f['filename'])
    out=tmp_path/'views';prepare(manifest,out,2);reader=RGBFrames(out,sha256(manifest))
    f=read_json(source/read_json(manifest)['sequences'][0]['file'])['frames'][3]
    with Image.open(source/f['filename']) as image:expected=preprocess_rgb(image)
    torch.testing.assert_close(expected,reader.load(f['token'],f['filename']),rtol=0,atol=0)
    with pytest.raises(ValueError,match='filename/token'):reader.load(f['token'],'wrong.jpg')
    with pytest.raises(ValueError,match='manifest'):RGBFrames(out,'wrong')


def test_patrol_reads_summary_for_current_stage_and_logs_only_on_anomaly(tmp_path):
    from scripts.prepare_patrol import prepare
    root=tmp_path;prefix=root/'patrol';run=root/'runs/current';run.mkdir(parents=True)
    for name in ('run.json','status.json'):write_json(run/name,dict(status='running'))
    log=root/'current.log';log.write_text('normal log\n')
    job=dict(job_id='current',status='running',run_dir=str(run),status_path=str(run/'status.json'),
        progress_path=str(run/'status.json'),log_path=str(log),notes=['child_process_alive'])
    plan=dict(identity='current-plan',jobs=[job]);path=root/'plan.json';write_json(path,plan)
    write_json(root/'monitoring/lifecycle_guard.json',dict(plan_identity='current-plan',plan=str(path)))
    request=prepare(root,prefix,dict(jobs=[job,{**job,'job_id':'old'}]))
    assert request['scope_run_ids']==['current']
    assert not any(t['mode']=='tail' or Path(t['path']).name=='run.json' for t in request['targets'])
    job['notes']=['log_unchanged_over_15min_inspect_before_action']
    request=prepare(root,prefix,dict(jobs=[job]))
    assert any(t['mode']=='tail' for t in request['targets'])


def test_failed_stage_cannot_stop_patrol_before_successful_comparison(tmp_path):
    from monitor import publication_allows_stop
    job=dict(status_path=str(tmp_path/'job.json'));plan=dict(identity='plan',prefix='fusion',jobs=[job])
    write_json(tmp_path/'plan.json',plan)
    write_json(tmp_path/'monitoring/lifecycle_guard.json',dict(plan_identity='plan',plan=str(tmp_path/'plan.json'),completion_requires_success=True))
    write_json(tmp_path/'monitoring/publication.json',dict(status='published',all_terminal=True,plan_identities=['plan'],report_commit='a',verified_remote_commit='a'))
    write_json(tmp_path/'job.json',dict(status='failed'))
    assert not publication_allows_stop(tmp_path)
    write_json(tmp_path/'job.json',dict(status='completed'))
    assert not publication_allows_stop(tmp_path)
    write_json(tmp_path/'reports/fusion_comparison_fusion/summary.json',dict(status='completed'))
    assert publication_allows_stop(tmp_path)


def test_packed_token_validation_has_no_cwd_filesystem_dependency(monkeypatch):
    from purevision_v5.flow import validate_pair_tokens
    monkeypatch.setattr(Path,'exists',lambda *_:(_ for _ in ()).throw(AssertionError('No directory probes for token validation')))
    validate_pair_tokens('source','target')
    for pair in (('same','same'),('../escape','target'),('source','')):
        with pytest.raises(ValueError):validate_pair_tokens(*pair)
