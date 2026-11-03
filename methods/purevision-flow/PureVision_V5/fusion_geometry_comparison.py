"""Read-only matched S23 geometry decisions from saved full-window predictions."""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import torch
from purevision_v5.common import read_json,write_json,sha256,object_hash,now
from purevision_v5.fusion_geometry_training import load_recovery,validate_config
from fusion_grid_comparison import (read_history,read_predictions,prediction_identity,summarize_rows,expected_rows,paired_comparison,difference)
from fusion_confirmation_comparison import lateral_diagnostics
from fusion_multiseed_comparison import atomic_csv,atomic_text


def inspect(plan,job):
    root=Path(job['run_dir']);path=root/'result.json'
    if not path.exists():path=root/'dev_ready.json'
    if not path.exists():return None,None,None
    result=read_json(path);run=read_json(root/'run.json');config=read_json(root/'config.json');validate_config(config)
    if result['status'] not in {'completed','dev_ready'}:raise ValueError('Preflight or invalid result cannot be collected as formal')
    ntrain,ndev=(7324,1826) if job['scope']=='pilot' else (132978,23480)
    updates=2300 if job['scope']=='pilot' else 41560
    if (config!=read_json(job['config_path']) or config['variant']!=job['variant'] or config['scope']!=job['scope']
            or run['code']['commit']!=plan['code']['commit'] or run['code']['dirty']
            or result['optimizer_updates']!=updates or result['completed_epochs']!=20
            or run['contract_hash']!=result['contract_hash'] or object_hash(run['contract'])!=run['contract_hash']):
        raise ValueError('Formal source/config/budget identity differs')
    recovery,loaded=load_recovery(root,run['contract_hash'])
    if recovery['completed_epochs']!=20 or loaded['current'][1]['optimizer_updates']!=updates:
        raise ValueError('Not a completed whole-epoch recovery')
    history=read_history(root/'metrics.jsonl')
    if len(history)!=20 or any(row['optimizer_updates']!=(i+1)*(updates//20) for i,row in enumerate(history)):
        raise ValueError('Epoch history/update budget differs')
    best=min(history,key=lambda x:x['dev']['overall']['epe'])
    if result['best_epoch']!=best['epoch'] or result['best_dev_epe']!=best['dev']['overall']['epe']:
        raise ValueError('Checkpoint selection or earliest tie rule differs')
    for name,label in (('best','best'),('last','current')):
        if sha256(root/f'{name}.pt')!=recovery[label]['sha256']:raise ValueError('Checkpoint link hash mismatch')
    index=read_json(Path(job['cache_root'])/'index.json');predictions={};summaries={}
    for part,count in (('dev',ndev),('train_core',ntrain)):
        if part=='train_core' and result['status']=='dev_ready':continue
        folder=root/f'eval_{part}';proof=read_json(folder/'evaluation.json');rows=read_predictions(folder/'predictions.csv')
        if (proof['status']!='completed' or proof['count']!=count or len(rows)!=count
                or proof['checkpoint_sha256']!=result['best_checkpoint_sha256']
                or proof['contract_hash']!=run['contract_hash'] or sha256(folder/'predictions.csv')!=proof['predictions_sha256']
                or prediction_identity(rows)!=expected_rows(job['cache_root'],index,part)):
            raise ValueError('Evaluation source, all-window coverage or actual identity differs')
        summary=summarize_rows(rows,config)
        expected=result['dev' if part=='dev' else 'train']['overall']['epe']
        if not math.isclose(summary['overall']['epe'],expected,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Saved prediction EPE mismatch')
        predictions[part]=rows;summaries[part]=summary
    value={k:result.get(k) for k in ('best_epoch','best_dev_epe','last_dev_epe','best_train_epe','trainable_parameters',
        'optimizer_updates','elapsed_seconds','best_checkpoint_sha256','last_checkpoint_sha256','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')}
    value.update(name=job['name'],run_id=job['job_id'],status=result['status'],scope=job['scope'],variant=job['variant'],
        dev=summaries['dev'],train=summaries.get('train_core'),lateral=lateral_diagnostics(predictions['dev']),
        initialization=run['initialization'],code=run['code'],geometry=run.get('geometry'),
        gap=None if result.get('best_train_epe') is None else result['best_dev_epe']-result['best_train_epe'],
        actual_gpu=run['runtime_gpu'],checkpoint_paths=dict(best=str(root/'best.pt'),last=str(root/'last.pt')))
    return value,history,predictions


def pilot_decision(a,b,ha,hb,proof):
    """No dev weight selection: one inclusive1% expansion gate only."""
    checks=dict(geometric_preflight=proof['status']=='passed',
        same_initialization=a['initialization']['state_sha256']==b['initialization']['state_sha256'],
        finite_training=all(math.isfinite(x['training_loss']) and math.isfinite(x['geometry_loss']) for x in ha+hb),
        actual_geometric_coverage=all(x['geometry_valid_window_fraction']>0 for x in hb),
        no_invalid_projected_depth=all(x['geometry_nonpositive_predicted_depth']==0 for x in hb),
        within_tolerance=b['best_dev_epe']<=a['best_dev_epe']*1.01)
    return dict(status='passed' if all(checks.values()) else 'rejected',checks=checks,
        pilot_A_best=a['best_dev_epe'],pilot_B_best=b['best_dev_epe'],limit=a['best_dev_epe']*1.01,
        rule='B best all-dev EPE <= A best all-dev EPE *1.01; one pilot only; tolerance is not evidence of benefit')


def collect(plan_path):
    plan=read_json(plan_path)
    if plan['identity']!=object_hash({k:v for k,v in plan.items() if k!='identity'}):raise ValueError('Plan identity mismatch')
    output=Path(plan['report_root']);output.mkdir(parents=True,exist_ok=True)
    rows={};histories={};predictions={};progress={};comparisons={}
    for job in plan['jobs']:
        progress[job['name']]=read_json(job['status_path'])
        value,history,pred=inspect(plan,job)
        if value is not None:rows[job['name']]=value;histories[job['name']]=history;predictions[job['name']]=pred
    for scope in ('pilot','full'):
        an,bn=f'{scope.upper()}_A',f'{scope.upper()}_B'
        if an in rows and bn in rows:
            a,b=rows[an],rows[bn]
            if (a['initialization']['state_sha256']!=b['initialization']['state_sha256']
                    or [x['actual_sample_order_sha256'] for x in histories[an]]!=[x['actual_sample_order_sha256'] for x in histories[bn]]):
                raise ValueError('A/B actual initialization or effective batches differ')
            pair=paired_comparison(bn,an,rows)
            pair['speed_bins']={key:dict(count=item['count'],metrics=difference(b['dev']['speed_bins'][key]['metrics'],item['metrics']))
                for key,item in a['dev']['speed_bins'].items() if item['count']}
            comparisons[scope]=pair
    gate=read_json(plan['pilot_gate']) if Path(plan['pilot_gate']).exists() else None
    complete=all(x['status'] in {'completed','blocked','failed','cancelled','interrupted'} for x in progress.values())
    value=dict(schema='v5_s23_geometry_comparison_v1',status='completed' if complete else 'partial',observed_at=now(),
        experiment_id=plan['prefix'],plan_identity=plan['identity'],training_commit=plan['code']['commit'],
        progress=progress,runs=rows,comparisons=comparisons,pilot_gate=gate,
        lambda_geo=read_json(plan['lambda_proof'])['lambda_geo'],geometry_preflight=read_json(plan['geometry_preflight']),
        limitations=['Single seed42; no cross-seed stability claim.','Offline orientations, frozen pretrained semantics and geometric priors are extra uncertain training supervision.',
            'PLANAR_PROXY is ground-projected rear-axle yaw-plane velocity, not CG or certified true sideslip.',
            'All dev windows are the primary metric; only training has a geometric cache. Dev geometry-valid-region attribution is unavailable.',
            'Above15m/s dev count remains11; repeated predictions are not independent labels.'])
    write_json(output/'summary.json',value)
    core=[];axes=[];scene=[];speed=[];lateral=[];curves=[]
    for name,row in rows.items():
        core.append({k:v for k,v in row.items() if k in ('name','scope','variant','status','best_epoch','best_dev_epe','last_dev_epe','best_train_epe','gap','optimizer_updates','elapsed_seconds','trainable_parameters')})
        axes.append(dict(name=name,**row['dev']['overall']))
        for sc,metrics in row['dev']['per_scene'].items():scene.append(dict(name=name,scene=sc,**metrics))
        for label,item in row['dev']['speed_bins'].items():
            speed.append(dict(name=name,speed_bin=label,**(item['metrics'] or {})))
            speed[-1]['count']=item['count']
        for label,item in row['lateral']['subsets'].items():lateral.append(dict(name=name,subset=label,**item))
        for item in histories[name]:curves.append(dict(name=name,epoch=item['epoch'],training_loss=item['training_loss'],geometry_loss=item['geometry_loss'],dev_epe=item['dev']['overall']['epe'],geometry_valid_fraction=item['geometry_valid_window_fraction']))
    for filename,data in (('core_results',core),('axis_errors',axes),('per_scene',scene),('speed_bins',speed),('lateral',lateral),('learning_curves',curves)):
        if data:
            columns=list(dict.fromkeys(k for row in data for k in row))
            atomic_csv(output/f'{filename}.csv',[{k:row.get(k) for k in columns} for row in data])
    lines=['# S23几何训练约束 A/B',f"状态：{value['status']}；固定提交 `{plan['code']['commit']}`。",'',
        '本轮比较原速度监督与加入离线姿态、冻结道路语义及几何先验的训练约束；部署输入和S23结构保持一致。',
        f"固定 λ={value['lambda_geo']:.9g}，只由pilot train初始梯度标定一次。",'',
        '| scope/组 | best轮 | best dev | last dev | best train | dev−train |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        lines.append(f"| {name} | {r['best_epoch']} | {r['best_dev_epe']:.6f} | {r['last_dev_epe']:.6f} | {r['best_train_epe'] if r['best_train_epe'] is not None else 'pending'} | {r['gap']} |")
    for scope,pair in comparisons.items():
        delta=pair['metrics']['epe'];high=pair['speed_bins'].get('10..15',{})
        contributions=pair['scene_contribution'];top=sorted(contributions['weighted_gain_m_s'].items(),key=lambda x:x[1],reverse=True)[:3]
        share=sum(x[1] for x in top)/contributions['total_gain_m_s'] if contributions['total_gain_m_s'] else None
        lines+=['',f"{scope} B−A：ΔEPE={delta['delta']:.6f} m/s（{delta['relative_percent']:.3f}%）；改善/退化场景={pair['improved_scenes']}/{pair['degraded_scenes']}。",
            f"10–15m/s：{high}。",f"前三正贡献场景：{top}；占净收益比例={share}（其他场景退化抵消时可超过100%）。"]
    if gate:lines+=['',f"pilot自动扩大判断：{gate['status']}；{gate['checks']}。1%是筛选容忍度。"]
    if 'full' in comparisons:
        pair=comparisons['full'];improves=pair['metrics']['epe']['delta']<0
        lines+=['',('完整集seed42：几何B优于匹配A。可考虑seed43/44匹配复核，须结合场景与速度段代价审阅，未自动启动。' if improves else
            '完整集seed42：几何B未优于匹配A。不建议直接扩展seed，先审阅具体退化及几何近似。')]
    lines+=['','完整两轴、速度段、全部场景及非零/正负/近零侧向结果见CSV；侧向充分性沿用原门槛。',
        '历史原S23 seed42 best19 dev≈0.178472，TAIL_FT≈0.178169，仅作协议不同的历史参照。','',*['- '+x for x in value['limitations']]]
    atomic_text(output/'summary.md','\n\n'.join(lines)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);a=p.parse_args();v=collect(a.plan)
    print('Report',v['status'],list(v['runs']))
