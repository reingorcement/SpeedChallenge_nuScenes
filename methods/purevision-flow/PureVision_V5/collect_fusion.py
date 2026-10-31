"""Verified 40-extra-epoch FLEX continuation/frozen/FT comparison and cost record."""
import argparse
from pathlib import Path
import numpy as np
from collect_resolution import inspect_run
from collect_results import require
from fusion_comparison import NAMES, DIFFERENCES, validate_inputs
from lateral_baselines import baseline_report
from pipeline import validate_plan
from purevision_v5.common import now, read_json, write_json, sha256
from purevision_v5.evaluation import lateral_acceptance, summarize
from purevision_v5.selection import label_rows


def collect(plan_path):
    plan=read_json(plan_path);validate_plan(plan);validate_inputs(plan)
    train,_=label_rows(plan['manifest'],'train_core');dev,_=label_rows(plan['manifest'],'dev')
    rows={j['name']:inspect_run(plan,j,train,dev) for j in plan['jobs'] if j['kind']=='training'}
    require(all(r['status']=='completed' for r in rows.values()),f'Run verification failed: {rows}')
    reference=rows['R02_FLEX_CONT']
    common=lambda r:{k:v for k,v in r['config'].items() if k not in DIFFERENCES}
    for r in rows.values():
        r['fusion_numerical_files']={name:sha256(Path(plan['code_root'])/name) for name in
            ('purevision_v5/rgb.py','purevision_v5/rgb_frames.py','purevision_v5/initialization.py','purevision_v5/optimization.py',
             'purevision_v5/_rgb_reference/flexinet_backbone.py','purevision_v5/_rgb_reference/modules.py')}
        require(common(r)==common(reference),'Shared optimization differs')
        for key in ('target_identity','sample_order','epoch_order_sha256'):
            require(r[key]==reference[key],f'Unfair comparison: {key}')
        for key in ('dataset_sha256','label','flow','flow_root'):
            require(r['data'][key]==reference['data'][key],f'Common data differs: {key}')
        for key in ('flow_checkpoint_sha256','source_run','source_training_commit','encoder_state_sha256'):
            require(r['initialization'][key]==reference['initialization'][key],f'Common flow starting state differs: {key}')
        require(r['initialization']['optimizer_inherited'] is False,'Old optimizer was inherited')
        r['end_to_end_train_windows_per_second']=len(train)*40/r['elapsed_seconds']
        r['cost_note']='Wall time includes validation/checkpoint IO; cost is recorded, not an adoption gate'
    frozen,ft=rows['R03_Frozen'],rows['R03_FT']
    for key in ('head_state_sha256','rgb_state_sha256','rgb_checkpoint_sha256'):
        require(frozen['initialization'][key]==ft['initialization'][key],f'Frozen/FT initial {key} differs')
    require(frozen['final_rgb_state_sha256']==frozen['initialization']['rgb_state_sha256'],'Frozen RGB weights/BN changed')
    require(ft['final_rgb_state_sha256']!=ft['initialization']['rgb_state_sha256'],'FT RGB state never changed')
    rule=reference['config']['lateral_rule']
    baseline=baseline_report(plan['manifest'],rule=rule,predictions=[r['predictions'] for r in rows.values()])
    median=np.median([[r['target_vx'],r['target_vy']] for r in train],axis=0)
    target=np.asarray([[r['target_vx'],r['target_vy']] for r in dev])
    baseline['train_median_xy']=dict(fit_partition='train_core',constant_m_s=median.tolist(),
        metrics=summarize(np.broadcast_to(median,target.shape),target,[r['scene'] for r in dev],rule))
    comparisons={}
    for candidate,base in (('R03_Frozen','R02_FLEX_CONT'),('R03_FT','R03_Frozen'),('R03_FT','R02_FLEX_CONT')):
        a,b=rows[candidate],rows[base]
        comparisons[f'{candidate}_vs_{base}']=dict(candidate=candidate,reference=base,status='comparable',
            delta_candidate_minus_reference={k:a['metrics']['overall'][k]-b['metrics']['overall'][k]
                for k in ('epe','mae_vx','mae_vy','rmse_vx','rmse_vy','bias_vx','bias_vy')},
            improved_scenes=sum(a['metrics']['per_scene'][s]['epe']<v['epe'] for s,v in b['metrics']['per_scene'].items()),
            total_scenes=103,lateral_acceptance=lateral_acceptance(a['metrics'],b['metrics'],rule))
    winner=min(rows,key=lambda n:rows[n]['metrics']['overall']['epe'])
    return dict(schema='v5_fusion_comparison_v1',status='completed',generated_at=now(),plan_identity=plan['identity'],
        training_commit=plan['code']['commit'],scope='full_train_dev_PLANAR_PROXY_single_seed',counts=plan['expected_counts'],
        rows=rows,baselines=baseline,comparisons=comparisons,
        recommendation=f'{winner} has the lowest seed42 dev EPE; validate with paired seed43/44 before claiming stable superiority',
        limits=['All three start from seed42 R02_FLEX best and receive the same additional 40 epochs',
                'Dev selection and dev evaluation are not independent held-out performance',
                '31 independent nonzero-vy windows / 5 scenes: insufficient_data; unchanged 100-window/5-scene threshold',
                'Original scene partition retained; not log-disjoint. Labels are PLANAR_PROXY, not real body-axis ground truth',
                'Accuracy priority: parameters, throughput and memory are observations, not selection gates'])


def markdown(report):
    lines=['# FLEX + RGB: same-start, 40 additional epochs','',f"Training commit: `{report['training_commit']}`",'',
        '| Run | EPE | vx MAE | vy MAE | vx RMSE | vy RMSE | vx bias | vy bias | Updates | AMP skips |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        r=report['rows'][name];m=r['metrics']['overall']
        values=[name]+[f'{m[k]:.6f}' for k in ('epe','mae_vx','mae_vy','rmse_vx','rmse_vy','bias_vx','bias_vy')]
        values += [str(r['optimizer_updates']),str(r['amp_skips'])]
        lines.append('| '+' | '.join(values)+' |')
    lines += ['',report['recommendation'],'']
    for name,c in report['comparisons'].items():
        lines.append(f"- {name}: EPE delta {c['delta_candidate_minus_reference']['epe']:.6f} m/s; improved {c['improved_scenes']}/103 scenes.")
    lines += ['','Per-scene metrics, lateral subsets, zero-vy / train-median baselines, full costs and hashes: summary.json.',
        'Per-window predictions and actual configurations: experiments/ corresponding run directories.','']
    lines += ['- '+limit for limit in report['limits']]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=collect(a.plan);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'summary.json',result);(out/'summary.md').write_text(markdown(result));print(result['recommendation'],flush=True)
