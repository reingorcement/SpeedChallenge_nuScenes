"""Print later launch commands by default. Execution requires --execute.

No queue or cron entries are installed. Run cache stage first, then train stage
only after an explicit later authorization and a complete common pilot cache.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

from purevision_v5.common import ROOT, code_version, now, sha256, write_json

ART=Path('/home/zhangyu/.local/share/speedest-v5-next')
PY='/home/zhangyu/.conda/envs/speedest/bin/python'
GPU={0:'GPU-cedbf5f4-647b-640c-1244-babecaef88e7',
     1:'GPU-5dbd27d5-0ae2-0ddc-ec5b-2077e99643e2',
     2:'GPU-b2986926-3ad1-16bc-f86c-28aae0c46598'}
MANIFEST='/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/data/nuscenes_proxy_v1/dataset.json'
FLOW='/mnt/sdb2/Workspace/zhangyu/dataset/nuscenes/derived/optical_flow/raft_large_C_T_SKHT_V2_1024x576_fwd_v1_20261030/packed_train_core_dev'
SOURCE=ART/'runs/20261101_rgb_path_s42_R03_APPEAR_FT_s42_18ep/best.pt'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['cache','train'],required=True)
    p.add_argument('--cache',required=True);p.add_argument('--run-root',required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();cache=Path(a.cache).resolve();runs=Path(a.run_root).resolve()
    if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.run_id):
        p.error('run-id must be letters/digits/underscore/hyphen')
    version=code_version(require_pushed=a.execute)
    if cache.is_relative_to(ROOT.parents[2]) or runs.is_relative_to(ROOT.parents[2]) or cache==runs:
        p.error('Cache/run directories must be external and distinct')
    if a.stage=='cache':
        entries=[('cache',0,[PY,'-B',str(ROOT/'fusion_abc_extract.py'),
            '--checkpoint',str(SOURCE),'--manifest',MANIFEST,
            '--selection',str(ART/'plans/resolution_pilot_s42_32t8v.json'),
            '--flow',FLOW,'--rgb-frames',str(ART/'rgb_frames/v3_uint8_train_dev_v1'),
            '--scope','pilot','--allow-pilot-cache','--batch-size','2',
            '--device','cuda:0','--output',str(cache)])]
    else:
        if a.execute:
            from purevision_v5.fusion_abc_cache import load_cache
            for part in ['train_core','dev']:load_cache(cache,part)
        entries=[]
        for variant,gpu in [('C',0),('B',1),('A',2)]:
            out=runs/f'{a.run_id}_{variant}'
            cmd=[PY,'-B',str(ROOT/'fusion_abc_train.py'),'--config',str(ROOT/f'configs/next/FUSION_ABC_{variant}.json'),
                 '--cache',str(cache),'--output',str(out),'--device','cuda:0']
            entries.append((variant,gpu,cmd))
    records=[]
    for label,gpu,cmd in entries:
        session=f'v5_abc_{a.run_id}_{label}'
        log=runs/f'{a.run_id}_{label}.log'
        envcmd=['env',f'CUDA_VISIBLE_DEVICES={GPU[gpu]}',*cmd]
        shellcmd=shlex.join(envcmd)+' >> '+shlex.quote(str(log))+' 2>&1'
        if a.stage=='train':
            # Re-evaluate both partitions after successful training; no automatic new experiment.
            out=runs/f'{a.run_id}_{label}'
            for part in ['dev','train_core']:
                evalcmd=['env',f'CUDA_VISIBLE_DEVICES={GPU[gpu]}',PY,'-B',str(ROOT/'fusion_abc_evaluate.py'),
                    '--checkpoint',str(out/'best.pt'),'--cache',str(cache),'--partition',part,
                    '--output',str(out/f'eval_{part}'),'--device','cuda:0','--require-completed-run',str(out)]
                shellcmd+=' && '+shlex.join(evalcmd)+' >> '+shlex.quote(str(log))+' 2>&1'
        tmuxcmd=['tmux','new-session','-d','-s',session,'-c',str(ROOT),shellcmd]
        print(shlex.join(tmuxcmd),flush=True)
        records.append(dict(stage=a.stage,group=label,physical_gpu=gpu,gpu_uuid=GPU[gpu],
                            process_device='cuda:0',session=session,log=str(log),tmux_command=tmuxcmd))
    if not a.execute:
        print('DRY RUN ONLY: no cache, training, tmux or scheduler created.');return
    # Check all destinations and GPU identities before launching any group.
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    for row in records:
        if f"{row['physical_gpu']}, {row['gpu_uuid']}" not in inventory:raise ValueError('Physical GPU/UUID mapping changed')
        if subprocess.run(['tmux','has-session','-t',row['session']],capture_output=True).returncode==0:
            raise ValueError('tmux session already exists')
        destination=cache if a.stage=='cache' else runs/f"{a.run_id}_{row['group']}"
        if destination.exists() and any(destination.iterdir()):raise ValueError('Output already populated')
        if Path(row['log']).exists():raise ValueError('Log already exists; use a new run ID')
    runs.mkdir(parents=True,exist_ok=True)
    receipt=runs/f'{a.run_id}_{a.stage}_launch.json'
    if receipt.exists():raise ValueError('Launch receipt already exists')
    write_json(receipt,dict(status='prepared',time=now(),code=version,entries=records,
                           scheduler_changes=False,script_sha256=sha256(__file__)))
    for row in records:subprocess.run(row['tmux_command'],check=True)
    write_json(receipt,dict(status='launched_processes_not_verified_training',time=now(),code=version,
                           entries=records,scheduler_changes=False,script_sha256=sha256(__file__)))

if __name__=='__main__':main()
