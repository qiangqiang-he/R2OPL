"""Run the real training entrypoint and enforce the local desktop VRAM reserve."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import psutil

parser = argparse.ArgumentParser()
parser.add_argument('algorithm', choices=['r2opl_base', 'pg_opd', 'eopd', 'opdvr', 'grpo', 'gspo'])
parser.add_argument('--probe', action='store_true', help='Force short R2OPL rollouts to exercise truncation probes')
args = parser.parse_args()
if args.probe and args.algorithm != 'r2opl_base':
    parser.error('--probe is only supported for r2opl_base')
config_name = args.algorithm + ('_probe' if args.probe else '')
runtime = {'conda_env': Path(sys.prefix).name, 'python': sys.version.split()[0],
           **{name: version(name) for name in ('torch', 'vllm', 'transformers', 'ray', 'TransferQueue')}}
if runtime['conda_env'] != 'r2opl-cu12' or runtime['vllm'] != '0.19.1':
    raise SystemExit('Local training tests require conda activate r2opl-cu12 with vLLM 0.19.1')
root = Path(__file__).resolve().parents[1]
out = root / 'tests/artifacts/gsm8k' / f'{config_name}_gsm8k_2steps'
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', RAY_DEDUP_LOGS='0',
           VERL_LOGGING_LEVEL='INFO', OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
log_path = out / 'train.log'
if log_path.exists():
    previous = out / 'attempts' / str(time.time_ns())
    previous.mkdir(parents=True)
    for name in ('train.log', 'result.json', 'gpu_memory.jsonl', 'rollouts', 'validation', 'checkpoints'):
        if (out / name).exists():
            (out / name).rename(previous / name)
owned = {}
minimum_free = float('inf')
reason = None
start = time.monotonic()
started_wall = time.time()
last_progress = None
last_step = 0
last_notice = start
seen_rollout_events = 0
seen_outputs = set()
with log_path.open('w') as log, (out / 'gpu_memory.jsonl').open('w') as telemetry:
    process = subprocess.Popen(['bash', 'scripts/start_train.sh', f'tests/configs/gsm8k/{config_name}.yaml'],
                               cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'Started {args.algorithm}: PID {process.pid}, log {log_path}', flush=True)
    try:
        while process.poll() is None:
            for child in psutil.Process(process.pid).children(recursive=True):
                try:
                    owned[child.pid] = child.create_time()
                except psutil.NoSuchProcess:
                    pass
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
            free = int(memory.strip().splitlines()[0])
            minimum_free = min(minimum_free, free)
            now = time.monotonic()
            telemetry.write(json.dumps({'elapsed_s': round(now-start, 1), 'free_mib': free}) + '\n')
            telemetry.flush()
            # Stop before the requested 3 GiB floor, allowing headroom for cleanup.
            if free < 4096:
                reason = f'VRAM reserve guard: only {free} MiB free (stop threshold 4096 MiB)'
                break
            text = log_path.read_text(errors='replace')
            # Ray does not always forward worker INFO messages to driver stdout.
            for worker_log in Path('/tmp/ray/session_latest/logs').glob('worker-*.err'):
                if worker_log.stat().st_mtime >= started_wall:
                    text += '\n' + worker_log.read_text(errors='replace')
            events = list(re.finditer(r'Rollout batch submitted:|Rollout group completed:', text))
            if len(events) > seen_rollout_events:
                last_progress = now
                seen_rollout_events = len(events)
            steps = [int(value) for value in re.findall(r'training/global_step:(\d+)\b', text)]
            completed = max(steps, default=0)
            if completed > last_step:
                last_progress, last_step = now, completed
            # Persisted generations also prove progress before the optimizer
            # and console metric line finish, even when Ray suppresses INFO.
            outputs = set(out.glob('rollouts/*.jsonl')) | set(out.glob('validation/*.jsonl'))
            if outputs - seen_outputs:
                last_progress = now
                seen_outputs.update(outputs)
            if last_progress is not None and now - last_progress > 180:
                reason = 'No completed training/rollout progress for 180 seconds; investigate before retrying'
                break
            if now - start > 180 and last_progress is None:
                reason = 'No first rollout within 180 seconds of launch; stopped for investigation'
                break
            if now - last_notice >= 30:
                print(f'{args.algorithm}: elapsed {now-start:.0f}s, free VRAM {free} MiB, completed step {last_step}', flush=True)
                last_notice = now
            time.sleep(2)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        # Ray may spawn processes in separate process groups. Restrict cleanup
        # to descendants observed during this run, verifying PID creation time.
        for pid, created in owned.items():
            try:
                child = psutil.Process(pid)
                if child.create_time() == created:
                    child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.wait()
result = {'algorithm': args.algorithm, 'exit_code': process.returncode, 'stop_reason': reason,
          'minimum_free_mib': minimum_free, 'elapsed_s': round(time.monotonic()-start, 1), 'runtime': runtime}
if process.returncode == 0 and reason is None:
    try:
        rows = {}
        for relative in ('rollouts/1.jsonl', 'rollouts/2.jsonl', 'validation/2.jsonl'):
            records = [json.loads(line) for line in (out / relative).read_text().splitlines() if line.strip()]
            assert len(records) == 8, f'{relative}: expected 8 outputs, got {len(records)}'
            rows[relative] = len(records)
        assert (out / 'checkpoints/latest_checkpointed_iteration.txt').read_text().strip() == '2'
        assert list((out / 'checkpoints/global_step_2/actor').glob('model*.pt'))
        completed_steps = set(re.findall(r'training/global_step:(\d+)\b', log_path.read_text(errors='replace')))
        assert completed_steps == {'1', '2'}, f'Unexpected completed steps: {completed_steps}'
        result['verified_outputs'] = rows
        result['verified_steps'] = [1, 2]
    except (AssertionError, OSError, ValueError) as error:
        reason = result['stop_reason'] = f'Output verification failed: {error}'
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
if process.returncode == 0 and reason is None:
    from summarize_gsm8k_formal import summarize
    verification = summarize(args.algorithm, directory=out)
    result['semantic_verification'] = {key: verification[key] for key in ('status', 'outcomes') if key in verification}
    if verification['status'] != 'passed':
        reason = result['stop_reason'] = verification.get('verification_error', verification['status'])
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
raise SystemExit(1 if reason else process.returncode)
