"""Verify real training artifacts and summarize completed local algorithm runs."""
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.answer_verifier import verify_response_answer
ARTIFACTS = ROOT / 'tests/artifacts/gsm8k'
ALGORITHMS = ('r2opl_base', 'pg_opd', 'eopd', 'opdvr', 'grpo', 'gspo')


def summarize(name):
    directory = ARTIFACTS / f'{name}_gsm8k_2steps'
    result_path = directory / 'result.json'
    if not result_path.exists():
        return {'algorithm': name, 'status': 'not_completed'}
    result = json.loads(result_path.read_text())
    if result['exit_code'] != 0 or result['stop_reason'] is not None:
        return dict(result, status='failed')
    runtime = result.get('runtime', {})
    if runtime.get('conda_env') != 'r2opl-cu12' or runtime.get('vllm') != '0.19.1':
        return dict(result, status='requires_r2opl_cu12_rerun')
    try:
        outcomes = {}
        for relative in ('rollouts/1.jsonl', 'rollouts/2.jsonl', 'validation/2.jsonl'):
            rows = [json.loads(line) for line in (directory / relative).read_text().splitlines() if line.strip()]
            assert len(rows) == 8, f'{relative}: expected 8 rows'
            assert sorted(Counter(row['input'] for row in rows).values()) == [4, 4], relative
            assert all(row['output'].strip() for row in rows), f'{relative}: empty generation'
            if relative.startswith('validation/') or name in {'r2opl_base', 'opdvr', 'grpo', 'gspo'}:
                expected = [float(verify_response_answer(row['output'], str(row['gts']))) for row in rows]
                assert [float(row['score']) for row in rows] == expected, f'{relative}: verifier score mismatch'
                outcomes[relative] = {'correct': sum(expected), 'incorrect': len(rows) - sum(expected)}
        assert (directory / 'checkpoints/latest_checkpointed_iteration.txt').read_text().strip() == '2'
        assert list((directory / 'checkpoints/global_step_2/actor').glob('model*.pt'))
        text = (directory / 'train.log').read_text(errors='replace')
        step_lines = [line for line in text.splitlines() if 'training/global_step:' in line]
        assert len(step_lines) == 2
        timings = []
        for line in step_lines:
            metrics = {}
            for key, value in re.findall(r'([\w/@.-]+):(?:np\.float64\()?(-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)', line):
                metrics[key] = float(value)
            for field in ('timing_s/gen', 'timing_s/update_actor', 'timing_s/update_weights'):
                assert math.isfinite(metrics[field]) and metrics[field] > 0, field
            timings.append({key: metrics[key] for key in (
                'training/global_step', 'timing_s/gen', 'timing_s/update_actor', 'timing_s/update_weights')})
        assert [item['training/global_step'] for item in timings] == [1, 2]
        assert result['minimum_free_mib'] >= 3072
        return dict(result, status='passed', train_rows=[8, 8], validation_rows=8, timings=timings, outcomes=outcomes)
    except (AssertionError, OSError, KeyError, ValueError) as error:
        return dict(result, status='verification_failed', verification_error=str(error))


if __name__ == '__main__':
    summary = [summarize(name) for name in ALGORITHMS]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if all(row['status'] == 'passed' for row in summary) else 1)
