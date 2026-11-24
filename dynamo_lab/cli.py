"""CLI for comparing routing strategies under identical workload settings."""
import argparse
import csv
import json
from pathlib import Path
from .simulator import Settings, simulate, workload


def main() -> None:
    parser = argparse.ArgumentParser(description='NVIDIA Dynamo-inspired routing experiment (synthetic)')
    parser.add_argument('--requests', type=int, default=240)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--groups', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arrival-spacing-ms', type=float, default=6.0)
    parser.add_argument('--failure-at', type=int, default=None)
    parser.add_argument('--output', default='results/comparison.json')
    args = parser.parse_args()
    s = Settings(requests=args.requests, workers=args.workers, groups=args.groups,
                 seed=args.seed, arrival_spacing_ms=args.arrival_spacing_ms,
                 failure_at=args.failure_at)
    tasks = workload(s)
    runs = [simulate(policy, s, tasks) for policy in ('round_robin', 'cache_only', 'balanced')]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'article': 'https://developer.nvidia.com/blog/reducing-rollout-bottlenecks-in-reinforcement-learning-with-nvidia-dynamo',
                                  'note': 'All times modeled; no real NVIDIA GPUs or Dynamo measured.',
                                  'runs': runs}, indent=2), encoding='utf-8')
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['policy','completed','cache_hit_rate','mean_latency_ms','p95_latency_ms','makespan_ms','throughput_requests_per_sim_second','worker_failures_injected'])
        writer.writeheader()
        for run in runs:
            writer.writerow({'policy': run['policy'], **{k: run['summary'][k] for k in writer.fieldnames if k != 'policy'}})
    print(f'Experiment: {s.requests} requests, {s.workers} workers, seed {s.seed}, failure_at={s.failure_at}')
    for run in runs:
        m = run['summary']
        print(f"{run['policy']:12s} hit={m['cache_hit_rate']:6.1%}  p95={m['p95_latency_ms']:9.1f}ms  makespan={m['makespan_ms']:9.1f}ms  throughput={m['throughput_requests_per_sim_second']:7.1f}/s")
    print(f'Wrote {output} and {output.with_suffix(".csv")}')

if __name__ == '__main__':
    main()
