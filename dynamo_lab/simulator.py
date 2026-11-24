"""Discrete-event approximation, NOT GPU or NVIDIA Dynamo execution.

All durations are synthetic milliseconds. Work is scheduled in fixed input order;
workers execute requests serially and retain a bounded prefix cache.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from statistics import mean
import random


@dataclass(frozen=True)
class Request:
    id: int
    prefix: str
    input_tokens: int
    output_tokens: int
    arrival_ms: float
    group: int


@dataclass
class Worker:
    id: int
    next_free_ms: float = 0.0
    cache: tuple[str, ...] = ()
    healthy: bool = True
    completed: int = 0


@dataclass(frozen=True)
class Settings:
    workers: int = 4
    requests: int = 240
    groups: int = 12
    seed: int = 42
    arrival_spacing_ms: float = 6.0
    failure_at: int | None = None
    failed_worker: int = 0
    replacement_delay_ms: float = 250.0
    cache_slots: int = 3
    cache_discount: float = 0.85
    prefill_ms_per_token: float = 0.12
    decode_ms_per_token: float = 0.55
    weight_wait: float = 0.0


def workload(s: Settings) -> list[Request]:
    if s.workers < 1 or s.groups < 1 or s.requests < 1:
        raise ValueError('workers, groups, and requests must be >= 1')
    if s.arrival_spacing_ms < 0 or s.cache_slots < 0 or not 0 <= s.cache_discount <= 1:
        raise ValueError('invalid timing/cache parameters')
    rng = random.Random(s.seed)
    result = []
    for i in range(s.requests):
        # Bursts of related prompts mimic repeated RL attempts on the same task.
        group = (i // 8) % s.groups
        result.append(Request(
            id=i, prefix=f'prompt-{group}', input_tokens=rng.randint(320, 850),
            output_tokens=rng.randint(24, 220),
            arrival_ms=i * s.arrival_spacing_ms, group=group,
        ))
    return result


def simulate(policy: str, s: Settings, requests: list[Request] | None = None) -> dict:
    """Policy is rr, cache_only, or balanced. Failure is injected before a request.

    A failed worker loses its cache, in-flight work would need replay in a real
    server (not modeled). New work is rerouted until its replacement is ready.
    """
    if policy not in ('round_robin', 'cache_only', 'balanced'):
        raise ValueError('unknown routing policy')
    if not 0 <= s.failed_worker < s.workers:
        raise ValueError('failed_worker out of range')
    tasks = requests if requests is not None else workload(s)
    nodes = [Worker(i) for i in range(s.workers)]
    rows = []
    hits = 0
    prefill_compute_ms = 0.0
    rr_cursor = 0
    replacement_at = None
    failures = 0
    for index, req in enumerate(tasks):
        if s.failure_at is not None and index == s.failure_at:
            if s.workers == 1:
                raise ValueError('worker failure requires at least 2 workers')
            victim = nodes[s.failed_worker]
            victim.healthy = False
            victim.cache = ()
            replacement_at = req.arrival_ms + s.replacement_delay_ms
            failures += 1
        if replacement_at is not None and req.arrival_ms >= replacement_at:
            victim = nodes[s.failed_worker]
            victim.healthy = True
            victim.cache = ()
            victim.next_free_ms = max(victim.next_free_ms, replacement_at)
            replacement_at = None
        eligible = [w for w in nodes if w.healthy]
        if not eligible:
            raise RuntimeError('no healthy workers')
        def forecast(w: Worker) -> float:
            cached = req.prefix in w.cache
            prefill = req.input_tokens * s.prefill_ms_per_token * (1 - s.cache_discount if cached else 1)
            decode = req.output_tokens * s.decode_ms_per_token
            return max(w.next_free_ms, req.arrival_ms) + prefill + decode
        if policy == 'round_robin':
            w = eligible[rr_cursor % len(eligible)]
            rr_cursor += 1
        elif policy == 'cache_only':
            cached_workers = [w for w in eligible if req.prefix in w.cache]
            w = min(cached_workers, key=lambda n: n.id) if cached_workers else min(eligible, key=lambda n: n.id)
        else:
            w = min(eligible, key=lambda n: (forecast(n), n.id))
        cached = req.prefix in w.cache
        hits += int(cached)
        prefill = req.input_tokens * s.prefill_ms_per_token * (1-s.cache_discount if cached else 1)
        prefill_compute_ms += prefill
        decode = req.output_tokens * s.decode_ms_per_token
        start = max(w.next_free_ms, req.arrival_ms)
        finish = start + prefill + decode
        w.next_free_ms = finish
        w.completed += 1
        if s.cache_slots > 0:
            updated = [v for v in w.cache if v != req.prefix] + [req.prefix]
            w.cache = tuple(updated[-s.cache_slots:])
        rows.append({'id': req.id, 'group': req.group, 'worker': w.id,
                     'prefix_hit': cached, 'arrival_ms': round(req.arrival_ms, 3),
                     'start_ms': round(start, 3), 'finish_ms': round(finish, 3),
                     'latency_ms': round(finish - req.arrival_ms, 3),
                     'queue_wait_ms': round(start - req.arrival_ms, 3)})
    ordered_lat = sorted(r['latency_ms'] for r in rows)
    def percentile(q: float) -> float:
        return round(ordered_lat[int((len(ordered_lat)-1)*q)], 3)
    makespan = max(r['finish_ms'] for r in rows) - min(r['arrival_ms'] for r in rows)
    return {
        'policy': policy, 'environment': 'deterministic synthetic simulation; no GPU execution',
        'settings': asdict(s),
        'summary': {
            'completed': len(rows), 'worker_failures_injected': failures,
            'cache_hits': hits, 'cache_hit_rate': round(hits / len(rows), 4),
            'prefill_compute_ms': round(prefill_compute_ms, 3),
            'mean_latency_ms': round(mean(r['latency_ms'] for r in rows), 3),
            'p50_latency_ms': percentile(0.50), 'p95_latency_ms': percentile(0.95),
            'makespan_ms': round(makespan, 3),
            'throughput_requests_per_sim_second': round(1000 * len(rows) / makespan, 3),
            'worker_counts': {str(w.id): w.completed for w in nodes},
        }, 'requests': rows,
    }
