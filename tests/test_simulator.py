from dataclasses import replace
import pytest
from dynamo_lab.simulator import Settings, Request, simulate, workload


def test_seed_reproducibility():
    config = Settings()
    assert workload(config) == workload(config)
    assert simulate('balanced', config) == simulate('balanced', config)


def test_all_requests_complete():
    for policy in ('round_robin', 'cache_only', 'balanced'):
        r = simulate(policy, Settings(requests=44))
        assert r['summary']['completed'] == 44
        assert len({x['id'] for x in r['requests']}) == 44


def test_cache_reuse_reduces_prefill_compute():
    s = Settings(requests=24, groups=1, workers=3)
    tasks = workload(s)
    no_cache = simulate('balanced', replace(s, cache_discount=0.0), tasks)
    cached = simulate('balanced', s, tasks)
    assert cached['summary']['prefill_compute_ms'] < no_cache['summary']['prefill_compute_ms']


def test_worker_failure_reroutes_and_recovers():
    s = Settings(workers=3, requests=80, failure_at=20, failed_worker=0, replacement_delay_ms=60)
    r = simulate('balanced', s)
    assert r['summary']['worker_failures_injected'] == 1
    assert r['summary']['completed'] == 80
    during = [x for x in r['requests'] if 20 <= x['id'] < 30]
    assert all(x['worker'] != 0 for x in during)
    assert any(x['worker'] == 0 for x in r['requests'] if x['id'] >= 30)


def test_one_worker_failure_not_allowed():
    with pytest.raises(ValueError):
        simulate('balanced', Settings(workers=1, failure_at=2))


def test_invalid_config():
    with pytest.raises(ValueError):
        workload(Settings(requests=0))
    with pytest.raises(ValueError):
        simulate('nonsense', Settings())


def test_expected_cache_vs_load_tradeoff():
    s = Settings(workers=4, requests=64, groups=1, arrival_spacing_ms=1)
    cache_only = simulate('cache_only', s)
    balanced = simulate('balanced', s)
    assert cache_only['summary']['cache_hit_rate'] >= balanced['summary']['cache_hit_rate']
    assert balanced['summary']['makespan_ms'] < cache_only['summary']['makespan_ms']
