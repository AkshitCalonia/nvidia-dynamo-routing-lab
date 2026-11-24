# Dynamo Routing Lab

A reproducible **discrete-event simulator** exploring the tradeoff between prefix/KV-cache locality and load balancing for multi-worker LLM inference serving. Inspired by NVIDIA's November 19, 2026 technical blog, [Reducing Rollout Bottlenecks in Reinforcement Learning with NVIDIA Dynamo](https://developer.nvidia.com/blog/reducing-rollout-bottlenecks-in-reinforcement-learning-with-nvidia-dynamo).

> **Scope:** Independent educational prototype, not an implementation of NVIDIA Dynamo, not affiliated with NVIDIA, and **not a GPU benchmark**. All timings are synthetic model assumptions and should not be interpreted as TensorRT-LLM, Dynamo, vLLM, or hardware performance numbers.

## The engineering question

If related RL rollouts reuse the same long prompt, cache affinity saves prefill work. But sending them to a single cached worker can produce a long queue. How do routing strategies trade off cache reuse, tail latency, and total completion time?

## Policies

- `round_robin`: rotates across healthy workers without considering prefix reuse.
- `cache_only`: aggressively reuses the earliest worker that already has the prompt; otherwise sends to the lowest-numbered healthy worker. This intentionally exposes queue hotspots.
- `balanced`: picks the worker with the earliest **estimated completion time**, incorporating queue delay, cache reuse, prefill, and decode cost. It is a *toy heuristic*, not NVIDIA's routing algorithm.

## Architecture

```
seeded workload (groups of 8 repeated prompts)
              |
              v
     routing policy selector
       /       |        \
 round robin  cache only  balanced
       \       |        /
              v
  virtual inference workers (serial queues, bounded prefix cache)
              |
              v
 request timeline + metrics (JSON and CSV)
```

## Windows 11 quick start

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest -v
python -m dynamo_lab.cli --output results/comparison.json
python -m dynamo_lab.cli --failure-at 90 --output results/failure_comparison.json
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python`.

## Experiments

Baseline: `python -m dynamo_lab.cli --output results/comparison.json`

Failure experiment: `python -m dynamo_lab.cli --failure-at 90 --output results/failure_comparison.json`

Stress test: `python -m dynamo_lab.cli --requests 500 --workers 8 --groups 30 --arrival-spacing-ms 2 --failure-at 150 --output results/stress_comparison.json`

Compare the following in each JSON file's `runs[*].summary`: `cache_hit_rate`, `prefill_compute_ms`, `p95_latency_ms`, `makespan_ms`, `throughput_requests_per_sim_second`, and `worker_counts`. Exported CSVs are easy to graph in Excel.

The failure is injected **before routing request 90**, and a replacement worker re-enters after a configured delay. Already scheduled work on a failed worker is not interrupted or replayed. This is a simplification and therefore **does not prove in-flight failure recovery**.

## Modeling assumptions and limitations

- Synthetic arrival times and linear compute estimates, not GPU kernels.
- Single server queue per worker, fixed prompt-cache size (LRU), discount factor for a prefix hit.
- No actual token streaming, transfer overhead, KV-block placement, batching, continuous batching, preemption, weight freshness, or networking.
- Simulated worker failure means new work cannot route to it until restored; existing scheduled work remains counted. No real loss of in-flight requests.
- Reported throughput is **requests per simulated second**, not measured requests/s.
- Policy improvements depend on chosen workload and cost coefficients; don't generalize them as hardware speedups.

## Reproduce and extend

The inputs are fixed using `--seed`. All three policies receive the **same request sequence**. Read the generated `results/*.csv` and commit them with the repo. For a strong next step, swap the synthetic cost function for **measured** TTFT and decode timing from AIPerf or an OpenAI-compatible model endpoint, and compare the predictions with actual throughput. Another extension is to model in-flight failure and task replay, plus checkpoint and weight-version eligibility.

## Code map

- `dynamo_lab/simulator.py`: generated requests, policies, bounded worker cache, schedules, summary metrics
- `dynamo_lab/cli.py`: experiment CLI, JSON and CSV export
- `tests/test_simulator.py`: reproducibility, cache reuse, failure rerouting, invalid input, and policy tradeoff tests
- `.github/workflows/tests.yml`: automated tests for Python 3.11 and 3.12

## License
MIT. Original educational implementation. Please cite the NVIDIA blog separately when discussing inspiration.
