# Mandatory home-node compute resource preflight — v1.0

**Effective:** 2026-10-19. **Scope:** every research repository under this GitHub account, for every NEW heavy scientific workload that may run on the user's **actual active home compute node**. This is a binding user requirement, not optional performance advice. Historical jobs, evidence, protocols and frozen scientific criteria are unchanged.

## Zero-work rule and host authenticity

- Do **not** interrupt, restart, duplicate or retroactively reconfigure an already running job. Apply the resource rule to its next permitted launch/step.
- Before selecting any parallel worker count, actually inspect the **execution machine**, not a ChatGPT/cloud/container host used only for orchestration. Identify the home node and the execution environment (host, container, cgroup, scheduler, allowed cpuset). An SSH status page, an old `nproc` reading or a static inventory is not current resource evidence.
- If active-node access, host identity, permitted CPU affinity, current memory or competing-workload evidence is unavailable: **BLOCKED_RESOURCE_PREFLIGHT**, not an excuse to launch with an arbitrary low worker count. Do not claim measurements that were not made.

## Obligatory live measurements immediately before the launch

Record UTC sampling start/end, host identity (redacted in public logs), tools, cgroup/scheduler context and evidence/checksums. At minimum:

1. **Effective logical CPUs:** online CPU IDs, `sched_getaffinity`/cpuset allowance, cgroup CPU quota (`cpu.max` or equivalent), per-CPU utilization sampled over an actual interval, and other hard reservations or pinned affinities. Compute an explicit **available CPU-ID set**, not merely a total hardware core count; distinguish eligible from currently unoccupied CPUs. Respect NUMA/SMT placement constraints if materially relevant.
2. **Memory:** `MemAvailable` (not just `MemFree`), effective cgroup memory allowance/current usage, swap/pressure, reserved system and competitor headroom, existing resident/shared memory, and a measured or justified **peak memory per worker** plus shared/IO overhead. Do not infer memory feasibility from CPU count.
3. **Competing work:** live process/scheduler inventory including PIDs (or redacted stable identifiers), owners where permitted, job names, states, elapsed time, CPU use/affinity/cgroups, memory consumption, other research runs and reservations. A currently quiet CPU used by an active competing job is not automatically available for exclusive reservation. If access is insufficient to establish safety, fail closed.
4. **Science/runtime contract:** allowed solver processes, threads per process, process-independent vs shared state, required serial phases, GPU/IO limits when applicable, deterministic/numerical constraints and existing immutable scientific protocol. A science-imposed cap is a *legitimate documented cap*; it cannot be silently lifted merely to increase throughput.

## Maximize safe effective parallelism, with explicit CPU affinity

- Let `F` be the verified available logical CPU-ID set after subtracting competing jobs and reservations, respecting quota/cpuset/NUMA; `t` the scientific/runtime threads required per worker; `M` the safely allocable memory after explicit margins and competing work; `m` the measured/justified per-worker peak plus overhead; `M_shared` fixed shared memory; `N_scheduler` the scheduler/cgroup cap; and `N_science` the science-specific cap (infinite when absent).
- Pick **the largest feasible integer** `N >= 1`, bounded by `floor(|F|/t)`, `floor((M-M_shared)/m)`, `N_scheduler`, and `N_science`, plus verified IO/GPU/NUMA limits. Account for a fractional cgroup CPU quota conservatively. The selected `N` must be the maximal **permitted**, not an arbitrary habitual `1`, `2`, or `4`.
- Select and record exact CPU IDs/affinity masks for parent and each worker/thread group; they must lie in `F` and must not overlap existing exclusive reservations. Pin or enforce via scheduler, `taskset`/affinity APIs, or an equivalent supported mechanism. Coordinate `OMP_NUM_THREADS`, BLAS/MKL/OpenBLAS/NumExpr thread settings, subprocess pools and nested libraries to prevent oversubscription.
- If peak memory or current competition is unknowable, do a separate prospectively scoped, safe capacity measurement first; **do not** improvise resource ceilings while conducting the frozen science run.
- Any `N` below the full currently available CPU ceiling requires a **specific pre-launch explanation** and the measured limiting bound (memory, science, competing jobs, quotas, scheduler, NUMA/IO/GPU), not a generic "conservative" label.

## Prospective evidence and launch binding (hard gate)

**Before** submitting the scientific workload, persist a versioned resource preregistration/manifest with at least:

`run_id_or_future_run_key, repo, commit, frozen_science_protocol, node_attestation, observation_utc, allowed_cpu_ids, cpu_quota, busy_reserved_cpu_ids, available_cpu_ids, competing_jobs_summary, MemAvailable, memory_cgroup_limit_and_usage, system_memory_reserve, estimated_peak_memory_per_worker, shared_memory, threads_per_worker, caps_by_source, limiting_reason, chosen_worker_count, per_worker_affinity, threading_environment, intended_launch_command, resource_manifest_sha256, preregistration_timestamp_utc`.

Record a durable immutable reference (commit/artifact and checksum) **before** the scientific launch. Redact machine/user/process-sensitive fields in public repositories, retaining verifiable full evidence in an authorized private location. Bind the launched execution to that exact manifest and report actual worker count/affinity and any drift. No silent edit after results, no retroactive preregistration, and **no alteration of previously frozen scientific thresholds or physics**.

If node availability/affinity changes between preflight and launch, **stop only the not-yet-started launch**, repeat the preflight and create a new prospective version. Do not interrupt other jobs.

## Classification and existing work

- `RESOURCE_READY`: current home-node measurements, maximal permitted worker selection, exact affinity and prospective manifest are all verified.
- `BLOCKED_RESOURCE_PREFLIGHT`, `BLOCKED_COMPETING_WORKLOAD`, `BLOCKED_MEMORY_CAPACITY`, `BLOCKED_SCIENTIFIC_PARALLELISM`, or `BLOCKED_SCHEDULER_LIMIT`: record evidence and actual reason; these are *infrastructure/workload planning* states, never physical model falsification.
- For ongoing research, the first next action is to check whether an authorized run is **already active**; consume its outputs, avoid duplicate calculations, and apply this protocol at the **next** execution boundary.

**Canonical principle:** real free resources on the actual home node first; maximal scientifically and operationally permissible concurrency second; explicit affinity and prospective resource record third; scientific run only after all three.
