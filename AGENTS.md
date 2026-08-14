# Gold AI repository instructions — V15.1 LEAN

Active execution contract: `D:\OneDrive - ueh.edu.vn\GOLD_AI_CODEX_EXECUTION_PROMPT_V15_1_LEAN.txt` (SHA-256 `D6F4302E9ABBCCA3C12F6945968BBD1974523325017261DD22CBB0478A5E66BE`). The former V15 Master Prompt is a requirements catalog only; consult it only for non-conflicting detail.

## Invariants

- Inspect `PROJECT_STATE.md`, the latest experiment artifact/log, active processes and repository status before acting. Preserve valid jobs, user data, checkpoints and unrelated changes. Apply control changes at a safe boundary; never rerun a completed experiment family.
- Exact legal Exness `XAUUSDm` is the controlling feed. PAXG and other legal sources are research/context proxies only. Record source provenance, UTC range and SHA-256.
- Pipelines are exactly M5 and M15. Bind data, label, model, signal, execution and management to the source timeframe. Higher timeframes are causal closed-candle context only.
- Across M5 and M15, open positions + accepted pending orders + reserved in-flight requests are globally limited to 3. Reconcile and reserve atomically; uncertainty fails closed.
- Use chronological nested walk-forward, label-end purge/embargo and realistic costs. Inner data selects; outer OOS only evaluates the frozen configuration. Final holdout is sealed and one-way until every development gate passes.
- The >75% directional-accuracy target remains a reported stretch target. Recognition requires observed outer-OOS/rolling-unseen accuracy >=60% **and** Wilson lower 95% >=60% with fixed support; report `MET_STRONG`, `MET_POINT_ONLY`, `NOT_MET` or `INCONCLUSIVE` and never brute-force abstention to fake it.
- Executable profit and capital survival precede accuracy. Stress net <0 or PF <1.0 always rejects a model, even above 75% accuracy. Accuracy >=65% with stress net >=0/PF >=1.0 can only retain a Challenger; Production Champion requires durable positive future-unseen economics, PF >1.10, risk/stability/parity and every registry gate.
- Dynamic lot remains bounded by risk, drawdown, margin, concentration and kill-switch gates. Never martingale or increase risk merely because leverage is high.
- `LIVE_TRADING_ENABLED=FALSE` unless the user gives new explicit live authorization and all safety gates pass. Research must not send real orders, transfer funds or mutate a live account.
- Respect the user's latest resource policy without changing a valid fold. New research jobs use all useful CPU cores, RAM and a smoke-tested GPU backend when supported; unrestricted full power still preserves thermal, OS, disk, checkpoint and fail-closed safety. A 45-minute plateau is diagnostic, not permission to kill a healthy fold.
- After the frozen anchored-auction cycle completes, ordinary in-scope research downloads are authorized only for a pre-registered use-case and from the publisher/provider's official HTTPS site, official package index or first-party repository. Stage under `D:\AI`; record domain/owner/version/license/time/size/SHA-256 and check a publisher-provided checksum/signature when available. Antivirus/Defender scanning is not mandatory under the user's latest low-lag policy. If first-party ownership is uncertain, do not download/run it. External data remains `SOURCE_MISMATCH`; never execute unreviewed dataset scripts or unpickle untrusted models.

## Canonical state and commands

- State: `PROJECT_STATE.md`; architecture/gaps: `PROJECT_AUDIT.md`; definitions/gates: `VALIDATION_PROTOCOL.md`; research evidence: `RESEARCH_LOG.md` and experiment artifacts.
- Continue the current exact-feed sequence: `.venv\Scripts\python.exe scripts\continue_mt5_dual_sequence.py` only when no valid sequence is already active.
- Run independent artifact validation with `scripts\validate_triple_artifacts.py` using the completed artifact selected in `PROJECT_STATE.md`.
- Run repository tests: `.venv\Scripts\python.exe -m pytest -q` through the configured resource governor when training is not competing for resources.
- Manual research queue: `MANUAL_TRAIN_PLAN.json`. At a safe cycle boundary, add at most the single next scientifically pre-registered job with a unique id, `PENDING` state, trainer and distinct validator under `scripts/`, fresh summary and validation artifacts, and bounded resources. Never queue a closed family/config, live action or final-holdout access; an honestly empty queue is valid when no eligible mechanism exists.
- The manual worker owns numbered Vietnamese run reports under `D:\AI\Ket qua`; Codex must not create a duplicate report.
- Every newly completed manual-train handoff carries `human_report.cycle_label` in the exact five-digit form `Vòng xxxxx`. After independent review of that handoff, notify the user beginning with that label whether the candidate was kept, rejected or failed; never claim improvement before validator evidence.
- `autonomous_m5_model_catalog_v1` is completed, independently reviewed and closed with all four frozen configurations rejected. Never rerun or retune it. Its code/artifacts are evidence and optional infrastructure, not the controlling research roadmap.
- Codex owns the scientific roadmap after each reviewed cycle: pre-register a genuinely new mechanism, run it autonomously through the single-writer worker, independently validate it, update state/traceability and continue without asking the user to choose a model. Method autonomy never relaxes leakage, sealed-holdout, realistic-cost or live-trading prohibitions.

Always keep these states separate in milestone reports: `ENGINEERING_READY`, `RESEARCH_CYCLE_COMPLETE`, `ACCURACY_TARGET_STATUS`, and `PRODUCTION_CERTIFICATION_STATUS`.

`OPERATIONAL_OBJECTIVE_POLICY.md` defines the profit-first evidence ladder and the preferred `GPT-5.6-sol` MEDIUM/HIGH/XHIGH allocation. Record the recommended tier before a new work group; runtime/orchestrator selection occurs only at a turn/safe boundary and never changes a frozen scientific protocol.
