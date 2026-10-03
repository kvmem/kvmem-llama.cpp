# Monitor and watchdog

Two stdlib-only Python sidecars for a resident `ninfer-serve`. Both read the server's `GET /stats`
(see [serving](../../docs/serving.md#stats)); start the server with `--stats-port` to give them a
listener of their own. When the server has an `--api-key`, export it in the environment variable
named by `--api-key-env` (default `NINFER_API_KEY`).

## Dashboard: `monitor.py`

Samples `/stats`, `nvidia-smi` and `/proc` every `--interval` seconds (default 5), tails the
server's `--request-log-jsonl` file, and serves a self-contained dashboard: scheduler occupancy,
admitted requests, decode and prefill throughput, Device and Host KV use, speculative acceptance,
TTFT and processing time (TTFT less queue wait), GPU, CPU and RAM, KV and state transfers,
pressure events (evictions, demotions, degradations, fallbacks), cache-reuse paths, and the recent
requests. `/api/samples` returns the same state as JSON and `/healthz` answers `ok`.

```bash
python3 tools/monitor/monitor.py --stats-url http://127.0.0.1:8081 \
  --jsonl /var/log/ninfer/requests.jsonl --port 8090
```

The dashboard binds `127.0.0.1` unless `--bind` says otherwise. The last 4320 samples (6 hours at
5 s) are kept in memory.

`--sensor-command` adds a temperature reading: a command that prints one Celsius value, run every
tick, drawn against `--sensor-warn-c` and `--sensor-critical-c` under `--sensor-label`. With
`--stop-server-pid-file`, a reading that stays above critical for `--stop-after-s` seconds
(default 60) sends SIGKILL to the PID in that file, and again every tick until the reading drops,
so hardware such as a power connector cools before the server draws power again. The PID file is
the launcher's to write; the monitor never looks a process up by name.

## Watchdog: `watchdog.py`

Restarts a wedged server: one whose scheduler reports work while none of its cumulative engine
counters -- tokens, decode rounds, prefill and control units, Host work, context transfers,
pressure searches, captures -- has advanced for `--stall-seconds` (default 150). A server that
stops answering with work outstanding counts as not progressing. Prefill advances its counters per
chunk, so a long prompt is progress, not a stall.

```bash
python3 tools/monitor/watchdog.py --stats-url http://127.0.0.1:8081 \
  --restart-command 'systemctl restart ninfer'
```

After a restart, a server start of its own (uptime going down) or a `503` while the model loads,
the watchdog waits `--grace-seconds` (default 90) before it can act again. After `--max-restarts`
restarts (default 3) within `--restart-window-seconds` (default 1800) it stops restarting and only
reports, since a wedge that keeps returning needs a person. It prints one status line per poll
(`--interval`, default 15 s).
