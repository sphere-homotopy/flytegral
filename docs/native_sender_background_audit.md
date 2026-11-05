# Native sender background launch audit — 2026-11-05

The Windows task `FlyTweets-V2-Native-X` now starts `wscript.exe //B //Nologo scripts/run_fly_v2_hidden.vbs -AllowLive`. The wrapper creates PowerShell hidden, waits for completion and returns its exit status. Python and Node inherit the hidden console. The existing browser launch remains headless.

The deployed task action was updated in place: its 15-minute triggers, user, overlap policy, battery settings and execution limit were preserved. The installer now produces the same hidden action for future installations. Interactive logon remains required, as before; this does not add logged-out execution.

Verification:
- Two Windows integration tests exercise actual WScript/PowerShell execution from a directory containing spaces, dry/live argument propagation, waiting and success/failure exit codes.
- Queue/reconciliation tests: 9 passed; native publishing core tests: 6 passed.
- Real scheduled live-mode job completed with LastTaskResult 0 and a new SUCCESS log entry.
- A second scheduled launch was monitored for eight seconds at 20 ms intervals using EnumWindows: zero new visible console or Windows Terminal windows. This is a sampled observation, not a guarantee about unrelated desktop processes.
- All four published native queue entries were independently opened in headless X; displayed text matched exactly, including the latest permalink https://x.com/fly_topology/status/2117486471087063445 (2026-11-03 05:18 UTC).

## Publishing limitation found

The native queue has four published records and no future ready records. Durable rows contain 68 published, 34 error, 13 paused and 3 missed records, with no future generation reserve. A successful sender run currently performs reconciliation and has nothing to publish; no new test tweet was fabricated or posted.

The production publishing path is native X through the authorized persistent Playwright profile, rather than Buffer. The historical Buffer integration remains in the repository.

GitHub orchestrator run 42058036210 failed during outbox sync with `ValueError: conflicting durable row for idempotency key: batch-2026-11-01-a1:0 (generation field: text)`. The last successful saved cycle is dated November 1 and generated the three scheduled November 3 posts. Thus delivery has worked, but daily queue replenishment is currently blocked. Existing source/durable content was preserved; resolving that data conflict is separate from this launch-only fix.
