# Restore Fly Tweets daily delivery — implementation plan

> Use systematic debugging and test-driven development for each fix; review the combined result before final deployment.

**Goal:** Restore automatic queue replenishment and verified native X delivery without visible windows or duplicate posts.

**Architecture:** Preserve the existing neural generation/cadence policy, persistent outbox and verified native sender. Treat the rendered v2 publication text as delivery data while validating the saved original generation. Deploy fixes to the production code branch used by the hosted orchestrator; retain the existing hidden Windows publisher.

**Constraints:** No LLM runtime generation; no invented posting cadence; no overwriting published text or history; no weakening idempotency checks for real conflicts; checkpoint commits and pushes after each phase, reporting any push failure immediately.

## Phase 1 — generation identity compatibility
- [x] Add failing replay tests for a native-v2 publication and real original-text/schedule conflicts.
- [x] Compare validated native-v2 provenance against the immutable outbox while preserving delivery fields.
- [x] Run focused publishing/reconciliation tests, review, commit and push.

## Phase 2 — deploy and refill
- [ ] Deploy the compatible production code and ensure one active publishing transport.
- [ ] Back up live JSON state before any direct recovery mutation.
- [ ] Run the real daily orchestrator, inspect every failure, and verify newly generated rows enter the native queue with unchanged fly-selected times.
- [ ] Verify independent training/stats health; remedy concrete failures that prevent the existing loop working.
- [ ] Test the scheduled hidden publisher, verifying a naturally due post in X when one becomes due. Do not alter the fly schedule to manufacture a success.

## Phase 3 — final verification
- [ ] Review all changes and run relevant regression checks.
- [ ] Record exact job/queue/publication evidence and limits; commit and push the final report.
- [ ] Verify remote SHAs and report the final branch/commit identifiers.
