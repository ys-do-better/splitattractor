# rca-repair

A first version of the two agents in the RCA taxonomy work: the **RCA agent** and the **repair agent**, implemented
as the flowcharts of the Liu deck (v17: pages 4–6 for RCA, page 10 for repair), and run end to end on one real example.

```
python3 -m rcarepair run examples/w1_ds28e17 --memory runs/memory --fresh    # 20–70 s, depending on load
python3 -m pytest -q --assert=plain -p no:anyio tests                        # unit checks
```

Output goes to `runs/w1_ds28e17/`: `runlog.json` (every flowchart box the run passed, with the values behind it),
`summary.md`, `diagnosis.json`, `final.diff`, `upstream.json`, `episodes.jsonl`, `ltm1.json`, `ltm2.json`.
Run it again without `--fresh` and the RCA answers from LTM 1 (case matching) and the repair starts from the LTM 2
compound.

## The example: w1_ds28e17

The DS28E17 1-Wire-to-I2C bridge driver takes the length of an SMBus block read from the device and reads that many
bytes into a 34-byte buffer. A device that answers 255 makes it write past the buffer. This is Kachi bug 11, fixed
upstream in 169ae5e65e5a (CVE-2026-80931).

`examples/w1_ds28e17/target/kernel/` holds Linux 6.18 code copied verbatim: the driver, the 1-Wire I/O functions it
calls, and the I2C SMBus emulation (`tools/extract_c_functions.py` did the copying). `shim/` only lets that code run
as a process: kernel API stand-ins, an emulated 1-Wire bus with a DS28E17 and I2C sensors behind it, and a polling
program. AddressSanitizer catches the overflow in the real `w1_read_block` loop.

What one run does (`runs/w1_ds28e17/summary.md`). Since 2026-10-31 the RCA asks up the build steps
(`core/walk.py`). Since 2026-11-01, walking back compares against replaying the failure as it is. A target that
answers differently from run to run (an agent, a person) is asked each question several times and judged on
failure rates.

| box | what happened |
|---|---|
| failure | ASan heap-buffer-overflow WRITE in `w1_read_block` (w1_io.c:141) |
| case matching | LTM 1 empty, not seen |
| round 1 | in hand: one failing trace -> causal chain tracing; walks `buf[i]` <- loop bound `len` (255) <- `count` <- `msgs[i+1].buf[0]`; stuck: who wrote that byte? |
| next evidence | replay under gdb with a watchpoint on the byte: written once, 0 -> 255, by `w1_read_block` called from `w1_f19_i2c_write_read`; the value is `w1_read_8()`, i.e. from the device |
| round 2 | localized: w1_ds28e17.c:392, `msgs[i+1].buf[0]` used as the read length |
| facts on the chain | the driver only rejects `count == 0`; the core's check (`i2c-core-smbus.c:307`) runs after the transfer has returned; the length comes from the device |
| ask up: (5) logic | are its inputs right? the device's length 255 is outside the assumption <= 32 (wrong); the 34-byte buffer holds the largest legal block (right); with the assumption true, the existing tests that pass here (values 32 and 4) do not overflow (logic right) -> go up to the step that made the assumption |
| ask up: (4) boundary | the assumption <= 32 does not match what the device really sent (255); the rule is already in the system (`i2c-core-smbus.c:307`) and the design expects it (`-EPROTO`) -> inputs right, output wrong: the root is this step, the rule is not at the boundary |
| walk back | this case: replayed as it is, w1_ds28e17.c:392 sees 255 and overflows; replayed with `msgs[i+1].buf[0]` held to <= 32 there (255 -> 32), no overflow (deterministic, so one run each); the class: the same step made 2 sites (392, 418), predicted 418 fails too, a generated test confirms it on the original build |
| column | root step -> Conceptual (models, rules); symptom -> Factual |
| stop-gap | running and recurring -> Factual x black box, cheapest first: Bl (stop issuing operations to that device) -> the polling workload stops crashing; kept as temporary |
| correction | Conceptual x {black, white}: Cd, Rb, Lg, ... -> Cd: condition from the invariant (`<= I2C_SMBUS_BLOCK_MAX`, taken from the late check), error exit borrowed from the same function (`i = ...; goto error;`), errno from the late check (`-EPROTO`); reproducer returns -EPROTO, regression outputs identical, invariant holds |
| generalisation | lift to "a device-written length used as a copy length"; static query finds 2 sites (392, 418); a generated test crashes site 418 on the original build; patched and validated; Bl removed |
| memories | LTM 1: the case; LTM 2: `Cd @ device boundary` with the query |
| upstream | the final diff is identical to commit 169ae5e65e5a |

## Layout

- `rcarepair/core/` — the algorithm, independent of the system: `table.py` (periodic table v14), `model.py`
  (episode, evidence, diagnosis), `memory.py` (LTM 1, LTM 2, episodic log), `rca.py`, `whychain.py` (facts along
  the causal chain), `walk.py` (ask up the five build steps with one question each, then walk back against a
  replay as it is; several runs and a Fisher test for targets that answer differently each run), `repair.py`,
  `runlog.py`.
- `rcarepair/methods/families.py` — the five rows of page 1: statistical association (Ochiai), model-based diagnosis
  (minimal hitting sets), causal chain tracing, experimental isolation (ddmin), hypothesis testing (needs a language
  model; not configured in this version).
- `rcarepair/targets/c_asan/` — a C program checked by ASan: build and run, replay under gdb (`gdb_driver.py`:
  crash state, last writer by watchpoint, reachability, a replay with a value held to a bound), tree-sitter facts (`cast.py`), the dependency walk, guards,
  late checks and the static site query (`analysis.py`), patching and validation (`target.py`).
  `rcarepair/targets/base.py` lists what any new kind of system has to provide.
- `examples/w1_ds28e17/` — `target.yaml` (build, tests, what an operator can do), `knowledge.yaml` (where outside
  data enters, which calls copy LEN bytes), `scenario.yaml` (running, can stop, access), `upstream/` (the real fix).

## Not in this version

- Hypothesis testing and parameter filling by a language model (the hooks exist; no model is configured).
- Compounds of two elements are ranked but only tried when LTM 2 already holds them.
- Rollback (Rb) and change-logic (Lg) have no parameter filler for C code yet; they are skipped and logged.
- The C target answers the walk's questions for data-flow bugs (a value from outside, a buffer, a rule on the value);
  races and ordering bugs need their own answers. The schema, domain-knowledge and requirement steps are exercised
  only by unit tests (`tests/test_walk.py`); the agent example on deck v17 page 8 was walked by hand.
- No agent target yet. The several-runs path (`deterministic = False`: majority answers, a control replay, a
  one-sided Fisher test on failure counts) is exercised only by a scripted fake agent in `tests/test_walk.py`.

Code under `examples/*/target/kernel/` is from Linux and stays under GPL-2.0.
