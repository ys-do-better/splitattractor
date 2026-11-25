"""What the two agents need from a system. The agents (core/) only talk to a target through these calls, so a new
kind of system (a service, a pipeline, an agent harness) plugs in by implementing them.

RCA side
  failure()                    -> (Signal, Evidence)      the failure as first seen
  signature(signal, ev)        -> dict                    key for LTM 1 case matching
  remind(signal)               -> [Episode]               this failure and similar past episodes
  units(access)                -> str                     what a candidate root-cause unit is at this depth
  in_hand(ev)                  -> set[str]                what kind of evidence is in hand (drives the method choice)
  backtrack(ev)                -> (hops, missing)         dependency walk for causal chain tracing
  measure(need, ev)            -> (key, value)            acquire the next piece of evidence
  guards_on(chain, ev) / later_checks(chain, ev) / origin(chain)   facts along the causal chain
  outside_value(chain) / state_fits(iota, chain) / logic_under(iota, L)     the walk at the logic step
  rule_known(iota) / case_expected(iota)                                    are the inputs of the boundary / schema right
  replay_as_is(root, iota, L) / replay_with_root_right(root, iota, L)       walking back, this case: control and fix
  class_sites(root, iota, L) / class_test(site)                             walking back, the class
  deterministic, runs          False for systems that answer differently from run to run (agents, people): every
                               question is then asked `runs` times and the walk back compares failure rates
Repair side
  recurring(D) / blackbox(sym) / stable(config)            stop-gap on the running system (black-box row)
  params(sym, D, ev)           -> iterator of theta        fill an element's parameters
  apply(theta) / revert(theta) / validate(D, theta)        patch and check (tests, reproducer, invariant)
  lift(theta, D) / sites(theta_star) / instantiate(theta_star, site) / site_test(site)   generalisation
  drop_stopgap(handle) / final_diff()
"""


class Target:
    name = "target"
    deterministic = True
    runs = 5
