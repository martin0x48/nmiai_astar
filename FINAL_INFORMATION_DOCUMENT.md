# Final Information Document — Astar Island NM i AI 2026

This document is the final handoff for the repo state at the end of the competition run on 2026-03-22.

It is meant to answer four questions:

1. What did the competition actually require?
2. What ended up working?
3. What looked promising but failed under realistic evaluation?
4. What is the final script state and what should future work start from?

## 1. Task and scoring

The task was to predict the final 50-year simulator state on a `40x40` grid for `5` stochastic seeds per round. Submissions are `H x W x 6` probability tensors.

Key scoring facts:
- The public docs describe entropy-weighted KL divergence on dynamic cells.
- Ocean and mountain are static and excluded from scoring.
- The local `analysis_seed*.json` files contain:
  - the exact submitted prediction
  - the 200-sim ground-truth distribution
  - the actual live score for that seed

Most important scoring discovery:
- Plain KL against the 200-sim `ground_truth` is too optimistic and is not a good model of the live API score.
- The best working proxy we found is:
  - draw a low-sample empirical target from `ground_truth`
  - use about `N = 10` effective samples per cell
  - score with entropy-weighted KL
- This fit historical live scores much better than the old offline metric.

Best scorer-fit found on real submitted predictions:
- `N = 10`
- entropy-weighted KL
- prediction floor around `1e-4`
- no target-side pseudocount smoothing
- `MAE = 8.53`
- correlation `0.9693`

What did not hold up:
- plain full-GT KL
- unweighted KL
- target-side smoothing / pseudocount theories

Practical meaning:
- the live scorer behaves like a noisy empirical version of the documented entropy-weighted KL, not like KL to the full 200-sim mean target

## 2. Core game facts that held up

These looked stable across the repo analysis and saved rounds:

- Ocean never changes.
- Mountain never changes.
- Initial maps differ across the 5 seeds in a round.
- The 5 seeds are not identical reruns, but they do share strong round-level structure.
- One observed seed often reveals useful round-level signal for the other seeds.

The round-level structure claim is an inference, not organizer-confirmed hidden parameters. The evidence for it is empirical:
- seeds within the same round cluster much more tightly than seeds across rounds
- using observations from one seed consistently helped predict the others

## 3. Evolution of approaches

### Early phase

Early observation-only approaches were bad despite using lots of queries.

Main reason:
- raw observations are noisy samples of the final state, not the target distribution itself
- naively treating them as truth or doing heavy correction was unstable

### Simulator phase

The simulator was useful for learning the game, but not as the main live policy.

What we learned:
- simulator quality under oracle parameters could look very strong
- fixed simulator settings were much worse
- observation-conditioned simulator deployment failed because mapping observations to the right regime/parameters was unreliable

Important result:
- oracle tuned simulator under proxy: about `86.04`
- realistic observation-conditioned simulator policy: about `80.84`
- ML baseline in the same comparison: about `82.68`

Conclusion:
- the simulator engine was not useless
- the deployment bottleneck was observation -> regime inference
- standalone simulator submissions were not the right endgame strategy

### ML phase

The main successful direction was per-cell supervised ML trained on completed-round analysis data.

Best model family progression:
- RF was useful early with little data
- XGBoost took over once enough rounds existed
- LightGBM was slightly competitive
- the best late-stage model was a simple mean ensemble:
  - XGBoost
  - LightGBM
  - CatBoost

## 4. Feature engineering conclusions

The base feature extractor around local terrain, settlement proximity, coastal structure, and round-level global features was the right backbone.

Things that helped:
- richer local terrain-count features
- settlement distance / density features
- coastal and mountain-adjacency features
- round-level global features:
  - settlement ratio
  - empty ratio
  - forest ratio

Things that did not justify their complexity:
- replay-derived settlement stats
- many ad hoc extra spatial pressure features
- broad feature spam

Late takeaway:
- the main bottleneck was not obviously missing one magic handcrafted feature
- most gains came from better objective alignment, calibration, and ensemble choice

## 5. Training objective conclusions

The biggest robust training change was moving toward entropy-weighted training.

Result:
- unweighted training lost to entropy weighting
- `entropy^2` sample weighting was the best of the tested simple powers

Under the live-like proxy:
- unweighted: `79.96`
- entropy^1: `81.39`
- entropy^2: `82.30`

This did not solve everything, but it was a real improvement and stayed valid after the scoring correction.

## 6. Query strategy conclusions

This was one of the most important strategic findings.

### What worked

The best observation policy was not "use every query for direct correction."

The useful pattern was:
- use observations to estimate round-level regime information
- feed that into the ML model
- let the ML model handle the cellwise prediction

### Seed-0 observation policy

The first stable working policy was:
- `9` queries on seed 0
- cover the whole map with a `3x3` tiling of `15x15` windows
- compute observed global features from that one seed
- use those globals for all 5 target seeds

Why it worked:
- it was cheap
- it captured round-level structure
- one noisy sample often matched the live empirical scoring target better than the 200-sim GT average

### Avg5 policy

The late-stage improvement was the `avg5` policy:
- observe all 5 seeds, `9` queries each
- compute one global-feature vector per observed seed
- for each target seed, run the model once for each observed global vector
- average the 5 predicted distributions

This uses `45/50` queries.

Deterministic validation result across rounds `1..21`:
- `seed0_only`: `83.23`
- `avg5_obs`: `83.84`
- `avg5_obs_multi3`: `83.87`

Wins:
- `avg5_obs`: `17/21`
- `avg5_obs_multi3`: `18/21`

Important meaning:
- the `avg5` gain survived deterministic reruns
- it was not just benchmark noise from unstable seeding

### What did not help

- per-seed observation correction with weak update rules
- Bayesian correction as a main late-stage policy
- using extra queries without a tested rule for converting them into better submitted probabilities

General lesson:
- more queries are only useful if there is a reliable model for turning them into better distributions

## 7. Calibration conclusions

Overconfidence was a real issue.

Two late-stage calibration findings held up:

### Targeted smoothing

Earlier in the run, targeted smoothing helped:
- only smooth dynamic cells
- only when `max_prob > 0.90`
- blend with uniform using `alpha = 0.05`

This helped under the proxy, but it was not a major endgame solution.

### Temperature scaling

Temperature scaling was the stronger late-stage calibration fix.

Best result:
- temperature around `1.08` to `1.09`
- applied to dynamic-cell logits/probabilities during prediction

Proxy comparison:
- baseline XGB+LGBM mean: `83.27`
- temp-scaled XGB+LGBM: `83.73`
- XGB+LGBM+Cat: `83.42`
- XGB+LGBM+Cat with temperature: `83.81`

Late conclusion:
- the final stack should use light softening globally through temperature scaling
- this was better justified than heuristic smoothing alone

## 8. Ensemble conclusions

The simple tree ensemble was worth keeping.

What worked:
- XGB + LGBM beat either one alone by a small but real margin
- adding CatBoost and temperature scaling gave another incremental gain

What did not work:
- RF+XGB ensemble
- broad simulator blending with one fixed alpha

Simulator blending result:
- a tiny blend could help on a narrow slice
- but on the broader check, pure ML was still best

Conclusion:
- tree-model ensembling yes
- simulator blending only as future research, not as the final live policy

## 9. Neural-model conclusions

We tested a real CNN probe late in the competition.

Result on held-out recent rounds:
- tree stack average proxy: `85.16`
- CNN probe average proxy: `77.94`
- delta: `-7.23`

Conclusion:
- CNN was not competitive enough
- transformer-style work was even less justified under the time constraints
- neural approaches were not the right late-stage gamble

## 10. Round-specific lessons

### Round 16

This was the round that exposed the offline-metric mismatch.

Live result:
- about `85.06`

Main lesson:
- the old offline metric was misleading model selection

### Round 17

Live result:
- `91.93`

Main lesson:
- the seed-0 observation/global-feature policy was working
- round 17 was not mainly a regime-estimation failure
- the remaining gap looked more like broad model headroom

### Round 18

Missed operationally.

Offline reconstruction suggested:
- the live policy itself would probably have been fine
- the failure was missing the round, not the strategy

### Round 19

Live result:
- `88.65`

Main lesson:
- sparse rounds remained hard
- higher confidence in sparse regimes was a danger sign

### Round 22

Live result:
- `81.14`

This was a major postmortem round.

Main finding:
- the bad result was not just luck
- it was not a stale-code issue
- it was specifically a case where global-feature conditioning hurt

Local reconstruction on round 22:
- actual submitted `avg5` proxy: `80.34`
- reconstructed `avg5` proxy: `79.77`
- reconstructed `seed0_only` proxy: `78.44`
- reconstructed `no_globals` proxy: `83.57`

That led to a narrow guardrail in the final script for ultra-sparse rounds.

## 11. Final-script state

At the end of the run, the intended live script was:

- `auto_solve.py`
- `ml_solver.py`
- model version `4`
- model family `xgb_lgbm_cat_ensemble`
- temperature scaling `1.09`
- `avg5` observation policy

Final important guardrail:
- if all 5 seeds are observed
- and the round-average observed globals satisfy:
  - settlement ratio `< 0.035`
  - empty ratio `> 0.72`
  - forest ratio `< 0.255`
- then skip global-feature conditioning and use the no-global model prediction instead

Reason:
- this was a narrow fix for the round-22-style ultra-sparse failure mode
- a simple low-settlement threshold alone was not safe

## 12. What future work should trust

These are the strongest conclusions from the entire run:

1. Use a live-like proxy, not plain KL to the 200-sim GT mean.
2. Entropy-aware training matters.
3. Query value comes mainly from regime estimation, not naive direct correction.
4. `avg5` is real enough to take seriously.
5. Temperature scaling is a real gain.
6. Simple tree ensembles are worth it.
7. Standalone simulator deployment was not good enough.
8. Neural models were not competitive in the late-stage tests.

## 13. What future work should be skeptical of

These ideas were attractive at some point but did not hold up cleanly:

- plain offline GT-KL as the main evaluator
- hard simulator regime classification
- fixed simulator as the main live policy
- heavy observation correction rules
- lots of ad hoc new features without strong validation
- CNN/transformer last-minute pivots
- naive low-settlement heuristics without more context

## 14. Best next research directions

If work continued after this run, the most credible next steps would be:

1. Better round-level gating for when globals should be used or ignored.
2. Better calibration against real API outcomes, possibly with an explicit learned proxy-to-live mapping.
3. Conditional simulator use as a feature source, not as a standalone predictor.
4. More principled mixture-of-experts logic for sparse vs normal rounds.

## 15. File map

Useful files in this repo:

- `auto_solve.py`: live submission script
- `ml_solver.py`: feature extraction, training, model load/save
- `LEARNINGS.md`: earlier compact competition notes
- `RESEARCH_LOG_2026-03-21.md`: detailed experiment log
- `benchmark.py`: main benchmarking code
- `benchmark_avg5.py`: deterministic avg5 validation
- `score_fit_probe.py`: scorer-family fit against real submitted predictions
- `sim_blend_probe.py`: simulator blending probe
- `cnn_probe.py`: neural-model probe
- `data/round*/`: saved round details, observations, and analysis

## 16. Bottom line

The final competition-quality approach was not a simulator and not a neural net.

It was:
- supervised per-cell ML
- with entropy-aware training
- calibrated with temperature scaling
- using a small tree ensemble
- and using observations mainly to estimate round-level context rather than to directly overwrite the prediction

The biggest strategic mistake early on was optimizing against the wrong offline metric.
The biggest strategic correction was switching model selection to a live-like low-sample entropy-weighted proxy.
The best late-stage practical improvement was the combination of:
- `avg5`
- temperature scaling
- XGB + LGBM + CatBoost

That is the clearest final summary of what was actually learned here.
