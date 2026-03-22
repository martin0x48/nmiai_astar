# Research Log — 2026-03-21

This file captures the main findings from the round 16 investigation and the inter-model exchange in `/tmp/modeltalk`, so they are not lost.

## 1. Round 16 outcome

- Live round 16 result was poor: about `85.06`, rank `40/272`.
- The actual submission path was fast because it reused a pre-trained local model.
- Round artifacts:
  - `data/round16/details.json` saved at `2026-03-21 16:41:00 UTC`
  - `data/round16/obs_seed0_phase1.json` saved at `2026-03-21 16:41:52 UTC`
  - `model.pkl` predates the round and was already trained

## 2. Critical scoring discovery

The repo's old offline evaluator is optimistic relative to historical API scores.

- Old offline metric:
  - compare prediction against `analysis["ground_truth"]`
  - score as `100 * exp(-KL)`
- Historical evidence:
  - across analyzed historical submissions, this overestimates the live API score by a large margin

### Working live-like proxy

Best current approximation to the live API:

1. Treat `analysis["ground_truth"]` as the latent true distribution.
2. Sample a low-count empirical target per cell from that latent distribution.
3. Use approximately `N = 10` samples per cell.
4. Score prediction against that empirical target using entropy-weighted KL.
5. Average over several Monte Carlo draws.

Empirical fit to historical API scores:
- `N=5`: MAE about `10.52`, corr about `0.962`
- `N=10`: MAE about `9.46`, corr about `0.967`
- `N=20`: MAE about `12.25`, corr about `0.966`
- `N=50`: MAE about `14.90`, corr about `0.966`

So `N = 10` is the current working proxy.

## 3. ML findings under the new proxy

### Training weights

The recent entropy-weighting change was not the main mistake. Under the proxy:

- unweighted: `79.9582`
- entropy^1: `81.3908`
- entropy^2: `82.3001`

So `entropy^2` still looks best among those variants.

### Global-feature modes

Under the proxy:

- `gt -> seed0_obs`: `82.6848`
- `seed0_obs -> seed0_obs`: `82.4262`
- `gt -> gt`: `82.3001`
- `none -> none`: `79.1735`

So the seed-0 observation policy still looks real, and removing global features is clearly worse.

### Calibration / smoothing

Small global smoothing helps a little, but targeted smoothing helps more.

Global smoothing sweep:
- raw: `82.2183`
- global uniform blend `alpha=0.005`: `82.2782`

Targeted smoothing sweep:
- best found: smooth only cells with `max_prob > 0.90`
- use `alpha = 0.05` toward uniform
- result: `82.6195`

Interpretation:
- ML predictions are too sharp on some cells under the live-like proxy.
- Smoothing only the most confident cells is better than smoothing the whole map.

## 4. Round 16 analysis details

Round 16 seed-0 observed global features:
- settlement ratio: `0.0983`
- empty ratio: `0.6644`
- forest ratio: `0.2321`

The main live error appears concentrated in near-deterministic / low-entropy cells rather than high-entropy cells.

Historical nearest-neighbor regime comparison suggested round 16 looked more like moderate / expansion-style rounds than collapse rounds.

## 5. Inter-model simulator findings

The other model in `/tmp/modeltalk` had a simulator path with very strong old-metric results, but those were initially on the optimistic offline metric.

### Old metric simulator result

- Per-round tuned simulator: about `94.53` to `94.73` on the old analysis-distribution metric
- Fixed-parameter simulator: about `86.8` old metric

### Simulator under the proxy

After re-evaluation under the live-like proxy:

- per-round tuned simulator: `86.04`
- fixed-parameter simulator: `78.80`

This is important:
- tuned simulator proxy `86.04` is materially above ML baseline `82.68`
- fixed simulator proxy `78.80` is below ML

Conclusion:
- simulator is only competitive if the round parameters can be identified well enough from observations

### Per-round tuned simulator proxy (selected notes)

Large drops under the proxy happen especially on expansion rounds.

Example subset reported by the other model:
- R1 old `94.83` -> proxy `85.64`
- R6 old `93.82` -> proxy `80.82`
- R10 old `96.99` -> proxy `94.78`
- R12 old `90.80` -> proxy `86.88`

Interpretation:
- collapse rounds hold up much better under low-sample evaluation
- expansion rounds suffer more from noisy empirical targets

## 6. Simulator deployment research

### Presets

The other model tested a 5-preset simulator family:

- A: collapse
- B: moderate single-faction
- C: expansion single-faction
- D: moderate multi-faction
- E: expansion multi-faction

Results:
- oracle best-preset proxy average: `84.18`
- initial GT-assigned preset policy: `81.67`

So presets can beat ML only if classification is good enough.

### Classification bottleneck

Hard classification is the main problem.

Observed issue:
- some low-settlement rounds are true collapse
- some low-settlement rounds are actually high-conflict expansion
- example: R8 looked low-density in observation but best preset was expansion-style, not collapse

### Free signal from `details.json`

Faction count is available for free from initial conditions and matters:

- rounds 1-9: many factions
- rounds 10-13: single faction

This is useful, but not sufficient by itself.

### Continuous mapping

The other model tested a linear observation-to-params map and it failed badly.

Reported subset:
- R6 proxy `75.68` vs oracle `79.51`
- R7 proxy `66.11` vs oracle `83.59`
- R8 proxy `75.84` vs oracle `78.64`
- R9 proxy `78.67` vs oracle `82.68`
- subset average `74.08` vs oracle `81.11`

Conclusion:
- simple linear obs->params mapping is not viable

### Current best simulator deployment idea

Not:
- one fixed simulator
- hard preset classification only
- linear continuous obs->params mapping

Best next candidate:
- a small observation-conditioned preset ensemble
- especially for ambiguous many-faction moderate-density rounds
- likely blend between D and E instead of forcing one

## 7. Round 16 simulator-specific notes

From the other model:
- round 16 has `58` settlements and `58` factions from `details.json`
- observed seed-0 settlement ratio is about `9.8%`

This puts round 16 in an ambiguous many-faction moderate-density region.

Best current simulator-side guess:
- hard classification between D and E is risky
- a D/E blend is more plausible than a single hard call

## 8. Working comparison targets

Useful benchmarks going forward:

- ML baseline under proxy: about `82.68`
- ML + targeted smoothing: about `82.62`
- simulator preset oracle: `84.18`
- tuned simulator oracle: `86.04`

If an observation-conditioned simulator policy can clear the ML band consistently under the proxy, it becomes worth integrating.

## 9. Best next steps

1. Evaluate an observation-conditioned preset ensemble under the proxy.
2. Use simulated one-sample seed-0 observations for historical rounds without real saved obs.
3. Compare ensemble simulator against:
   - ML baseline
   - ML with targeted smoothing
4. Focus especially on ambiguous rounds like R8 / round 16 where low observed settlement density may reflect conflict rather than food-collapse.

## 10. Message files

Inter-model exchange happened in `/tmp/modeltalk` using files like:
- `msg_002_from_opus.md`
- `msg_004_from_opus.md`
- `msg_006_from_opus.md`
- `msg_008_from_opus.md`
- `msg_010_from_opus.md`
- `msg_012_from_opus.md`
- `msg_014_from_opus.md`
- `msg_016_from_opus.md`
- `msg_018_from_opus.md`

Those are in `/tmp` and may be ephemeral, so the core conclusions are summarized here.

## 11. Later update: end-to-end observation-conditioned simulator failed

After the earlier optimistic oracle and preset results, the other model completed a realistic end-to-end evaluation using observation-conditioned preset selection.

Result:
- full observation-conditioned simulator policy: **`80.84` proxy average**
- ML baseline remains about **`82.68`**

So under realistic observation-to-regime mapping, the standalone simulator lost to the ML path.

### Reported per-round results

Selected outcomes from the end-to-end simulator policy:
- R1: `82.68`
- R3: `94.49`
- R4: `80.07`
- R7: `76.32`
- R10: `94.58`
- R12: `72.30`
- R13: `66.67`

### Failure modes

- Oracle simulator quality did not translate into deployable observation-conditioned quality.
- Ambiguous many-faction rounds remain the main weakness.
- Hard or weakly learned regime selection loses too much of the simulator's upside.

## 12. Code state implemented before round 17

The following changes are already in the repo and committed in `c60cf65`:

- `ml_solver.py`
  - versioned model metadata (`MODEL_VERSION = 2`)
  - entropy-squared training weights (`sample_weight_power = 2.0`)
  - version-aware save/load helpers
- `auto_solve.py`
  - auto-retrain when newer analysis data exists
  - 9-query seed-0 observation policy as the default live path
  - targeted smoothing:
    - smooth only dynamic cells
    - only when `max_prob > 0.90`
    - `alpha = 0.05`

So round 17 used the updated entropy-squared model recipe plus targeted smoothing.

## 13. Round 17 live result

Round 17 was submitted with the updated ML path and completed with:

- score: **`91.9325`**
- rank: **`35/283`**
- seed scores:
  - `93.8015`
  - `92.3642`
  - `90.8796`
  - `92.8178`
  - `89.7994`
- queries used: `9/50`

Comparison:

- round 16: `85.0622`
- round 17: `91.9325`
- round 15: `93.8842`

Interpretation:

- The round 16 recovery was real.
- The round 17 result is still below the best ML rounds, so the current fixes are helping but not enough.
- Targeted smoothing is a small calibration gain, not a multi-point solution.
- The main remaining gap is likely some combination of:
  - imperfect regime inference from seed-0 observation
  - residual calibration error on confident cells
  - limited model capacity / missing features for certain round types

## 14. Next analysis target after round 17

When round 17 analysis becomes available, compare:

1. actual submitted predictions vs ground truth
2. raw ML vs targeted-smoothed ML on round 17 specifically
3. error concentration by:
   - cell entropy
   - terrain / coastal / settlement-adjacent regions
   - confidence bucket (`max_prob`)
   - faction-count / round-regime indicators

The goal is to decide whether round 17 was mainly:

- a calibration miss
- a global-regime miss
- or a feature / model-capacity miss

## 15. Round 17 postmortem after analysis arrived

Round 17 analysis files were fetched into `data/round17/analysis_seed*.json` and the live submission was reconstructed from:

- `data/round17/details.json`
- `data/round17/obs_seed0_phase1.json`
- the pre-round-17 `model.pkl` (version 2)

### Key result: not mainly a regime miss

Observed seed-0 global features used live:

- settlement ratio: `0.28385`
- empty ratio: `0.49747`
- forest ratio: `0.19986`

Actual round-17 averages across the 5 ground-truth seeds:

- settlement ratio: `0.27801`
- empty ratio: `0.49146`
- forest ratio: `0.19025`

So the live seed-0 observation was close to the true round outcome.

More importantly, on round 17 itself:

- no global features: `89.02` weighted old-metric score
- observed seed-0 global features: `97.23`
- oracle per-seed GT global features: `96.91`

Interpretation:

- the 9-query seed-0 observation policy was absolutely useful
- round 17 was not lost because the global regime estimate was bad
- the one-sample observation again appears to capture live seed-specific signal better than the 200-sim GT average

### Targeted smoothing effect on round 17

On round 17 specifically, targeted smoothing was basically neutral:

- weighted old-metric score:
  - raw: `97.2302`
  - smoothed: `97.2325`
- live-like proxy:
  - raw: `83.4477`
  - smoothed: `83.3854`

So smoothing was not the cause of the mediocre live score, and it did not create a large gain on this round either.

### Where the old-metric loss was concentrated

Confidence buckets on dynamic cells:

- `0.00-0.70`: score `97.26`, count `5484`
- `0.70-0.80`: score `97.06`, count `940`
- `0.80-0.90`: score `97.33`, count `304`
- `0.90-0.95`: score `96.31`, count `26`
- `0.95-1.01`: score `88.73`, count `3`

Interpretation:

- there were only a handful of ultra-confident cells
- the round 17 miss was not dominated by a single large overconfidence failure mode

Initial-terrain buckets:

- settlements: `98.06`
- ports: `96.91` (tiny sample)
- forest: `96.79`
- plains: `97.35`

This suggests the model is somewhat weaker on forest-origin cells than plains, but the gap is not huge.

### Current read after round 17

The most likely explanation is:

- the query/global-feature policy is working
- the remaining gap is mostly broad model headroom:
  - calibration still imperfect
  - forest/plains transition regions still hard
  - missing features or model capacity on some expansion-style dynamics

So round 17 does not support abandoning the ML path or abandoning the 9-query seed-0 policy.

## 16. Pre-round-18 refresh and experiment pass

Round 17 analysis was incorporated into the local model before the next submission cycle.

### Retrain status

Manual retrain after adding round 17:

- loaded `115721` dynamic-cell examples
- skipped `20279` static cells
- saved `model.pkl` with:
  - `version = 2`
  - `sample_weight_power = 2.0`

The reported in-sample training score after retrain was:

- unweighted: `95.79`
- weighted: `97.87`

This is not CV, but it confirms the updated model artifact was built correctly.

### Corrected-proxy model comparison

Using the same faster proxy harness for all models:

- XGBoost: `81.63`
- LightGBM: `81.76`

Interpretation:

- LightGBM is only slightly ahead of XGBoost under the corrected proxy.
- The margin is too small to justify swapping the live path blindly.

An `XGB + LGBM` ensemble run was started, but the full average was not yet recorded here. Early per-round results were mixed rather than a clear blowout.

### Corrected-proxy weighting sweep

Using the same faster proxy harness and live-style seed-0 observation setup:

- entropy power `1.0`: `82.25`
- entropy power `1.5`: `82.43`
- entropy power `2.0`: `82.48`

Interpretation:

- entropy-weighted training still beats plain `power=1.0`
- `power=2.0` remains slightly ahead of `1.5`
- there is no evidence for downgrading away from the current entropy-squared trainer before round 18

### Practical takeaway before round 18

Best current safe policy remains:

- keep the existing ML live path
- keep the 9-query seed-0 observation strategy
- keep entropy-squared training
- keep targeted smoothing as a small calibration tweak

No pre-round-18 experiment so far has shown a large enough edge to justify a risky architecture change at inference time.

## 17. Round 18 retrospective after missing the live window

Round 18 was missed live.

API state:

- `seeds_submitted = 0`
- `queries_used = 0`
- `round_score = None`

Artifacts were pulled locally into `data/round18/`:

- `details.json`
- `analysis_seed0.json` through `analysis_seed4.json`

### Held-out offline evaluation of the current model

This uses the refreshed post-round-17 model and treats round 18 as unseen test data.

Round-18 average GT global features:

- settlement ratio: `0.36667`
- empty ratio: `0.42777`
- forest ratio: `0.16368`

#### No-query fallback

- weighted old-metric score: `72.67`
- live-like proxy: `63.05`

So round 18 was another case where ignoring round-level regime information would have gone very badly.

#### Simulated live policy: 9-query seed-0 full-map observation

Using simulated full-map seed-0 observations from GT and then applying the normal policy:

- weighted old-metric score: `95.79`
- live-like proxy: `83.07`

Across 48 simulated seed-0 observation draws:

- proxy std was only about `0.035`

So the current live policy appears very stable on round 18 once the seed-0 round-level signal is available.

#### Oracle per-seed GT global features

- weighted old-metric score: `95.18`
- live-like proxy: `82.55`

Interpretation:

- the current seed-0 observation policy would likely have done well on round 18
- on this round, simulated seed-0 observation was actually slightly better than oracle GT globals again
- that matches the earlier pattern where one sampled observation can align better with the live empirical target than the 200-sim GT average

### Main lesson from round 18

The miss was operational, not model-quality:

- the live policy itself still looks sound on this held-out round
- the costly failure was simply not submitting

## 18. Follow-up model experiments after round 18

### Clean corrected-proxy single-model comparison

Re-ran the corrected-proxy comparison cleanly on rounds with analysis through round 18, using:

- train with GT global features
- test with a simulated seed-0 observation global feature estimate
- the same proxy harness for all models

Results:

- XGBoost: `81.78`
- LightGBM: `81.96`
- XGB + LGBM average ensemble: `82.08`

Interpretation:

- LightGBM is slightly better than XGBoost under the corrected proxy.
- The simple XGB/LGBM mean ensemble is best of the three, but only by a small margin.
- This is real enough to be interesting, but not yet a dramatic breakthrough.

### Expansion-focused feature patch

A narrow feature patch was tested locally, adding signals for:

- inverse-distance settlement pressure
- nearest/second-nearest settlement gap
- nearest plains distance
- nearest forest distance

Early held-out-round comparisons were mixed and did not show a convincing gain over the existing extractor. The patch was therefore reverted instead of leaving unproven complexity in the repo.

Conclusion:

- keep the current feature extractor for now
- if model work continues, the next credible direction is the small XGB/LGBM ensemble rather than ad hoc feature growth

Main reason:
- the observation-based classifier cannot reliably infer the needed growth regime

Examples:
- R13 was routed to a moderate single-faction preset but needed high growth
- R12 sat on a boundary between moderate and expansion
- R7 and R4 were misrouted among the multi-faction presets

Interpretation:
- simulator engine quality is strong under oracle params
- deployment bottleneck is observation -> params inference
- current simulator policy is not good enough to replace the ML baseline live

### Updated practical conclusion

Most plausible next simulator uses are now:
- use simulator outputs as additional ML features
- try ensembleing over several simulator presets instead of hard routing
- or train a learned regime/parameter predictor

But:
- **do not trust standalone observation-conditioned simulator policy yet**
- current realistic proxy result (`80.84`) is below ML (`82.68`)

### Ensemble implementation

The measured next step was implemented in the repo:

- `ml_solver.py` now trains an `XGB + LGBM` mean ensemble
- model metadata/version bumped to `version=3`
- saved payload now records `model_family=xgb_lgbm_ensemble`

Important implementation note:
- `python ml_solver.py train` initially produced an unloadable pickle because `EnsembleModel` was serialized as `__main__`
- fixed by binding the script module to `ml_solver` before pickling
- verified by retraining `model.pkl` and loading it successfully through both `ml_solver.load_saved_model()` and `auto_solve.load_ml_model()`

## 19. Round 19 submission record

Round 19 opened on `2026-03-22` and was submitted with the new ensemble live path.

Submission details:
- round id: `597e60cf-d1a1-4627-ac4d-2a61da68b6df`
- status at submit time: `active`
- seeds submitted: `5/5`
- queries used: `9/50`
- code path: `auto_solve.py` using `model.pkl` version `3`
- model family: `xgb_lgbm_ensemble`
- smoothing: targeted smoothing enabled (`alpha=0.05`, `max_prob > 0.90`)

Observed round-19 seed-0 global features from the 9-query phase:
- settlement ratio: `0.0136986301369863`
- empty ratio: `0.7159336697909157`
- forest ratio: `0.26820475847152125`

Saved artifacts:
- `data/round19/details.json`
- `data/round19/obs_seed0_phase1.json`

Pre-close local risk read:
- round 19 looks much sparser than rounds 15 and 17
- model confidence is unusually high in this regime
- targeted smoothing fired on about `1017` dynamic cells per seed on average
- ensemble member disagreement is higher than on rounds 15, 16, and 17

Interpretation before score is known:
- low 90s is possible, but not the base expectation
- this round looks riskier than round 17 and somewhat more dangerous than round 16 by internal confidence/disagreement signals
- likely band before seeing live score: high 80s to low 90s

## 20. Avg5 idea for follow-up

Latest report (`REPORT_2026-03-22.md`) suggests a possible next-step policy:

- observe all 5 seeds (`9 queries × 5 = 45`)
- compute global features from each observed seed
- run the ML model 5 times per target seed, once with each observed global-feature set
- average the 5 predicted distributions before submission

Current read:
- promising, but not trusted enough yet for default live use
- reported proxy gain is modest
- reported round wins are mixed rather than dominant
- likely effect is calibration through observation-noise ensembling, which could be fragile

Important caveat before using live:
- `benchmark_avg5.py` currently seeds with `test_round.__hash__()`, which is not stable across Python processes
- benchmark should be made deterministic and rerun before treating avg5 as production-ready

Plan for tomorrow:
- patch `benchmark_avg5.py` to use deterministic seeds
- rerun avg5 validation cleanly
- only consider adopting avg5 if the gain survives deterministic reruns and still looks worth spending `45/50` queries

## 21. Neural model probe

Question explored:
- whether something more radical, like a CNN or transformer-style approach, might beat the current tree ensemble

Environment check:
- deep learning frameworks were not installed initially
- `torch` was installed locally for research on the disposable machine
- no GPU was available (`torch.cuda.is_available() == False`)

What was actually tested:
- a small patch-based CNN (`cnn_probe.py`)
- input: `7x7` local patches with terrain one-hot channels plus settlement/port masks
- auxiliary inputs: observed global features and normalized `(x, y)` position
- training target: soft 6-class distributions with entropy-squared weighting
- evaluation: recent held-out rounds `19`, `20`, `21` scored with the N=10 live-like proxy
- baseline for comparison: current tree stack on the same holdouts

Results:
- round 19: tree `86.80`, CNN `74.77`  (delta `-12.02`)
- round 20: tree `83.61`, CNN `82.74`  (delta `-0.88`)
- round 21: tree `85.08`, CNN `76.30`  (delta `-8.78`)
- average proxy: tree `85.16`, CNN `77.94`  (delta `-7.23`)

Interpretation:
- this CNN probe is not competitive enough to justify a last-round switch
- the current tree model remains much stronger on recent hard rounds
- a transformer is even less attractive as a late-stage gamble: more complexity, more training sensitivity, and no evidence yet that spatial context is the missing ingredient

Practical takeaway:
- do not switch to CNN/transformer for the final round
- if neural ideas are revisited later, they need more than a quick patch model: likely a full-grid architecture, stronger conditioning on round-level signals, and a larger research window

## 20. Temperature scaling + CatBoost benchmark (2026-03-22)

### Motivation

The N=10 live-like proxy from section 2 shows the live scorer penalizes overconfident predictions. Instead of heuristic targeted smoothing (section 12), a learned temperature parameter can systematically soften all predictions.

### Benchmark protocol

- Leave-one-round-out CV across 18 rounds
- Train with GT global features, test with simulated seed-0 observation
- Score with N=10 proxy (20 Monte Carlo draws per evaluation)
- Temperature T learned via grid search on in-sample training predictions

### Results

| Model | Proxy avg | Delta vs baseline |
|-------|-----------|-------------------|
| baseline (XGB+LGBM mean) | 83.27 | — |
| temp_scaled (XGB+LGBM + T=1.09) | 83.73 | **+0.46** |
| catboost3 (XGB+LGBM+Cat mean) | 83.42 | +0.15 |
| catboost3_temp (XGB+LGBM+Cat + T=1.08) | **83.81** | **+0.54** |

Temperature scaling wins on 15/18 rounds. Biggest gains on R9 (+1.32), R10 (+1.22), R8 (+0.98).

Optimal temperature is consistently ~1.08-1.09: predictions need slight softening.

### Changes deployed

- `ml_solver.py` version 4:
  - `MODEL_TEMPERATURE = 1.09`
  - CatBoost added as third ensemble member (`xgb_lgbm_cat_ensemble`)
  - `apply_temperature()` applied to dynamic cells during prediction
  - Temperature stored in model pickle metadata
- `auto_solve.py`:
  - Removed heuristic targeted smoothing (replaced by temperature scaling)
  - Temperature scaling happens inside `predict_round()` automatically

### Old metric comparison

Temperature scaling slightly reduces the old metric (94.98 → 94.70 avg), which confirms the old offline metric was rewarding overconfident predictions that the live API penalizes.

## 22. Deterministic avg5 rerun + simulator blend follow-up (2026-03-22)

### Deterministic `avg5` rerun

Reason for rerun:
- the original `benchmark_avg5.py` used `test_round.__hash__()`, which is not stable across Python processes
- the claimed `avg5` gain is small enough that benchmark noise matters

Changes made:
- `benchmark_avg5.py` now uses a deterministic `stable_round_seed(...)`
- repeated observation averaging was made explicit via `OBS_REPEATS`
- the benchmark is being rerun cleanly with deterministic seeds

Interim results so far (first 7 rounds completed):
- R1: `83.52 -> 85.00` (`avg5`, `+1.48`)
- R2: `84.74 -> 84.71` (`-0.02`)
- R3: `88.37 -> 88.40` (`+0.03`)
- R4: `85.80 -> 86.06` (`+0.26`)
- R5: `83.79 -> 84.43` (`+0.64`)
- R6: `82.54 -> 82.64` (`+0.09`)
- R7: `81.78 -> 81.91` (`+0.12`)

Current read:
- the deterministic rerun has not invalidated `avg5`
- the effect still looks modest, but mostly positive so far
- this is exactly why the rerun was necessary: the gain is real-sized enough that methodology quality matters

Final deterministic result across rounds 1-21:
- `seed0_only`: `83.23` proxy avg
- `avg5_obs`: `83.84` proxy avg
- `avg5_obs_multi3`: `83.87` proxy avg
- `avg5_obs` vs `seed0_only`: `+0.61`
- `avg5_obs_multi3` vs `seed0_only`: `+0.64`
- win counts:
  - `avg5_obs`: `17/21`
  - `avg5_obs_multi3`: `18/21`

Most important late-round results:
- R12: `70.01 -> 72.20` (`avg5`, `+2.19`)
- R15: `79.92 -> 80.85` (`+0.94`)
- R16: `84.14 -> 85.57` (`+1.43`)
- R20: `81.24 -> 85.72` (`+4.48`)

Notable small negatives:
- R8: `-0.20`
- R14: `-0.07`
- R19: `-0.06`

Interpretation:
- the deterministic rerun materially strengthens confidence in `avg5`
- the gain is no longer a fragile one-off benchmark artifact
- `avg15` is a little better still, but only slightly, and it costs even more observations

### Simulator blend probe

Question explored:
- whether a fixed-parameter simulator can help as a small blend on top of the current `avg5` ML path

What was tested:
- new local probe: `sim_blend_probe.py`
- baseline: current `avg5` ML path with deterministic observation repeats
- simulator: fixed-parameter `best_params_v2.npy`
- score: N=10 live-like proxy

Focused recent-round result (`rounds 19-21`, `obs_repeats=3`, `sim_sims=60`):
- alpha `0.00` (pure ML): `86.54`
- alpha `0.10`: `86.69`
- alpha `0.20`: `86.48`
- alpha `0.30`: `86.03`
- alpha `0.40`: `85.41`
- alpha `0.50`: `84.62`
- alpha `1.00` (sim only): `78.15`

Best on this narrow slice:
- fixed blend `alpha=0.10`
- gain vs ML: `+0.15`

Broader follow-up (`rounds 17-21`, `obs_repeats=2`, `sim_sims=40`):
- alpha `0.00` (pure ML): `85.59`
- alpha `0.10`: `85.52`
- alpha `0.20`: `85.13`
- alpha `0.30`: `84.51`
- alpha `0.40`: `83.68`
- alpha `0.50`: `82.64`
- alpha `1.00` (sim only): `72.22`

Per-round best-alpha summary on rounds 17-21:
- R17: best `alpha=0.00`, no gain
- R18: best `alpha=0.00`, no gain
- R19: best `alpha=0.10`, `+0.02`
- R20: best `alpha=0.10`, `+0.35`
- R21: best `alpha=0.10`, `+0.02`

Interpretation:
- fixed sim blending is not strong enough to become the default live path
- the simulator may still have some conditional value on particular regimes, but the safe global choice remains pure ML
- the standalone fixed simulator remains far worse than ML

Practical takeaway:
- keep sim blending as a conditional research direction, not a production switch
- if revisited, the right next step is regime-gated blending rather than one fixed alpha everywhere

## 23. Direct scorer-family fit against real submitted predictions (2026-03-22)

Question explored:
- whether the live API scorer might be something more exotic than low-sample KL, e.g. extra smoothing or pseudocounts on the target side

Key enabler:
- the local `analysis_seed*.json` files contain:
  - the exact submitted `prediction`
  - the `ground_truth`
  - the actual live seed `score`
- that makes it possible to fit scorer variants directly against real submissions rather than proxy-on-proxy comparisons

Dataset used:
- `100` scored seed submissions from completed rounds on disk

What was tested:
- old metric baselines:
  - full-GT KL, weighted / unweighted
  - prediction floor sweep
- empirical low-sample families:
  - sample sizes `N in {5, 8, 10, 12, 15, 20}`
  - weighted / unweighted KL
  - prediction floor `1e-6`, `1e-4`, `1e-3`
  - target pseudocount smoothing `tau in {0.0, 0.1, 0.5, 1.0}`

Best coarse fit found:
- `empirical N=10`
- `entropy-weighted KL`
- `pred_floor = 1e-4` (effectively same as `1e-6` / `1e-3`)
- `target_tau = 0.0`

Metrics for best fit:
- `MAE = 8.53`
- `corr = 0.9693`
- mean bias `+4.90`

Nearby alternatives:
- `N=12`, weighted, `tau=0`: `MAE = 8.65`, `corr = 0.9696`
- `N=8`, weighted, `tau=0`: `MAE = 9.02`, `corr = 0.9694`
- `N=15`, weighted, `tau=0`: `MAE = 9.13`, `corr = 0.9696`

Important negative result:
- target-side pseudocount smoothing hurt across the board
- there was no evidence that the live scorer is better modeled by a smoothed empirical target

Old-metric comparison:
- old unweighted full-GT KL: `MAE = 20.40`, `corr = 0.814`
- old weighted full-GT KL: `MAE = 15.08`, `corr = 0.965`

Interpretation:
- the earlier theory still holds up
- the live scorer is best modeled as low-sample empirical, entropy-weighted KL
- the evidence does **not** support adding scorer-side smoothing / pseudocounts
- the remaining mismatch looks more like calibration than formula mismatch

Useful calibration note:
- for the best `N=10` proxy, the raw proxy still overpredicts live scores by about `+4.90` on average
- fitting a simple linear map from proxy to actual live score reduced MAE further to about `6.56`
- this suggests the biggest remaining error is scale/offset, not a fundamentally different scorer family

## 24. Round 22 postmortem and final-script guardrail (2026-03-22)

Confirmed live result:
- round 22 score: `81.1356`
- seed scores: `79.6024`, `78.8889`, `82.3244`, `81.2128`, `83.6494`

Important conclusion:
- this was not a wrong-code-path failure
- round 22 really did use the intended live stack:
  - model version 4
  - XGB + LGBM + CatBoost
  - temperature scaling
  - avg5 observation path

Direct local comparison on round 22:
- actual submitted `avg5` prediction:
  - live avg `81.14`
  - local proxy `80.34`
- reconstructed `avg5` with current saved v4 model:
  - proxy `79.77`
- reconstructed `seed0_only`:
  - proxy `78.44`
- reconstructed `no_globals`:
  - proxy `83.57`

So the round-22 failure appears to be:
- not just bad luck
- not a seed-0-vs-avg5 issue alone
- specifically a case where global-feature conditioning hurt badly

Observed round-22 averages across all 5 observed seeds:
- settlement ratio: `0.0273`
- empty ratio: `0.7222`
- forest ratio: `0.2478`

Comparison to round 21:
- round 21 observed averages:
  - settlement ratio: `0.0653`
  - empty ratio: `0.6648`
  - forest ratio: `0.2604`
- on round 21, globals still helped:
  - `seed0_only` proxy `86.40`
  - `avg5` proxy `86.31`
  - `no_globals` proxy `85.75`

Final-script change added:
- `auto_solve.py` now has a narrow ultra-sparse guardrail
- if all 5 seeds are observed and the observed round-average features match the round-22-like corner:
  - `settlement_ratio < 0.035`
  - `empty_ratio > 0.72`
  - `forest_ratio < 0.255`
- then global-feature conditioning is skipped and the script uses the no-global ML prediction instead

Why this is intentionally narrow:
- a simple low-settlement threshold alone is not safe
- round 19 also had very low settlement, but local tests still preferred global conditioning there
- so the guard is only meant to catch the specific ultra-sparse / very-empty pattern that matched round 22
