# Harbor Bend

Designed 16 x 20 racetrack with a four-cell-wide approach and a broad right turn.
Map input: harbor_bend.map. Settings: harbor_bend.json.
F=track, #=wall, S=start, G=finish. Training acceleration failure: 10%.

Run from Experiments/Scripts/RL/Python:

```powershell
py -3.12 exp_multi_map_racetrack.py --maps maps/harbor_bend.json --episodes 5000 --seeds 1 --init-q -100 --epsilon 0.1 --eval-max-steps 300 --eval-repeats 5 --output-dir results/harbor_bend_seed1 --plot --show
```

Measured result: training completed 5,000 episodes; 8/20 noise-free evaluation
trials finished within 300 steps (four starts, five trials per start). Crashes
still cause random start resets. This does not establish convergence/optimality.
JSON metrics, sparse policy NPZ and figures are in results/harbor_bend_seed1.

Other trials retained for transparency:
- results/harbor_bend: seed 0, initial Q=-1000, epsilon=.1, 5,000 episodes;
  0/20 greedy evaluation trials finished.
- Optimistic initial Q=0, epsilon=.2, seed 0: stopped after at least 3,000 of
  the planned 10,000 episodes due to expensive exploration and no improving
  block returns (approximately -1596, -1876, -1893). No final policy or
  evaluation result was saved for this interrupted run.

All results above come from actual agent execution on the unchanged map.
