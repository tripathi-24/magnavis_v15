# Baseline parameter sweep (Apr 27 long GT)

- Created: `20260522_071302`
- CSV: `/Users/kamaltripathi/Documents/CSE@IITK/Summer_Term_26/Thesis_Work/magnavis_v15/Datafiles/magnetic_data_20260426_060000_to_20260427_090000_1hz.csv`
- Cases run: 17 (17 ok, 0 failed)
- Skip initial minutes: 62.0

## Best F1 per mode

| mode | k | ewma_α | med_win | sav_win | recall | precision | F1 | event_recall |
|------|---|--------|---------|---------|--------|-----------|-----|--------------|
| ewma | 1.0 | 0.35 | 31 | 31 | 0.987 | 0.929 | 0.957 | 0.600 |
| median | 1.0 | 0.35 | 31 | 31 | 0.983 | 0.928 | 0.955 | 0.600 |
| savgol | 1.0 | 0.35 | 31 | 31 | 0.959 | 0.928 | 0.943 | 0.600 |

## Reference (prior benchmark, k=3)

Apr-27 runs at **k=3** reported ~4% baseline recall vs **~75%** for GRU pretrained.
This sweep shows **k is the dominant knob** on long-GT timelines.

At **k=1**, EWMA/median reach **~98% point recall** (F1 ~0.95) but **specificity ~0.01** on the tiny non-GT slice (~363 s). Event recall stays **0.60** (3/5 GT intervals) because predictions merge into few long events.

Secondary sweeps at **k=2** (forecast window / EWMA α) peaked at **F1 ~0.55** — much worse than lowering k.

## Top 10 by F1 (all cases)

| run_id | mode | phase | k | recall | F1 | TP | FN |
|--------|------|-------|---|--------|-----|----|----|
| ewma_k1_ea0p35_mw31_sw31_da0p995_k_sweep | ewma | k_sweep | 1.0 | 0.987 | 0.957 | 4678 | 64 |
| median_k1_ea0p35_mw31_sw31_da0p995_k_sweep | median | k_sweep | 1.0 | 0.983 | 0.955 | 4660 | 82 |
| savgol_k1_ea0p35_mw31_sw31_da0p995_k_sweep | savgol | k_sweep | 1.0 | 0.959 | 0.943 | 4549 | 193 |
| ewma_k1p5_ea0p35_mw31_sw31_da0p995_k_sweep | ewma | k_sweep | 1.5 | 0.913 | 0.923 | 4330 | 412 |
| median_k1p5_ea0p35_mw31_sw31_da0p995_k_sweep | median | k_sweep | 1.5 | 0.900 | 0.917 | 4268 | 474 |
| savgol_k1p5_ea0p35_mw31_sw31_da0p995_k_sweep | savgol | k_sweep | 1.5 | 0.838 | 0.880 | 3973 | 769 |
| median_k2_ea0p35_mw7_sw31_da0p995_median_window_sweep | median | median_window_sweep | 2.0 | 0.388 | 0.554 | 1839 | 2903 |
| median_k2_ea0p35_mw61_sw31_da0p995_median_window_sweep | median | median_window_sweep | 2.0 | 0.377 | 0.540 | 1789 | 2953 |
| ewma_k2_ea0p5_mw31_sw31_da0p995_ewma_alpha_sweep | ewma | ewma_alpha_sweep | 2.0 | 0.370 | 0.534 | 1756 | 2986 |
| ewma_k2_ea0p15_mw31_sw31_da0p995_ewma_alpha_sweep | ewma | ewma_alpha_sweep | 2.0 | 0.363 | 0.526 | 1721 | 3021 |
