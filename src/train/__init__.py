'''Training, evaluation, and experiment-orchestration package.

Modules
  metrics    : scale-free forecast accuracy metrics (MASE, RMSSE, ...) + DM test.
  cv         : rolling-origin cross-validation fold generator.
  dataset    : turn the (N, T, C) feature tensor into supervised windows.
  trainer    : PyTorch training loop (point + quantile), leakage-safe per fold.
  experiment : orchestrator tying panel + tensor + graph + models + CV + metrics.

All fold-fitted objects (scalers, graphs, STL decompositions) are refit on the
training slice of each fold only (see design docs §5, §6, §12).
'''
