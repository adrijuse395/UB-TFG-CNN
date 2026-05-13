export const PLOT_CONFIG = { responsive: true, displaylogo: false };
export const LIVE_REFRESH_MS = 8000;

export const AXIS_FIELDS = [
  { key: "rank_scalar", label: "Rank" },
  { key: "accuracy", label: "Accuracy (%)" },
  { key: "compression_ratio", label: "Compression ratio (x)" },
  { key: "latency_ms", label: "Latency (ms)" },
  { key: "throughput_fps", label: "Throughput (samples/s)" },
  { key: "macs_g", label: "GMACs" },
  { key: "total_parameters", label: "Parameters" },
  { key: "compression_time_s", label: "Compression time (s)" },
  { key: "model_memory_mb", label: "Model weights (MiB)" },
  { key: "peak_inference_memory_mb", label: "Peak CUDA test pass (MiB)" },
  { key: "test_eval_time_s", label: "Test eval time (s)" },
  { key: "fine_tuning_time_s", label: "Fine-tune time (s)" },
];

export const SERIES_OPTIONS = [
  { key: "fine_tuning_enabled", label: "Fine-tuning" },
  { key: "experiment_name", label: "Experiment" },
  { key: "phase", label: "Phase" },
  { key: "method", label: "Method" },
  { key: "run_id", label: "Run" },
];

export const LS_COMPARE_SOURCES = "analyzer_compare_sources_v2";
