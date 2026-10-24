// Shape of results.json written by scripts/run_pipeline.py.
// Keep in sync with build_results() in that script.

export type MetricName = "accuracy" | "balanced_accuracy" | "precision" | "recall" | "f1";

export interface MetricRow extends Record<MetricName, number> {
  model: string;
}

export interface BuildInfo {
  generated_at: string;
  duration_seconds: number;
  source: "github-actions" | "local";
  commit_sha: string | null;
  commit_message: string | null;
  commit_author: string | null;
  commit_time: string | null;
  branch: string | null;
  repository: string | null;
  commit_url: string | null;
  run_url: string | null;
  python: string;
  packages: Record<string, string | null>;
}

export interface Settings {
  random_state: number;
  test_size: number;
  stratify: boolean;
  smote_k_neighbors: number;
  dropped_classes: string[];
  zero_day_source_classes: string[];
  zero_day_label: string;
  dropped_columns: string[];
  xgboost_params: Record<string, string | number | null>;
  lightgbm_params: Record<string, string | number | null>;
  calibration_method: string;
  calibration_cv: number;
  fl_clients: number;
  fl_non_iid_alpha: number;
}

export interface EfficiencyRow {
  model: string;
  inference_time_per_sample_ms: number | null;
  memory_footprint_kb: number | null;
  total_inference_time_s: number | null;
  training_time_s: number | null;
}

export interface FeatureImportance {
  feature: string;
  importance?: number;
  mean_abs_shap?: number;
}

export interface Results {
  schema_version: number;
  build: BuildInfo;
  settings: Settings;
  stages: { calibration: boolean; shap: boolean; federated: boolean };
  paper: { title: string; venue: string; benchmark_targets: Partial<Record<MetricName, number>> };
  dataset: {
    file: string;
    raw_rows: number;
    raw_columns: number;
    raw_class_counts: Record<string, number>;
    rows_after_cleaning: number;
    class_counts_after_cleaning: Record<string, number>;
    class_counts_after_smote: Record<string, number>;
    n_features: number;
    feature_names: string[];
    train_rows: number;
    test_rows: number;
    classes: string[];
  };
  metrics: MetricRow[];
  final_model: string;
  paper_comparison: { metric: MetricName; paper: number; reproduction: number; difference: number }[];
  efficiency: EfficiencyRow[];
  confusion_matrix: { model: string; labels: string[]; matrix: number[][] };
  per_class: { class: string; precision: number; recall: number; f1: number; support: number }[];
  feature_importance_rf: FeatureImportance[];
  shap_top_features: Record<string, FeatureImportance[]>;
  federated: null | {
    n_clients: number;
    non_iid_alpha: number;
    shard_sizes: number[];
    metrics: Record<MetricName, number>;
    mean_pairwise_spearman: number | null;
  };
}
