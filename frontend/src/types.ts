export type ModelId = 'simple_cnn' | 'mobilenetv3';
export interface ModelInfo {
  id: ModelId; name: string; description: string; available: boolean; status_message: string | null;
  class_names: string[]; input_shape: number[]; preprocessing: Record<string, unknown>;
  parameter_count: number | null; size_bytes: number | null; sha256: string | null;
  split_fingerprint?: string | null;
  metrics: { accuracy: number; macro_f1: number; balanced_accuracy: number; images: number } | null;
}
export interface Health { status: string; models: Record<ModelId, boolean>; tensorflow_version: string | null; max_upload_bytes: number }
export interface Prediction {
  request_id: string; created_at: string; model: ModelId; model_name: string; predicted_class: string;
  prediction_score: number; top_3_predictions: { class_name: string; score: number }[];
  all_predictions: { class_name: string; score: number }[];
  image: { width: number; height: number; format: string; size_bytes: number };
  input_shape: number[]; timings: { preprocessing_ms: number; inference_ms: number; total_ms: number };
  model_sha256: string; note: string;
}
