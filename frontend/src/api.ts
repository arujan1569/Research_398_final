import type { Health, ModelId, ModelInfo, Prediction } from './types';

export const API_BASE = (import.meta as unknown as { env: { VITE_API_URL?: string } }).env.VITE_API_URL?.replace(/\/$/, '') || '';
async function read<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed (${response.status}). Please try again.`;
    try { const error = await response.json(); if (typeof error.detail === 'string') message = error.detail; } catch { /* Keep the HTTP error. */ }
    throw new Error(message);
  }
  return response.json();
}
export async function getStatus(signal?: AbortSignal): Promise<[Health, ModelInfo[]]> {
  return Promise.all([
    fetch(`${API_BASE}/api/health`, { signal }).then(read<Health>),
    fetch(`${API_BASE}/api/models`, { signal }).then(read<ModelInfo[]>),
  ]);
}
export async function predict(file: File, model: ModelId, signal: AbortSignal): Promise<Prediction> {
  const body = new FormData(); body.append('file', file); body.append('model', model);
  return read<Prediction>(await fetch(`${API_BASE}/api/predict`, { method: 'POST', body, signal }));
}
