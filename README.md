# step-jitai-rl-research — 卒業研究の実験ノート / Undergraduate Research Notebooks

> 横浜市立大学データサイエンス学部 戸田研究室での卒業研究の実験コードです。
> Experiment code for my undergraduate research at Toda Lab, Faculty of Data Science, Yokohama City University.

## 概要 / Overview

**日本語:** 歩数増加を目的とした JITAI（Just-in-Time Adaptive Intervention）を強化学習で最適化するシミュレーション研究。Gymnasium で介入環境を作り、Stable-Baselines3 の PPO で方策を学習して、固定ポリシーと比較しています。

**English:** Simulation study optimising a step-count JITAI with reinforcement learning. A custom Gymnasium environment is trained with PPO (Stable-Baselines3) and compared with fixed policies.

## ノートブック / Notebooks

| ファイル | 内容 |
|---|---|
| `20250604.ipynb`, `20250605.ipynb`, `20250619..ipynb` | 初期の実験 |
| `20250703.ipynb` | 固定ポリシーとPPOの比較（1,000回 / 50,000回試行） |
| `20250703 - コピー.ipynb` | 上の作業用コピー |
| `感度分析  離脱多め.ipynb` | 離脱が多い条件での感度分析 |

## 技術 / Tech
Python, Gymnasium, Stable-Baselines3 (PPO), pandas, matplotlib

## 関連 / Related
- `StepJITAI` — ベースにした StepCountJITAI 環境
- `power_analysis` — 検定力分析
