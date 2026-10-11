# Personality-Aware JITAI with Reinforcement Learning
### 誠実性を考慮した JITAI（ジャストインタイム適応型介入）の強化学習による最適化

> Undergraduate research · Toda Lab, Faculty of Data Science, Yokohama City University (2025)
> 横浜市立大学 データサイエンス学部 戸田研究室 卒業研究

**TL;DR** — I built a simulation of a person working toward a task deadline, whose response to reminders depends on their **conscientiousness** (a Big Five trait). A **PPO** agent learned *when* and *how strongly* to intervene. Across low / medium / high conscientiousness, the learned policy completed the task in **100% of evaluation episodes** and achieved the **highest reward** of all policies, outperforming every fixed intervention strategy.

---

## 1. Research question / 研究目的

Just-in-Time Adaptive Interventions (JITAI) send support (e.g. reminders) at the moment it is most useful. Too few interventions → people disengage; too many → they get used to them (habituation) and interventions cost effort.

**Question:** Can a reinforcement-learning agent learn an intervention policy that adapts to a person's conscientiousness, and does it beat simple fixed strategies?

**問い：** 個人の誠実性（性格特性）に合わせて「いつ・どの強さで」介入するかを強化学習で学習し、固定的な介入方針より良い結果を出せるか。

## 2. Method / 手法

**Simulation environment** — custom `gymnasium` environment `SimplifiedTaskJITAIEnv`, inspired by [StepCountJITAI](https://arxiv.org/abs/2411.00336) (Karine & Marlin, 2024) and adapted to a task-completion setting.

| Element | Description |
|---|---|
| Episode | Up to 90 time steps to finish a task |
| Actions | `0` no intervention · `1` weak intervention · `2` strong intervention |
| State | Disengagement risk **D**, effort **S**, task progress **TP**, conscientiousness **CO**, time remaining, … |
| Person model | Conscientiousness (CO = 0.2 / 0.5 / 0.8) changes start probability, disengagement, urgency and response to interventions |
| Realistic effects | **Habituation** (repeated interventions lose effect), **urgency** near the deadline, **disengagement** after repeated inaction, **intervention cost** |
| Reward | Progress made + bonus for completion − intervention cost |
| Episode ends | Task completed / disengaged (D over threshold) / time out |

**Agent** — PPO (`stable-baselines3`), 50,000 training timesteps per CO level.

**Baselines** — fixed policies that always choose action 0, 1 or 2.

**Evaluation** — 100 episodes per policy × CO level: average reward, completion rate, disengagement rate, steps to completion.

## 3. Results / 結果

| CO | Policy | Avg. reward | Completion | Avg. steps |
|---|---|---:|---:|---:|
| 0.2 (low) | Fixed: none | 0.5 | 0% | 85.4 |
| | Fixed: weak | 51.2 | 77% | 84.0 |
| | Fixed: strong | 56.3 | 96% | 58.8 |
| | **RL agent (PPO)** | **79.9** | **100%** | 75.4 |
| 0.5 (mid) | Fixed: none | 5.7 | 0% | 89.7 |
| | Fixed: weak | 82.3 | 100% | 75.5 |
| | Fixed: strong | 69.0 | 100% | 51.0 |
| | **RL agent (PPO)** | **96.1** | **100%** | 75.6 |
| 0.8 (high) | Fixed: none | 42.9 | 31% | 88.9 |
| | Fixed: weak | 87.7 | 100% | 64.7 |
| | Fixed: strong | 75.4 | 100% | 44.7 |
| | **RL agent (PPO)** | **102.9** | **100%** | 68.9 |

**Key findings / 主な発見**
- The RL agent reached **100% completion for every personality type** with the highest reward — it balanced task success against intervention cost better than any fixed rule.
- **Low-conscientiousness** people need support most: without intervention none completed the task, and 16% disengaged.
- For **mid / high conscientiousness**, *strong* interventions were over-kill: they finished fastest but lowered reward because of their cost — weak or well-timed interventions were enough.
- → One-size-fits-all intervention is inefficient; **personalising by personality trait matters.**

## 4. Limitations / 限界

- **Simulation only** — behaviour parameters were set by hand and checked with sensitivity analysis, not fitted to real user data.
- Only **three CO levels** and one personality trait; other traits and real-world noise are not modelled.
- Training budget is modest (50k timesteps) and results come from a single training run per CO level.
- Next steps: calibrate with real wearable / behaviour data, compare other RL algorithms, and test robustness across random seeds.

## 5. Repository structure / 構成

```
.
├── notebooks/
│   ├── 01_main_rl_vs_fixed_policies.ipynb   ← ★ start here / まずこれを見る
│   └── 02_sensitivity_analysis.ipynb        ← parameter sensitivity & calibration
├── experiments/                             ← earlier experiment logs (kept as research history)
│   ├── 2025-06-04_initial_ppo.ipynb
│   ├── 2025-06-05_env_v2.ipynb
│   ├── 2025-06-19_habituation.ipynb
│   └── 2025-07-03_previous_cost_setting.ipynb
├── requirements.txt
└── README.md
```

## 6. How to run / 実行方法

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_main_rl_vs_fixed_policies.ipynb
```

## Tech
Python · Gymnasium · Stable-Baselines3 (PPO) · pandas · matplotlib · seaborn

## Reference
- Karine & Marlin (2024). *StepCountJITAI: simulation environment for RL with application to physical activity adaptive intervention.* NeurIPS 2024 Workshop on Behavioral ML. [arXiv:2411.00336](https://arxiv.org/abs/2411.00336)
