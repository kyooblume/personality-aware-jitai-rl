# Personality-Aware JITAI with Reinforcement Learning
### 誠実性を考慮した JITAI（ジャストインタイム適応型介入）の強化学習による最適化

> Undergraduate research project (Apr–Jul 2025) · Toda Lab, Faculty of Data Science, Yokohama City University
> 横浜市立大学 データサイエンス学部 戸田研究室 学部研究（2025年4〜7月）
> *Simulation-based, unpublished student research. No human-subject data is used.*

**TL;DR** — I built a simulation of a person working toward a task deadline, whose response to interventions depends on their **conscientiousness** (a Big Five trait). A **PPO** agent learned *when* and *how strongly* to intervene. In 100 evaluation episodes per personality level, the learned policy completed the task every time and had the highest average reward of all policies compared.

---

## 1. Background / 背景

- Just-in-Time Adaptive Interventions (JITAI) deliver support (e.g. smartphone prompts) when it is most likely to help (Nahum-Shani et al., 2018).
- Personality is linked to health behaviour: for example, more conscientious people tended to increase their daily step counts more during a physical-activity intervention (Stieger et al., 2020).
- RL-based JITAI simulators already exist — e.g. **StepCountJITAI**, which learns intervention timing to promote physical activity (step counts) (Karine & Marlin, 2024).
- **Gap:** few studies build personality traits directly into the JITAI timing model.

**Relation to step counts:** step-count research motivated this project, but the simulation here models a **generic task to be completed before a deadline**, not step counts.

## 2. Research question / 研究目的

Can an RL agent learn an intervention policy that adapts to a person's conscientiousness, and how does it compare with fixed intervention strategies?

個人の誠実性に合わせて「いつ・どの強さで」介入するかを強化学習で学習し、固定的な介入方針と比較する。

## 3. Method / 手法

**Simulation environment** — custom `gymnasium` environment `SimplifiedTaskJITAIEnv`, inspired by StepCountJITAI and adapted to task completion.

| Element | Description |
|---|---|
| Episode | Up to 90 time steps to finish a task |
| Actions | `0` no intervention · `1` weak intervention · `2` strong intervention |
| State | Disengagement risk **D**, effort **S**, task progress **TP**, conscientiousness **CO**, time remaining, … |
| Person model | Conscientiousness (CO = 0.2 / 0.5 / 0.8) changes start probability, disengagement, urgency and response to interventions |
| Behavioural effects | Habituation to repeated interventions, urgency near the deadline, disengagement after repeated inaction, intervention cost |
| Reward | Progress made + bonus for completion − intervention cost |
| Episode ends | Task completed / disengaged (D over threshold) / time out |

**Agent** — PPO (`stable-baselines3`), 50,000 training timesteps per CO level.
**Baselines** — fixed policies that always choose action 0, 1 or 2.
**Evaluation** — 100 episodes per policy × CO level; average reward, completion rate, average steps.
**Sensitivity analysis** — urgency, habituation and intervention-cost parameters (`02_sensitivity_analysis.ipynb`).

## 4. Results / 結果

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

*Values from the output of `notebooks/01_main_rl_vs_fixed_policies.ipynb`.*

**Key findings / 主な発見**
- In this simulation the RL agent completed the task in all evaluation episodes for every CO level, with the highest average reward.
- Low-conscientiousness agents needed support most: with no intervention none completed the task and 16% disengaged.
- For mid / high conscientiousness, strong interventions finished fastest but scored lower because of their cost.

## 5. Limitations / 限界

- **Simulation only** — behaviour parameters were set by hand from the literature and checked with sensitivity analysis; they are not fitted to real data.
- Only one trait (conscientiousness) at three levels.
- Single training run per CO level (50k timesteps); robustness across random seeds is not tested.
- Results come from an unpublished student project and have not been peer-reviewed.

## 6. Repository structure / 構成

```
.
├── notebooks/
│   ├── 01_main_rl_vs_fixed_policies.ipynb   ← ★ start here / まずこれを見る（最終版）
│   └── 02_sensitivity_analysis.ipynb        ← parameter sensitivity & calibration
├── experiments/                             ← earlier experiment logs (research history / 試行錯誤の記録)
│   ├── 2025-06-04_initial_ppo.ipynb
│   ├── 2025-06-05_env_v2.ipynb
│   ├── 2025-06-19_habituation.ipynb
│   └── 2025-07-03_previous_cost_setting.ipynb
├── requirements.txt
└── README.md
```

## 7. How to run / 実行方法

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_main_rl_vs_fixed_policies.ipynb
```

## Tech
Python · Gymnasium · Stable-Baselines3 (PPO) · pandas · matplotlib · seaborn

## References
- Nahum-Shani, I. et al. (2018). Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health. *Annals of Behavioral Medicine*.
- Stieger, M. et al. (2020). The relationship of personality and behavior change in a physical activity intervention. *Personality and Individual Differences*.
- Karine, K. & Marlin, B. M. (2024). StepCountJITAI: simulation environment for RL with application to physical activity adaptive intervention. NeurIPS 2024 Workshop on Behavioral ML. [arXiv:2411.00336](https://arxiv.org/abs/2411.00336)
