# Research briefing for Direction 16: Machine unlearning vs poisoning/backdoors

## Executive summary

Your Direction 16 project sits at a very active—and surprisingly fragile—intersection: **security-motivated data poisoning/backdoors**, and **post-hoc machine unlearning** as a purported remediation mechanism. The foundational backdoor result (**BadNets**) demonstrates that a model can retain high “clean” accuracy while behaving maliciously on attacker-chosen triggered inputs, and that such behaviors can persist through subsequent fine-tuning. citeturn0search1turn0search9turn0search5 The recent anchor result (**Machine Unlearning Fails to Remove Data Poisoning Attacks**, published at entity["organization","ICLR","ml conference"] 2025) finds that multiple state-of-the-art *approximate* unlearning methods can appear effective under common evaluation settings yet **fail to remove poisoned-data effects**, motivating stronger evaluations and introducing poisoning-based audits (including the Gaussian Unlearning Score). citeturn1view1turn1view0turn4view0turn6view0

For a milestone-based course project, the most feasible path is to build a **compact “unlearning robustness” evaluation suite** around one backdoor threat model (e.g., BadNets-style patch triggers) and one dataset (e.g., CIFAR-10), then compare **two unlearning methods** (e.g., SCRUB vs SSD, or NegGrad+ vs SSD) against a **retraining oracle**. This is strongly course-aligned: it tests robustness under distributional “triggered” subpopulations, exposes optimization/regularization failure modes in post-hoc parameter updates, and emphasizes rigorous evaluation over superficial metrics. citeturn6view0turn11view0turn0search1turn7search2

A key thesis you can plausibly support by Milestone 3 is: **unlearning success is evaluation-dependent**, and naïve “forget” metrics can be consistent with **residual backdoor vulnerability**. This thesis is reinforced by adjacent work showing (i) unlearning pipelines can enable new poisoning dynamics (“camouflaged” attacks), and (ii) even verification strategies based on backdoor traces can be fragile against adaptive behavior. citeturn7search0turn12search0turn12search5turn12search1

## Research questions and threat model framing

A clean project story benefits from stating the threat model precisely, because “unlearning” means different things depending on whether your goal is **privacy** (erasing membership signal) or **integrity/security** (removing poisoned behaviors). citeturn9search0turn9search1turn1view0

**Core research questions (fit the course themes)**
1. **Does post-hoc unlearning truly remove backdoor behavior, or just reduce direct traceability of the forget set?** The “fails” paper argues that evaluations focusing on “direct influence” (e.g., membership inference adaptations) can miss “indirect influence” revealed by poisoning tests. citeturn1view0turn4view0turn6view0  
2. **Which evaluation probes are most sensitive to residual backdoors after unlearning?** You can treat this as an evaluation methodology project: define multiple probes, then report which ones detect residual vulnerability earliest or most reliably. citeturn9search1turn11view0turn12search0  
3. **How do optimization choices in unlearning (budget, learning rate/noise, retraining scope) trade off clean accuracy vs residual backdoor success rate?** The recent “fails” paper explicitly studies the dependence on task and compute budget, and argues budgets needed to fix poisoned-induced model shifts can exceed practical constraints. citeturn3view0turn1view0turn1view1

**Threat model choices that keep the project feasible**
- **Backdoor mechanism:** Start with a standard data-poisoning backdoor (BadNets-style patch trigger) where poisoned training samples contain a trigger and are labeled as an attacker-chosen target, yielding high clean accuracy but high triggered misclassification rate. citeturn0search1turn11view0turn5view0  
- **Unlearning setting:** Evaluate “idealized deletion” first: the unlearner is given **the full set of poisoned points** to delete. This matches the core evaluation setup of the “fails” paper (“given all poison samples as the forget set”) and isolates algorithmic capability rather than detection. citeturn1view0turn13search22turn6view0  
- **Optional realism knob:** Partial identification (only a subset of poison points known) turns this into **corrective unlearning**, which is formally differentiated from privacy unlearning and has distinct requirements. citeturn9search0turn9search4

## Key research to read and how it maps to your project

This section prioritizes papers that directly inform (a) unlearning method selection, (b) attack selection, and (c) evaluation design.

**Anchor paper: BadNets (backdoor injection and persistence)**
- BadNets frames backdoors as a **model supply chain** risk: a malicious trainer can return a model that behaves normally on clean data but misbehaves on triggered inputs, and backdoors can persist even after later retraining/fine-tuning. citeturn0search1turn0search9turn0search5  
- For your project, BadNets provides the canonical “clean accuracy + hidden malicious behavior” template that unlearning must remove.

**Anchor paper: Machine Unlearning Fails to Remove Data Poisoning Attacks (evaluation as the main battleground)**
- This paper evaluates multiple unlearning algorithms and finds none removes all poisoning effects across attacks/tasks, even under a compute budget allocation they describe as generous (a fixed fraction of original training compute). citeturn1view1turn3view0turn1view0  
- It explicitly argues that common evaluations often target “direct influence,” while poisoning reveals “indirect influence” that can persist. citeturn1view0turn4view0  
- It introduces a poisoning-based evaluation metric (Gaussian Unlearning Score / GUS) that yields efficient hypothesis-testing style auditing without shadow models. citeturn4view0turn6view0

**Unlearning methods and evaluation philosophy you can reuse**
- **Adversarial evaluations for inexact unlearning:** proposes the Interclass Confusion test and simple baselines (EU-k and CF-k), emphasizing adversarial evaluation rather than relaxed metrics. citeturn9search1turn9search5turn9search13  
- **Corrective machine unlearning:** formalizes the setting where only a subset of corrupted data is identified, arguing it differs from classical privacy-oriented unlearning. citeturn9search0turn9search4  
- **SISA training:** classic exact unlearning framework by limiting per-point influence through sharding/slicing; useful as a conceptual “gold standard class” when you discuss what true removal could require. citeturn2search5turn2search13  
- **SCRUB:** a prominent student–teacher framing for unlearning that appears in the unlearning literature and is implemented in OpenUnlearn. citeturn0search3turn3view2turn6view0  
- **NegGrad+:** a finetuning-based method that explicitly negates gradients on the forget set while training on retain data (as described in the “fails” paper’s appendix and implemented in OpenUnlearn). citeturn3view2turn4view0turn6view0  
- **SSD (Selective Synaptic Dampening):** a retraining-free, post-hoc method based on Fisher-information-style parameter importance, with an official implementation; extremely attractive for course feasibility because it provides a strong “fast unlearning” baseline. citeturn7search2turn7search6turn7search10  
- **Noisy/DP-flavored unlearning (Langevin unlearning):** formalizes noisy gradient descent as an unlearning mechanism and has an official implementation; this is a good optional third method if you want a “noise as regularization/forgetting” perspective aligned with your optimization lecture themes. citeturn9search2turn9search6turn8view1

**Security-specific adjacent work that strengthens the “unlearning vs backdoors” story**
- **Hidden Poison / camouflaged poisoning:** shows unlearning or retraining dynamics can enable new poisoning strategies that “activate” after deletions. citeturn7search0turn7search4  
- **Backdoor attacks via machine unlearning (entity["organization","AAAI","ai association"] 2024):** studies malicious backdoors injected through unlearning requests rather than classic training-time poisoning—useful to motivate why “unlearning as defense” must consider adaptive adversaries. citeturn2search2turn2search25turn8view0  
- **Clean-unlearning-triggered backdoors:** proposes attacks where deletion requests themselves can trigger backdoor behavior, reinforcing the theme that unlearning pipelines can create new attack surfaces. citeturn2search26turn2search32  
- **Poisoning attacks on certified unlearning:** demonstrates that even provable/certified unlearning systems can face integrity threats that force costly retraining under adversarial design. citeturn7search33

**Backdoor toolkits, benchmarks, and conventional defenses as “diagnostic probes”**
- **BackdoorBench:** benchmark + modular codebase with standardized protocols and metrics (Clean Accuracy, ASR, etc.). This is one of the most practical foundations for your project. citeturn11view0turn0search2turn0search10  
- **Dataset Security survey (journal / taxonomy):** provides a unified view of poisoning/backdoor threat models and defenses and is a strong citation backbone for your introduction. citeturn10search0turn10search4  
- **Neural Cleanse and Fine-Pruning:** classic post-training mitigation ideas; even if you do not “defend,” you can use them as *evaluation probes* (e.g., “can triggers still be reverse-engineered after unlearning?”). citeturn10search21turn10search2turn10search14turn10search33  
- **TrojAI program resources (optional extension):** entity["organization","NIST","us standards agency"] provides a Trojan detection leaderboard and research resources; helpful if you want to cite broader backdoor evaluation ecosystems, though it may be heavy for a course project. citeturn10search35turn10search7turn10search15

## Implementation resources and development aids

A central feasibility decision is whether you build from a **single integrated framework** or “compose” multiple repos. The path of least resistance is usually:

- **Backdoor creation + evaluation protocol:** BackdoorBench  
- **Unlearning algorithms + standard unlearning metrics:** OpenUnlearn  
- **Add your backdoor-specific residual vulnerability tests as custom evaluation code**

This gives you “batteries included” baselines without spending weeks re-implementing methods. citeturn11view0turn6view0turn0search2

### Minimal resource map

| Resource | What it gives you | Why it matters for D16 |
|---|---|---|
| entity["organization","BackdoorBench","backdoor benchmark toolkit"] (paper + codebase) | Standardized backdoor attacks/defenses, protocols, metrics like clean accuracy and ASR | Lets you implement BadNets-style patch triggers and evaluate attack success consistently citeturn11view0turn0search2turn0search10 |
| entity["organization","OpenUnlearn","unlearning eval library"] (code released with “fails” paper) | Implementations for multiple unlearning methods (GD/NGD/GA/EUk/CFk/SCRUB/NegGrad+/SSD) plus membership-style auditing metrics incl. GUS | Gives fast access to state-of-the-art approximate unlearning methods and evaluation hooks citeturn6view0turn1view0turn4view0 |
| SSD official code | Reference implementation for Selective Synaptic Dampening | Strong retraining-free baseline; useful contrast with retrain-based methods citeturn7search2turn7search6turn7search10 |
| Goel et al. code for adversarial evaluation | Code for Interclass Confusion testing and EU-k / CF-k baselines | Strong evaluation framing; baselines that are simple but meaningful citeturn9search1turn9search5turn9search13 |
| Δ-Influence code | Identifies poisoned points from affected behavior; supports multiple poisoning attacks incl. BadNet-style patch triggers | If you add “partial identification,” this becomes your realistic detection+unlearning pipeline citeturn5view0turn2search16 |
| ERASURE framework | Modular experimentation and reproducibility scaffolding for unlearning | Useful if you want standardized experiments and reporting; good for reproducibility story citeturn13search0turn13search10 |

### Practical integration notes

- **BackdoorBench already implements BadNets-style attack scripts** (e.g., “badnet attack” modules), which means you can avoid writing data poisoning glue from scratch. citeturn0search33turn11view0  
- **OpenUnlearn supports eight unlearning methods** and includes membership inference-based metrics and GUS; it is designed as an evaluation library, so you can extend it with your backdoor residual metrics. citeturn6view0turn4view0  
- If you want a single “unlearning method comparison” story without overengineering, **choose one retrain-heavy method (SCRUB or NegGrad+) and one retrain-free method (SSD)**. This yields a clean axis: “Does speed/approximation correlate with residual backdoor vulnerability?” citeturn6view0turn7search2turn3view2

## Evaluation suite design and novelty options

The single most important factor for D16 is that you do **not** evaluate unlearning solely by “forget-set” performance or classic MIAs; your core claim is precisely that those can be misleading in integrity problems. citeturn1view0turn9search1turn12search0

### Minimal evaluation suite you can defend rigorously

You can treat your evaluation suite as two layers:

**Layer A: standard backdoor behavior metrics**  
BackdoorBench explicitly measures at least clean accuracy (C-Acc) and attack success rate (ASR), and uses these to compare attack/defense pairs under standardized protocols. citeturn11view0turn0search6  
Recommended metrics:
- **Clean Accuracy (C-Acc):** accuracy on untriggered test data. citeturn11view0  
- **Attack Success Rate (ASR):** fraction of triggered test inputs classified as the attacker’s target. citeturn11view0turn10search0  
- **(Optional) Robust/repair accuracy (R-Acc):** if you want parity with BackdoorBench’s reporting; use only if you can define it consistently in your implementation. citeturn11view0

**Layer B: unlearning success metrics that can be “fooled”**  
OpenUnlearn provides membership-style auditing tools (standard threshold MIA, and GUS for poisoning-based auditing) and frames evaluation as hypothesis testing between “trained with x” vs “trained without x.” citeturn6view0turn4view0  
Recommended metrics:
- **MIA-style distinguishability between forget-set points and test points** (baseline audit of deletion). citeturn6view0turn1view0  
- **Compute/time to unlearn** (important in the “practical unlearning budget” frame). citeturn6view0turn3view0  
- **Oracle gap to retraining-from-scratch** (the “fails” paper repeatedly uses the idea that ideal unlearning should match retraining without poisons). citeturn1view0turn13search22

### Residual vulnerability probes (your likely novelty)

A solid “mini research contribution” is to add 2–3 additional probes that stress-test backdoor remnants beyond a single canonical trigger:

1. **Trigger family generalization test**  
Backdoor triggers may generalize under location shifts, small transformations, or intensity changes; testing a trigger family can detect whether unlearning only overfits to one trigger template. BackdoorBench discusses diverse trigger types and evaluates across poisoning ratios, suggesting systematic evaluation matters. citeturn11view0turn10search0  

2. **Post-unlearning trigger discoverability**  
Run a trigger reverse-engineering method (e.g., Neural Cleanse) on the unlearned model; if a compact trigger remains discoverable or yields strong targeted behavior, that is evidence of residual vulnerability. citeturn10search21turn10search33turn10search1  

3. **Adaptive attacker sanity check**  
Unlearning evaluation is known to be fragile under adversarial behavior in other contexts (verification strategies can be circumvented), so including one “adaptive” check strengthens your methodology section: e.g., after unlearning, attempt a small fine-tune on a tiny trigger set and see if the backdoor “reactivates” faster than in a clean model. The broader fragility of verification under adversarial settings is documented in unlearning verification work. citeturn12search0turn12search5turn12search1  

### Suggested experiment pipeline diagram

```mermaid
flowchart TD
  A[Train clean model on D_clean] --> B[Inject backdoor to form D_poisoned]
  B --> C[Train poisoned model f_poisoned]
  C --> D[Define forget set D_forget = poisoned points]
  D --> E1[Unlearning method U1 -> f_U1]
  D --> E2[Unlearning method U2 -> f_U2]
  D --> F[Oracle retrain on D_clean (or D_poisoned \ D_forget) -> f_retrain]

  E1 --> G[Evaluate: C-Acc, ASR, trigger family tests]
  E2 --> G
  F --> G
  C --> G

  G --> H[Compare: residual backdoor vs forget metrics vs compute]
```

## Pre-plan: recommended concrete configuration, feasibility, and risks

This is not yet a full development plan; it is a **research- and implementation-grounded pre-plan** designed to make writing Milestone 1–3 documents straightforward.

### Recommended minimal configuration (high likelihood of success)

**Dataset & model**
- Start with **CIFAR-10** on a standard CNN/ResNet variant used widely in both poisoning and backdoor benchmarks. CIFAR-10 appears in the “fails” paper’s vision experiments and is a standard in backdoor evaluations. citeturn1view0turn11view0  

**Backdoor attack**
- Use **BadNets-style patch trigger** (small patch in a corner, label set to target class) because it is canonical and directly supported in modern poisoning/backdoor toolchains and reported in poisoning-unlearning papers. citeturn0search1turn11view0turn5view0  

**Unlearning methods (2-method comparison, plus oracle retrain)**
- Method 1: **SCRUB or NegGrad+** (retrain-based / finetuning-based) via OpenUnlearn; both are described in the “fails” paper appendix and are directly callable in OpenUnlearn. citeturn3view2turn6view0turn4view0  
- Method 2: **SSD** (retrain-free) via either OpenUnlearn’s implementation or SSD’s official repo; this gives a strong contrast in compute assumptions. citeturn7search2turn6view0turn7search6  
- Baseline: **Full retraining** without the poisoned points (oracle) as the “should-be” target—consistent with the “fails” paper’s framing that ideal unlearning should match training as if poisons were never included. citeturn1view0turn13search22

**Compute budgeting**
- Treat unlearning compute as a fraction of initial training compute (the “fails” paper explicitly discusses allocating a fixed fraction of training compute to each unlearning method and notes even that can be large in modern settings). citeturn3view0turn1view1turn1view0  
- Expected compute level: **Medium** for a team project if you keep to CIFAR-10-scale models, reuse toolkits, and constrain sweeps to a few seeds.

### Optional “stretch” variant (if you finish early)

Add a corrective unlearning condition: you only “know” a subset of poisons, aligning with **Corrective Machine Unlearning** framing. This can be implemented by sampling only part of the attacked points as the forget set, or using a detection step from Δ-Influence to propose candidate poisons. citeturn9search0turn5view0turn13search14

### Milestone fit (practical narrative checkpoints)

- **Milestone 1 (abstract & plan):**  
  Your brief can cite that unlearning methods can fail to remove poisoning effects and that evaluation criteria can be insufficient; you propose an evaluation suite centered on ASR + trigger-family testing + MIA/GUS-style audits. citeturn1view0turn9search1turn11view0turn6view0  
- **Milestone 2 (background slidedeck):**  
  Background is naturally split into: (i) backdoors/poisoning taxonomy and metrics (BackdoorBench + dataset security survey), and (ii) unlearning formulations and evaluation pitfalls (Goel evaluations + “fails” paper). citeturn11view0turn10search0turn9search1turn1view0  
- **Milestone 3 (full slidedeck with prelim results):**  
  Minimal successful figure set is: **Clean accuracy vs ASR** across (poisoned, unlearned-1, unlearned-2, retrained oracle), plus **one residual probe** (trigger variations or Neural Cleanse discoverability). citeturn11view0turn10search21turn10search1  
- **Milestone 4 (essay):**  
  Essay can emphasize evaluation methodology: show where “forget metrics” and integrity metrics diverge, connect failures to optimization constraints/budget and to indirect influence concepts in the “fails” paper. citeturn1view0turn3view0turn4view0

### Risks and challenge mitigation

**Risk: “Unlearning success” depends on metric and can look good superficially**  
This is the point of your project, but it can also create ambiguity if you don’t pre-register what counts as success. The literature explicitly warns that heuristic evaluations can be misleading and advocates broader perspectives. citeturn1view0turn9search1  
Mitigation: Define success as “approaches retraining oracle” on (i) C-Acc and (ii) ASR and (iii) at least one residual probe.

**Risk: Implementation scope blow-up**  
Backdoor research contains many attack/defense variants; BackdoorBench itself spans many attacks/defenses and settings. citeturn11view0turn0search6  
Mitigation: fix one dataset + one architecture + one trigger type; allow only 1–2 hyperparameter sweeps per method.

**Risk: Unlearning pipelines can be attacked or produce unexpected behavior**  
Work on camouflaged poisoning and unlearning-request-based backdoors show the unlearning process itself can introduce new attack surfaces. citeturn7search0turn2search2turn2search26  
Mitigation: keep your core claim scoped to “unlearning as defense against known training-time backdoor poisons,” and treat adaptive/unlearning-request attacks as “related work + optional stress test.”

**Risk: Verification and auditing can be fragile under adversarial settings**  
If you rely on backdoor-style verification alone, there is evidence verification strategies can be circumvented or are fragile, motivating multi-probe evaluation. citeturn12search0turn12search5turn7search27  
Mitigation: do not treat any single probe as definitive; triangulate with 2–3 probes and always compare to the retraining oracle.

**Risk: Compute budget constraints**  
The “fails” paper emphasizes that even a seemingly small fraction of training compute can be substantial at scale, and that poisoned-induced shifts may require more updates than practical approximate unlearning budgets allow. citeturn3view0turn1view1  
Mitigation: keep models small (CIFAR-scale), keep unlearning steps fixed (budgeted), and report results as a function of unlearning budget so negative results are informative rather than inconclusive.

