# Architecture Plan: Markov Sequence Selection & Late Acceptance Hyper-Heuristic

This document details the architectural plan for integrating **Ahmed Kheiri's winning ROADEF 2016 methodology** (*"Heuristic Sequence Selection for Inventory Routing Problem"*, Transportation Science 2020) with our **exact continuous LP/MIP mathematical programming solvers** (HiGHS / Gurobi).

---

## 1. Executive Summary & Core Diagnosis

### The Generalization Problem
The current native solver struggles to generalize cold starts across diverse Set B instances because:
1. **Greedy Single-Move Traps**: [`src/vrp_solver/solver/surgical_search.py`](../src/vrp_solver/solver/surgical_search.py) evaluates one atomic operator at a time and requires an immediate reduction in total error/deficit. In tightly constrained instances (e.g. V2.13, V2.14, V2.19 with layover and trailer coupling), a single move cannot resolve a defect without temporarily causing a deficit elsewhere before a follow-up move completes the transformation.
2. **Inner-Loop LP Bottleneck**: Solving full-horizon continuous quantity LP at every single micro-candidate slows candidate evaluation throughput down to ~100–200 per minute and discards promising structural routes if intermediate quantities are not immediately feasible.

### The Kheiri Solution
1. **Sequence-Based Selection Hyper-Heuristic (SSHH)**: Uses a **Markov Transition Matrix** ($M$) to dynamically learn and generate chains of complementary low-level heuristics (LLHs):
   $$\text{Ruin / Mutational} \longrightarrow \text{Recreate / Insertion} \longrightarrow \text{Local Search / Repair}$$
2. **Late Acceptance Hill Climbing (LAHC)**: Accepts moves if their cost/error is better than or equal to the solution **$L$ iterations ago** ($L \approx 50\text{--}200$), allowing the search to cross intermediate non-improving valleys.
3. **Decoupled Exact LP/MIP Polish**: Fast heuristic simulation in the sequence exploration inner loop, reserving exact full-horizon HiGHS/Gurobi LP/MIP calls for accepted sequence milestones and elite solutions.

---

## 2. Component Architecture

```mermaid
flowchart TD
    subgraph Markov_Engine ["1. Markov Sequence Selection (SSHH)"]
        State[Current Solution State S] --> HMM[Adaptive Markov Transition Matrix M]
        HMM -->|Select Sequence| Seq[Heuristic Sequence: h1 -> h2 -> ... -> hk]
        Seq --> Apply[Apply Low-Level Heuristics]
        Apply --> S_cand[Candidate Solution S']
    end

    subgraph Evaluation_Acceptance ["2. Evaluation & Late Acceptance (LAHC)"]
        S_cand --> FastEval[Fast Structural & Simulation Scoring]
        FastEval --> LAHC{Score(S') <= Buffer[iter % L] ?}
        LAHC -->|Yes: Accept| Accept[Accept S' -> S]
        LAHC -->|No: Reject| Reject[Reject S']
        Accept --> UpdateM_Reward[Reward Transitions in Sequence M[hi, hj] += delta]
        Reject --> UpdateM_Decay[Decay / Penalize Transitions in M]
        Accept --> UpdateBuffer[Buffer[iter % L] = Score(S')]
    end

    subgraph Exact_Polish ["3. Exact LP / MIP Solver Polish"]
        Accept --> Milestone{Feasible / New Best / Stagnation?}
        Milestone -->|Yes| HiGHS[Joint Timing MIP + Full-Horizon Quantity LP]
        HiGHS --> Check[Local Rules Validation & Official ROADEF Checker]
    end
```

---

## 3. Detailed Specifications

### A. Low-Level Heuristic (LLH) Catalog
Heuristics are categorized into three functional tiers:

1. **Tier 1: Ruin / Mutational Heuristics (Disruption)**
   * `LLH_RuinRandomShifts`: Completely unassign $k$ random shifts ($k \in [1, 4]$).
   * `LLH_RuinPressureCustomers`: Remove all deliveries for the top $k$ customers under high runout pressure.
   * `LLH_EjectCustomerFromShift`: Remove a single customer delivery from a driver shift.
   * `LLH_SplitShift`: State-preserving split of a long driving shift into two legal segments.

2. **Tier 2: Recreate / Insertion Heuristics (Construction)**
   * `LLH_ClusterGreedyInsert`: Re-insert unassigned customers using spatial/temporal proximity.
   * `LLH_PressureUrgencyInsert`: Insert deliveries for customers nearest to tank runout deadline.
   * `LLH_CrossShiftReinsert`: Insert ejected operations into compatible adjacent driver shifts.
   * `LLH_TrailerBlockRecombine`: Reassign whole trailer assignment blocks to feasible drivers.

3. **Tier 3: Local Search / Repair Heuristics (Intensification)**
   * `LLH_PointSwap`: Swap delivery order of two points within a shift.
   * `LLH_TimingShift2Opt`: Adjust arrival and departure timestamps within shift time windows.
   * `LLH_FastQuantityHeuristic`: Direct proportional fill calculation before exact LP.

---

### B. Adaptive Markov Transition Model
Let $\mathcal{H} = \{h_1, h_2, \dots, h_m\}$ be the set of low-level heuristics.
The transition matrix $M \in \mathbb{R}^{m \times m}$ stores transition weights.

1. **Transition Probability**:
   $$P(h_j \mid h_i) = \frac{M[i, j]}{\sum_{k=1}^m M[i, k]}$$
2. **Sequence Generation**:
   * Start at a chosen or random Tier 1 (Ruin) operator $h_{\text{start}}$.
   * Sample next operator from row $h_{\text{start}}$ in $M$.
   * Continue until a Tier 3 (Local Search) operator executes or max sequence length $K_{\max} \approx 4$ is reached.
3. **Reinforcement Learning Update**:
   * If sequence $\sigma = (h_{(1)}, h_{(2)}, \dots, h_{(k)})$ leads to an **accepted** solution under LAHC:
     $$M[h_{(r)}, h_{(r+1)}] \leftarrow M[h_{(r)}, h_{(r+1)}] + \delta_{\text{accept}} \quad \forall r \in [1, k-1]$$
   * If the sequence achieves a **new global best**:
     $$M[h_{(r)}, h_{(r+1)}] \leftarrow M[h_{(r)}, h_{(r+1)}] + \delta_{\text{best}}$$
   * At the end of each round, apply decay:
     $$M[i, j] \leftarrow \max(M_{\min}, (1 - \alpha) \cdot M[i, j])$$

---

### C. Late Acceptance Hill Climbing (LAHC)
* Maintain a circular array $\text{Buffer}$ of size $L$ (default $L = 50$).
* Initialize $\text{Buffer}[i] = \text{Score}(S_{\text{initial}})$ for all $i \in [0, L-1]$.
* At iteration $t$, evaluate candidate $S'$:
  $$\text{threshold} = \text{Buffer}[t \pmod L]$$
  $$\text{Accept if } \text{Score}(S') \le \text{threshold}$$
* If accepted: $S \leftarrow S'$.
* Always update the buffer: $\text{Buffer}[t \pmod L] = \text{Score}(S)$.

---

### D. Exact MIP/LP Polish Strategy
* **Inner Loop**: Uses ultra-fast Python/Numba replay to evaluate structural violations (`DRI03`, `LAY02`, `SHI04`) and rough tank runout penalties.
* **Milestone Triggers**:
  1. Whenever a candidate achieves zero structural errors in fast simulation.
  2. Every $N_{\text{polish}} \approx 25$ accepted LAHC moves.
  3. Search stagnation (no improvement for $N_{\text{stag}} \approx 100$ iterations).
* **Exact Solvers**:
  * [`highs_time_opt.py`](../src/vrp_solver/highs_time_opt.py) / [`joint_block_timing.py`](../src/vrp_solver/joint_block_timing.py): Exact arrival/departure/break timing LP/MIP.
  * [`highs_repair.py`](../src/vrp_solver/highs_repair.py): Exact multi-period tank inventory dynamics and delivery quantity optimization.

---

## 4. Implementation Phasing

1. **Phase 1: LAHC Acceptance Buffer**:
   * Implement circular buffer acceptance in [`surgical_search.py`](../src/vrp_solver/solver/surgical_search.py) to replace the strict greedy acceptance threshold.
2. **Phase 2: Markov Transition Matrix Engine**:
   * Create `src/vrp_solver/solver/markov_sequence.py` managing the $M$ matrix, sequence generator, and reward/decay updates.
3. **Phase 3: LLH Tier Packaging**:
   * Standardize existing mutation, ruin, insertion, and split operators into uniform LLH interfaces returning modified shift sets.
4. **Phase 4: Set B Generalization Benchmark**:
   * Benchmark against the full Set B corpus (V2.12 – V2.26) under `solve_native.sh` with fail-closed official checker verification.
