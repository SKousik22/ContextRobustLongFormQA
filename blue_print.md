# RAG Retrieval Strategy Experiments: ELI5 Long-Form QA

This document outlines a series of experiments designed to evaluate how different retrieval strategies impact the generation quality of Large Language Models (LLMs) in a Retrieval-Augmented Generation (RAG) pipeline.

The task focuses on Long-Form Question Answering (Explain Like I'm 5), utilizing a custom dataset. The experiments are inspired by findings from the paper *"The Power of Noise: Redefining Retrieval for RAG Systems"*, which highlights that the relevance, position, and number of passages in a prompt context significantly impact LLM performance.

To rigorously stress-test extended context windows and the "lost in the middle" phenomenon, these experiments will scale across three distinct context loads: 9, 19, and 29 injected documents.

To isolate the impact of the core information source, these experiments are divided into two main categories:

- **Category A (Document-Grounded):** The LLM is provided with the "Gold" document ($G$), which contains the necessary information to answer the query, but it is not given the explicit, concise answer.
- **Category B (Answer-Grounded):** The LLM is provided with the explicit, concise "Answer" ($A$) directly, rather than the Gold document.

### Notation

| Symbol | Description                                                                                                   |
| ------ | ------------------------------------------------------------------------------------------------------------- |
| $G$  | Gold document — contains the information needed to answer the query                                          |
| $A$  | Answer — the explicit, concise ground-truth answer                                                           |
| $S$  | Distractor document — topically related but does not answer the query; sourced from the `distractor` array |
| $R$  | Noise document — completely unrelated, randomly sampled from different query entries in the dataset          |

### Evaluation Metrics

The following baseline metrics are used to assess generation quality across all experiments:

- **BERTScore (Primary):** Measures semantic similarity between the generated ELI5 explanation and the reference answer $A$. Captures meaning preservation better than surface-level overlap.
- **ROUGE-L (Secondary):** Measures longest common subsequence overlap. Used as a fast sanity-check for gross failures rather than a primary signal.
- **Faithfulness Score (Category A only):** An LLM-as-judge prompt assesses whether the generated explanation contains claims unsupported by the Gold document $G$. Particularly important in experiments where the model must synthesize rather than extract.
- **PropRecall (Custom-made for these experiments):** It is a Recall version of FActScore, where the facts to be recalled are obtained from the reference answer (the ground truth in this case).
- ** Average Contextual Similarity (Custom-made for these experiments):** It's a custom-made metric that gives a measure of contextual similarity between reference answers and LLM-generated answers.
- ** Weighted PropRecall (Custom-made):** It's just weighting the proportional recall of each proportion, and then computing the average.

---

## Category A: Document-Grounded Experiments

### Experiment 1A: Impact of Distracting Documents (Document-Grounded)

**Objective:** To measure how the inclusion of topically related but non-answering documents ("distractors") degrades the LLM's ability to generate an accurate ELI5 explanation. Distracting documents ($S$) are sourced from the `distractor` array.

**Methodology:** We keep the Gold document ($G$) in a fixed position immediately preceding the user query, and incrementally introduce distracting documents ($S$) before it.

**Test Variations:**

| Variation                | Layout                               |
| ------------------------ | ------------------------------------ |
| 0 Distractors (Baseline) | `[ Instruction, G, Query ]`        |
| 4 Distractors            | `[ Instruction, 4×S, G, Query ]`  |
| 9 Distractors            | `[ Instruction, 9×S, G, Query ]`  |
| 14 Distractors           | `[ Instruction, 14×S, G, Query ]` |
| 19 Distractors           | `[ Instruction, 19×S, G, Query ]` |
| 24 Distractors           | `[ Instruction, 24×S, G, Query ]` |
| 29 Distractors           | `[ Instruction, 29×S, G, Query ]` |

---

### Experiment 2A: Impact of Gold Positioning (Document-Grounded)

**Objective:** To evaluate the "lost in the middle" phenomenon in long-form reasoning across different context lengths.

**Methodology:** We test three different context sizes (9, 19, and 29 distracting documents). For *each* context size, we shift the position of the Gold document ($G$) across distinct depth intervals while the total number of distractors remains fixed.

#### Case 1: 9 Distractors (10 Total Documents)

| Position | Depth | Layout                                    |
| -------- | ----- | ----------------------------------------- |
| 1        | 0%    | `[ Instruction, G, 9×S, Query ]`       |
| 3        | 50%   | `[ Instruction, 4×S, G, 5×S, Query ]` |
| 5        | 100%  | `[ Instruction, 9×S, G, Query ]`       |

#### Case 2: 19 Distractors (20 Total Documents)

| Position | Depth | Layout                                     |
| -------- | ----- | ------------------------------------------ |
| 1        | 0%    | `[ Instruction, G, 19×S, Query ]`       |
| 2        | 25%   | `[ Instruction, 4×S, G, 15×S, Query ]` |
| 3        | 50%   | `[ Instruction, 9×S, G, 10×S, Query ]` |
| 4        | 75%   | `[ Instruction, 14×S, G, 5×S, Query ]` |
| 5        | 100%  | `[ Instruction, 19×S, G, Query ]`       |

#### Case 3: 29 Distractors (30 Total Documents)

| Position | Depth | Layout                                      |
| -------- | ----- | ------------------------------------------- |
| 1        | 0%    | `[ Instruction, G, 29×S, Query ]`        |
| 2        | 12.5% | `[ Instruction, 3×S, G, 26×S, Query ]`  |
| 3        | 25%   | `[ Instruction, 7×S, G, 22×S, Query ]`  |
| 4        | 37.5% | `[ Instruction, 10×S, G, 19×S, Query ]` |
| 5        | 50%   | `[ Instruction, 14×S, G, 15×S, Query ]` |
| 6        | 62.5% | `[ Instruction, 18×S, G, 11×S, Query ]` |
| 7        | 75%   | `[ Instruction, 22×S, G, 7×S, Query ]`  |
| 8        | 87.5% | `[ Instruction, 25×S, G, 4×S, Query ]`  |
| 9        | 100%  | `[ Instruction, 29×S, G, Query ]`        |

---

### Experiment 3A: Impact of Noise (Document-Grounded)

**Objective:** To test the hypothesis that adding completely unrelated, out-of-distribution documents ("noise") improves LLM reasoning by forcing better attention distribution.

**Methodology:** We combine the Gold document ($G$) with pure noise documents ($R$), randomly sampled from completely different query entries within the dataset to ensure they contain no relevant information. $G$ is held in a fixed position immediately preceding the user query.

**Test Variations:**

| Variation          | Layout                               |
| ------------------ | ------------------------------------ |
| 0 Noise (Baseline) | `[ Instruction, G, Query ]`        |
| 4 Noise            | `[ Instruction, 4×R, G, Query ]`  |
| 9 Noise            | `[ Instruction, 9×R, G, Query ]`  |
| 14 Noise           | `[ Instruction, 14×R, G, Query ]` |
| 19 Noise           | `[ Instruction, 19×R, G, Query ]` |
| 24 Noise           | `[ Instruction, 24×R, G, Query ]` |
| 29 Noise           | `[ Instruction, 29×R, G, Query ]` |

---

### Experiment 4A: Impact of Gold Positioning Across Noise (Document-Grounded)

**Objective:** To evaluate whether the "lost in the middle" phenomenon persists or changes in character when the surrounding documents are pure noise ($R$) rather than topically related distractors ($S$). This allows a direct comparison with Experiment 2A to isolate whether position sensitivity is driven by semantic interference or simply by context length.

**Methodology:** We mirror the structure of Experiment 2A exactly, substituting all distractor documents ($S$) with noise documents ($R$). The Gold document ($G$) is shifted across the same depth intervals for each of the three context sizes.

#### Case 1: 9 Noise (10 Total Documents)

| Position | Depth | Layout                                    |
| -------- | ----- | ----------------------------------------- |
| 1        | 0%    | `[ Instruction, G, 9×R, Query ]`       |
| 3        | 50%   | `[ Instruction, 4×R, G, 5×R, Query ]` |
| 5        | 100%  | `[ Instruction, 9×R, G, Query ]`       |

#### Case 2: 19 Noise (20 Total Documents)

| Position | Depth | Layout                                     |
| -------- | ----- | ------------------------------------------ |
| 1        | 0%    | `[ Instruction, G, 19×R, Query ]`       |
| 2        | 25%   | `[ Instruction, 4×R, G, 15×R, Query ]` |
| 3        | 50%   | `[ Instruction, 9×R, G, 10×R, Query ]` |
| 4        | 75%   | `[ Instruction, 14×R, G, 5×R, Query ]` |
| 5        | 100%  | `[ Instruction, 19×R, G, Query ]`       |

#### Case 3: 29 Noise (30 Total Documents)

| Position | Depth | Layout                                      |
| -------- | ----- | ------------------------------------------- |
| 1        | 0%    | `[ Instruction, G, 29×R, Query ]`        |
| 2        | 12.5% | `[ Instruction, 3×R, G, 26×R, Query ]`  |
| 3        | 25%   | `[ Instruction, 7×R, G, 22×R, Query ]`  |
| 4        | 37.5% | `[ Instruction, 10×R, G, 19×R, Query ]` |
| 5        | 50%   | `[ Instruction, 14×R, G, 15×R, Query ]` |
| 6        | 62.5% | `[ Instruction, 18×R, G, 11×R, Query ]` |
| 7        | 75%   | `[ Instruction, 22×R, G, 7×R, Query ]`  |
| 8        | 87.5% | `[ Instruction, 25×R, G, 4×R, Query ]`  |
| 9        | 100%  | `[ Instruction, 29×R, G, Query ]`        |

---

### Experiment 5A: Noise–Distractor Composition Tradeoff (Document-Grounded)

**Objective:** To examine the tradeoff between noise ($R$) and distractor ($S$) documents at fixed total context sizes. While Experiments 1A and 3A vary the *count* of a single filler type, this experiment fixes the total number of filler documents and varies their *composition* — testing whether a mixed context is more or less disruptive than a homogeneous one.

**Methodology:** The Gold document ($G$) is held in a fixed position immediately preceding the user query. The remaining filler slots are filled with varying ratios of $R$ and $S$ documents. All $R$ and $S$ documents are arranged in blocks (all $R$ before all $S$) to minimize interaction with position effects. Three total context sizes are tested.

#### Case 1: 10 Total Documents (1 Gold + 9 Filler)

| Variation          | Noise ($R$) | Distractor ($S$) | Layout                                    |
| ------------------ | ------------- | ------------------ | ----------------------------------------- |
| All Distractor     | 0             | 9                  | `[ Instruction, 9×S, G, Query ]`       |
| 2R + 7S            | 2             | 7                  | `[ Instruction, 2×R, 7×S, G, Query ]` |
| 5R + 4S (Balanced) | 5             | 4                  | `[ Instruction, 5×R, 4×S, G, Query ]` |
| 7R + 2S            | 7             | 2                  | `[ Instruction, 7×R, 2×S, G, Query ]` |
| All Noise          | 9             | 0                  | `[ Instruction, 9×R, G, Query ]`       |

#### Case 2: 20 Total Documents (1 Gold + 19 Filler)

| Variation           | Noise ($R$) | Distractor ($S$) | Layout                                     |
| ------------------- | ------------- | ------------------ | ------------------------------------------ |
| All Distractor      | 0             | 19                 | `[ Instruction, 19×S, G, Query ]`       |
| 2R + 17S            | 2             | 17                 | `[ Instruction, 2×R, 17×S, G, Query ]` |
| 4R + 15S            | 4             | 15                 | `[ Instruction, 4×R, 15×S, G, Query ]` |
| 7R + 12S            | 7             | 12                 | `[ Instruction, 7×R, 12×S, G, Query ]` |
| 9R + 10S (Balanced) | 9             | 10                 | `[ Instruction, 9×R, 10×S, G, Query ]` |
| 11R + 8S            | 11            | 8                  | `[ Instruction, 11×R, 8×S, G, Query ]` |
| 14R + 5S            | 14            | 5                  | `[ Instruction, 14×R, 5×S, G, Query ]` |
| 17R + 2S            | 17            | 2                  | `[ Instruction, 17×R, 2×S, G, Query ]` |
| All Noise           | 19            | 0                  | `[ Instruction, 19×R, G, Query ]`       |

#### Case 3: 30 Total Documents (1 Gold + 29 Filler)

| Variation            | Noise ($R$) | Distractor ($S$) | Layout                                      |
| -------------------- | ------------- | ------------------ | ------------------------------------------- |
| All Distractor       | 0             | 29                 | `[ Instruction, 29×S, G, Query ]`        |
| 2R + 27S             | 2             | 27                 | `[ Instruction, 2×R, 27×S, G, Query ]`  |
| 4R + 25S             | 4             | 25                 | `[ Instruction, 4×R, 25×S, G, Query ]`  |
| 7R + 22S             | 7             | 22                 | `[ Instruction, 7×R, 22×S, G, Query ]`  |
| 9R + 20S             | 9             | 20                 | `[ Instruction, 9×R, 20×S, G, Query ]`  |
| 12R + 17S            | 12            | 17                 | `[ Instruction, 12×R, 17×S, G, Query ]` |
| 14R + 15S (Balanced) | 14            | 15                 | `[ Instruction, 14×R, 15×S, G, Query ]` |
| 17R + 12S            | 17            | 12                 | `[ Instruction, 17×R, 12×S, G, Query ]` |
| 19R + 10S            | 19            | 10                 | `[ Instruction, 19×R, 10×S, G, Query ]` |
| 22R + 7S             | 22            | 7                  | `[ Instruction, 22×R, 7×S, G, Query ]`  |
| 24R + 5S             | 24            | 5                  | `[ Instruction, 24×R, 5×S, G, Query ]`  |
| 27R + 2S             | 27            | 2                  | `[ Instruction, 27×R, 2×S, G, Query ]`  |
| All Noise            | 29            | 0                  | `[ Instruction, 29×R, G, Query ]`        |

---

## Category B: Answer-Grounded Experiments

### Experiment 1B: Impact of Distracting Documents (Answer-Grounded)

**Objective:** To determine if the presence of a clear, undeniable Answer ($A$) makes the LLM robust against topically related distracting documents ($S$). Distracting documents ($S$) are sourced from the `distractor` array.

**Methodology:** We keep the Answer ($A$) in a fixed position immediately preceding the user query, and incrementally introduce distracting documents ($S$) before it.

**Test Variations:**

| Variation                | Layout                               |
| ------------------------ | ------------------------------------ |
| 0 Distractors (Baseline) | `[ Instruction, A, Query ]`        |
| 4 Distractors            | `[ Instruction, 4×S, A, Query ]`  |
| 9 Distractors            | `[ Instruction, 9×S, A, Query ]`  |
| 14 Distractors           | `[ Instruction, 14×S, A, Query ]` |
| 19 Distractors           | `[ Instruction, 19×S, A, Query ]` |
| 24 Distractors           | `[ Instruction, 24×S, A, Query ]` |
| 29 Distractors           | `[ Instruction, 29×S, A, Query ]` |

---

### Experiment 2B: Impact of Answer Positioning (Answer-Grounded)

**Objective:** To evaluate if the "lost in the middle" effect still occurs across different context lengths when the target information is a direct Answer ($A$) rather than a full document.

**Methodology:** We test three different context sizes (9, 19, and 29 distracting documents). For *each* context size, we shift the position of the Answer ($A$) across the same depth intervals used in Experiment 2A, allowing direct cross-category comparison.

#### Case 1: 9 Distractors (10 Total Documents)

| Position | Depth | Layout                                    |
| -------- | ----- | ----------------------------------------- |
| 1        | 0%    | `[ Instruction, A, 9×S, Query ]`       |
| 3        | 50%   | `[ Instruction, 4×S, A, 5×S, Query ]` |
| 5        | 100%  | `[ Instruction, 9×S, A, Query ]`       |

#### Case 2: 19 Distractors (20 Total Documents)

| Position | Depth | Layout                                     |
| -------- | ----- | ------------------------------------------ |
| 1        | 0%    | `[ Instruction, A, 19×S, Query ]`       |
| 2        | 25%   | `[ Instruction, 4×S, A, 15×S, Query ]` |
| 3        | 50%   | `[ Instruction, 9×S, A, 10×S, Query ]` |
| 4        | 75%   | `[ Instruction, 14×S, A, 5×S, Query ]` |
| 5        | 100%  | `[ Instruction, 19×S, A, Query ]`       |

#### Case 3: 29 Distractors (30 Total Documents)

| Position | Depth | Layout                                      |
| -------- | ----- | ------------------------------------------- |
| 1        | 0%    | `[ Instruction, A, 29×S, Query ]`        |
| 2        | 12.5% | `[ Instruction, 3×S, A, 26×S, Query ]`  |
| 3        | 25%   | `[ Instruction, 7×S, A, 22×S, Query ]`  |
| 4        | 37.5% | `[ Instruction, 10×S, A, 19×S, Query ]` |
| 5        | 50%   | `[ Instruction, 14×S, A, 15×S, Query ]` |
| 6        | 62.5% | `[ Instruction, 18×S, A, 11×S, Query ]` |
| 7        | 75%   | `[ Instruction, 22×S, A, 7×S, Query ]`  |
| 8        | 87.5% | `[ Instruction, 25×S, A, 4×S, Query ]`  |
| 9        | 100%  | `[ Instruction, 29×S, A, Query ]`        |

---

### Experiment 3B: Impact of Noise (Answer-Grounded)

**Objective:** To test if the addition of random, unrelated noise documents ($R$) provides any benefit to attention distribution when the ground truth is explicitly provided as an Answer ($A$).

**Methodology:** We combine the Answer ($A$) with pure noise documents ($R$), randomly sampled from completely different query entries within the dataset to ensure they contain no relevant information. $A$ is held in a fixed position immediately preceding the user query.

**Test Variations:**

| Variation          | Layout                               |
| ------------------ | ------------------------------------ |
| 0 Noise (Baseline) | `[ Instruction, A, Query ]`        |
| 4 Noise            | `[ Instruction, 4×R, A, Query ]`  |
| 9 Noise            | `[ Instruction, 9×R, A, Query ]`  |
| 14 Noise           | `[ Instruction, 14×R, A, Query ]` |
| 19 Noise           | `[ Instruction, 19×R, A, Query ]` |
| 24 Noise           | `[ Instruction, 24×R, A, Query ]` |
| 29 Noise           | `[ Instruction, 29×R, A, Query ]` |

---

### Experiment 4B: Impact of Answer Positioning Across Noise (Answer-Grounded)

**Objective:** To evaluate whether the "lost in the middle" effect persists when the Answer ($A$) is a compact, self-contained unit surrounded by noise ($R$) rather than distractors ($S$). Comparing results against Experiment 2B reveals whether position sensitivity in the answer-grounded setting is driven by semantic competition or by raw context depth.

**Methodology:** We mirror the structure of Experiment 2B exactly, substituting all distractor documents ($S$) with noise documents ($R$). The Answer ($A$) is shifted across the same depth intervals for each of the three context sizes.

#### Case 1: 9 Noise (10 Total Documents)

| Position | Depth | Layout                                    |
| -------- | ----- | ----------------------------------------- |
| 1        | 0%    | `[ Instruction, A, 9×R, Query ]`       |
| 3        | 50%   | `[ Instruction, 4×R, A, 5×R, Query ]` |
| 5        | 100%  | `[ Instruction, 9×R, A, Query ]`       |

#### Case 2: 19 Noise (20 Total Documents)

| Position | Depth | Layout                                     |
| -------- | ----- | ------------------------------------------ |
| 1        | 0%    | `[ Instruction, A, 19×R, Query ]`       |
| 2        | 25%   | `[ Instruction, 4×R, A, 15×R, Query ]` |
| 3        | 50%   | `[ Instruction, 9×R, A, 10×R, Query ]` |
| 4        | 75%   | `[ Instruction, 14×R, A, 5×R, Query ]` |
| 5        | 100%  | `[ Instruction, 19×R, A, Query ]`       |

#### Case 3: 29 Noise (30 Total Documents)

| Position | Depth | Layout                                      |
| -------- | ----- | ------------------------------------------- |
| 1        | 0%    | `[ Instruction, A, 29×R, Query ]`        |
| 2        | 12.5% | `[ Instruction, 3×R, A, 26×R, Query ]`  |
| 3        | 25%   | `[ Instruction, 7×R, A, 22×R, Query ]`  |
| 4        | 37.5% | `[ Instruction, 10×R, A, 19×R, Query ]` |
| 5        | 50%   | `[ Instruction, 14×R, A, 15×R, Query ]` |
| 6        | 62.5% | `[ Instruction, 18×R, A, 11×R, Query ]` |
| 7        | 75%   | `[ Instruction, 22×R, A, 7×R, Query ]`  |
| 8        | 87.5% | `[ Instruction, 25×R, A, 4×R, Query ]`  |
| 9        | 100%  | `[ Instruction, 29×R, A, Query ]`        |

---

### Experiment 5B: Noise–Distractor Composition Tradeoff (Answer-Grounded)

**Objective:** To examine the tradeoff between noise ($R$) and distractor ($S$) documents at fixed total context sizes when the ground truth is explicitly provided as Answer ($A$). Results from this experiment can be directly compared against Experiment 5A to determine whether the composition effect is sensitive to the type of core information source.

**Methodology:** The Answer ($A$) is held in a fixed position immediately preceding the user query. The remaining filler slots are filled with varying ratios of $R$ and $S$ documents. All $R$ and $S$ documents are arranged in blocks (all $R$ before all $S$) to minimize interaction with position effects. Three total context sizes are tested.

#### Case 1: 10 Total Documents (1 Answer + 9 Filler)

| Variation          | Noise ($R$) | Distractor ($S$) | Layout                                    |
| ------------------ | ------------- | ------------------ | ----------------------------------------- |
| All Distractor     | 0             | 9                  | `[ Instruction, 9×S, A, Query ]`       |
| 2R + 7S            | 2             | 7                  | `[ Instruction, 2×R, 7×S, A, Query ]` |
| 5R + 4S (Balanced) | 5             | 4                  | `[ Instruction, 5×R, 4×S, A, Query ]` |
| 7R + 2S            | 7             | 2                  | `[ Instruction, 7×R, 2×S, A, Query ]` |
| All Noise          | 9             | 0                  | `[ Instruction, 9×R, A, Query ]`       |

#### Case 2: 20 Total Documents (1 Answer + 19 Filler)

| Variation           | Noise ($R$) | Distractor ($S$) | Layout                                     |
| ------------------- | ------------- | ------------------ | ------------------------------------------ |
| All Distractor      | 0             | 19                 | `[ Instruction, 19×S, A, Query ]`       |
| 2R + 17S            | 2             | 17                 | `[ Instruction, 2×R, 17×S, A, Query ]` |
| 4R + 15S            | 4             | 15                 | `[ Instruction, 4×R, 15×S, A, Query ]` |
| 7R + 12S            | 7             | 12                 | `[ Instruction, 7×R, 12×S, A, Query ]` |
| 9R + 10S (Balanced) | 9             | 10                 | `[ Instruction, 9×R, 10×S, A, Query ]` |
| 11R + 8S            | 11            | 8                  | `[ Instruction, 11×R, 8×S, A, Query ]` |
| 14R + 5S            | 14            | 5                  | `[ Instruction, 14×R, 5×S, A, Query ]` |
| 17R + 2S            | 17            | 2                  | `[ Instruction, 17×R, 2×S, A, Query ]` |
| All Noise           | 19            | 0                  | `[ Instruction, 19×R, A, Query ]`       |

#### Case 3: 30 Total Documents (1 Answer + 29 Filler)

| Variation            | Noise ($R$) | Distractor ($S$) | Layout                                      |
| -------------------- | ------------- | ------------------ | ------------------------------------------- |
| All Distractor       | 0             | 29                 | `[ Instruction, 29×S, A, Query ]`        |
| 2R + 27S             | 2             | 27                 | `[ Instruction, 2×R, 27×S, A, Query ]`  |
| 4R + 25S             | 4             | 25                 | `[ Instruction, 4×R, 25×S, A, Query ]`  |
| 7R + 22S             | 7             | 22                 | `[ Instruction, 7×R, 22×S, A, Query ]`  |
| 9R + 20S             | 9             | 20                 | `[ Instruction, 9×R, 20×S, A, Query ]`  |
| 12R + 17S            | 12            | 17                 | `[ Instruction, 12×R, 17×S, A, Query ]` |
| 14R + 15S (Balanced) | 14            | 15                 | `[ Instruction, 14×R, 15×S, A, Query ]` |
| 17R + 12S            | 17            | 12                 | `[ Instruction, 17×R, 12×S, A, Query ]` |
| 19R + 10S            | 19            | 10                 | `[ Instruction, 19×R, 10×S, A, Query ]` |
| 22R + 7S             | 22            | 7                  | `[ Instruction, 22×R, 7×S, A, Query ]`  |
| 24R + 5S             | 24            | 5                  | `[ Instruction, 24×R, 5×S, A, Query ]`  |
| 27R + 2S             | 27            | 2                  | `[ Instruction, 27×R, 2×S, A, Query ]`  |
| All Noise            | 29            | 0                  | `[ Instruction, 29×R, A, Query ]`        |

