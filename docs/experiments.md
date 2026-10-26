# Experiment Log

This log preserves the configurations, artifacts, and interpretations for the
project's recorded experiments. The opening template below defines the fields
used by the subsequent completed entries.

Each experiment will record:

- experiment ID;
- date;
- Git commit;
- architecture version;
- question;
- stances;
- model configuration;
- research configuration;
- termination policy;
- summary policy;
- output artifact path;
- metric results;
- token usage;
- estimated cost;
- observations;
- keep/change/remove decision.

## M0-UBI-001

- **Milestone:** 0
- **Architecture:** Fixed single-round MVP
- **Question:** Should governments provide a universal basic income to all adult citizens?
- **Model:** `qwen3:8b`
- **Temperature:** `0.0`
- **Reasoning output:** Disabled
- **Grounding:** Simulated model-internal knowledge; not externally verified
- **Transcript:** `experiments/baselines/m0_ubi_qwen3_8b.txt`
- **Observation:** Routing, structured research, rebuttal, and neutral synthesis worked correctly. Arguments were coherent but relatively shallow, which is acceptable for the MVP baseline.
- **Decision:** Preserve as the Milestone 0 baseline for comparison.

- **Metric version:** `1.0`
- **Scoring implementation commit:** `87574308352ff7690960e68f3672fcb7c02f0eab`
- **Scorecard:** `experiments/scores/m0_ubi_qwen3_8b.metrics-v1.0.json`
- **Scorecard SHA-256:** `d2e666079bba7d63d98be1092c19959e4221a826b00b114a0d4cf849abff374a`
- **Research utilization:** `0.6986111111111111`
  - Finding 1: `1.0`
  - Finding 2: `0.5625`
  - Finding 3: `0.5333333333333333`
- **Team novelty:** `0.9885057471264368`
  - Jaccard similarity: `0.011494252873563218`
- **Synthesis representation:** `0.6734537956462252`
  - Team A coverage: `0.6808510638297872`
  - Team B coverage: `0.717948717948718`
  - Balance: `0.9629023458810693`
- **Summary-policy compliance:** `1`
  - Matched patterns: none
- **Output word counts:**
  - Research: `67` (`18`, `23`, `26`)
  - Team A: `74`
  - Team B: `60`
  - Synthesis: `172`
  - Total: `373`
- **Measurement method:** Scored the previously saved transcript without regenerating the debate or invoking Ollama.

## M2.1-UBI-001

- **Milestone:** 2.1B2B
- **Run timestamp (UTC):** `2026-08-05T11:37:55.647723Z`
- **Implementation commit:** `4a3eb4a4e05a882dd77eb5511e7a245a04fa3e83`
- **Implementation tag:** `debate-m2-1b2b-phase-a`
- **Architecture:** `milestone-2.1b2b`
- **Run-record version:** `1.0`
- **Question:** Should governments provide universal basic income to all adult citizens?
- **Stances:**
  - Team A: Support the proposition stated in the question.
  - Team B: Oppose the proposition stated in the question.
- **Model configuration:**
  - Model: `qwen3:8b`
  - Temperature: `0.0`
  - Reasoning output: Disabled
  - Structured-output method: `json_schema`
- **Decomposition:** `complex`
- **Ordered subquestions:**
  1. What are the economic impacts of implementing a universal basic income program in different countries?
  2. How does universal basic income affect labor market participation and employment rates?
  3. What are the fiscal implications of funding a universal basic income for governments?
- **Research configuration:**
  - Provider: Tavily
  - Endpoint: `https://api.tavily.com/search`
  - MCP tool: `search_web`
  - Search depth: `basic`
  - Maximum results per query: `3`
  - Include answer: `false`
  - Include raw content: `false`
  - Include images: `false`
  - Execution: `3` searches and `9` normalized sources
- **Ordered source information:**
  1. Subquestion 1, rank 1 — “Short-term impacts of Universal Basic Income on population mental health inequalities in the UK: A microsimulation modelling study”; `https://pmc.ncbi.nlm.nih.gov/articles/PMC10947674`; provider score `0.7283819`; source type `web`.
  2. Subquestion 1, rank 2 — “Countries with Universal Basic Income 2026”; `https://worldpopulationreview.com/country-rankings/countries-with-universal-basic-income`; provider score `0.72559077`; source type `web`.
  3. Subquestion 1, rank 3 — “The macroeconomic effects of universal basic income programs”; `https://www.sciencedirect.com/science/article/abs/pii/S0304393224000680`; provider score `0.66569364`; source type `web`.
  4. Subquestion 2, rank 1 — JSON-escaped title `"The Impact of Universal Basic Income on Labor Market Participation\n\t\t\t\t\t\t\t| International Journal of Economic Policy"`; `https://carijournals.org/journals/IJECOP/article/view/1761`; provider score `0.9323611`; source type `web`.
  5. Subquestion 2, rank 2 — “Study: Recipients of universal basic income work fewer hours, are less productive”; `https://www.carolinajournal.com/study-recipients-of-universal-basic-income-work-fewer-hours-are-less-productive`; provider score `0.86331123`; source type `web`.
  6. Subquestion 2, rank 3 — “The Impact of Universal Basic Income Programs on Labor Force Participation Rates and Entrepreneurial Activity in Developed Economies | `”; `https://assajournal.com/index.php/36/article/view/1373`; provider score `0.8190992`; source type `web`.
  7. Subquestion 3, rank 1 — “How to Fund a Universal Basic Income”; `https://citizen-network.org/library/how-to-fund-a-universal-basic-income.html`; provider score `0.72768575`; source type `web`.
  8. Subquestion 3, rank 2 — “Universal basic income - Wikipedia”; `https://en.wikipedia.org/wiki/Universal_basic_income`; provider score `0.680136`; source type `web`.
  9. Subquestion 3, rank 3 — “Universal Basic Income Would Undermine the Success of Our Safety Net | George W. Bush Presidential Center”; `https://www.bushcenter.org/catalyst/are-we-ready/doar-universal-basic-income`; provider score `0.56258565`; source type `web`.
- **Timings:**
  - Decomposition: `5.571920917020179` seconds
  - Research 1: backend `1.9049741249764338` seconds; complete MCP research `1.921051291981712` seconds
  - Research 2: backend `1.685134792001918` seconds; complete MCP research `1.7427030840190127` seconds
  - Research 3: backend `3.11407612496987` seconds; complete MCP research `3.1283684170339257` seconds
  - Team A: `17.550220834033098` seconds
  - Team B: `16.075776832993142` seconds
  - Neutral synthesis: `22.13898008398246` seconds
  - Total: `64.47438204201171` seconds
- **Run artifact:** `experiments/runs/m2_1b2b_ubi_qwen3_8b.json`
- **Run artifact SHA-256:** `6d88274129f2ae6d1f708847d300a00f76f953bc1d42fbd493bef7f190c506c7`
- **Metric version:** `1.0`
- **Scorecard:** `experiments/scores/m2_1b2b_ubi_qwen3_8b.metrics-v1.0.json`
- **Scorecard SHA-256:** `08544bea4800a6bba4ce66a87fb6313adc4edb9dab26493107f6f835ba702981`
- **Metric comparison:**

  | Metric                    |                 M0 |      M2.1 grounded |               Delta |
  | ------------------------- | -----------------: | -----------------: | ------------------: |
  | Research utilization      | 0.6986111111111111 | 0.2290656413443515 | -0.4695454697667596 |
  | Team novelty              | 0.9885057471264368 |                1.0 | +0.0114942528735632 |
  | Synthesis representation  | 0.6734537956462252 | 0.5413676705034739 | -0.1320861251427513 |
  | Summary-policy compliance |                  1 |                  1 |                   0 |

- **M2.1 metric components:**
  - Research-utilization findings: `0.19444444444444445`, `0.23622047244094488`, `0.12037037037037036`, `0.23404255319148937`, `0.2465753424657534`, `0.1509433962264151`, `0.21518987341772153`, `0.25396825396825395`, `0.4098360655737705`
  - Team-novelty Jaccard similarity: `0.0`
  - Team-novelty shingles: Team A `90`; Team B `43`; shared `0`; union `133`
  - Synthesis Team A coverage: `0.5421686746987951`
  - Synthesis Team B coverage: `0.5581395348837209`
  - Synthesis mean coverage: `0.550154104791258`
  - Synthesis balance: `0.9840291398150742`
  - Summary-policy matched patterns: none
- **Operational word-count comparison:**
  - Research: `67` → `1,582` (M2.1 findings: `200`, `227`, `193`, `178`, `141`, `187`, `187`, `114`, `155`)
  - Team A: `74` → `126`
  - Team B: `60` → `64`
  - Synthesis: `172` → `245`
  - Total output: `373` → `2,017`
- **Token usage:** Not measured; deferred to Milestone 3.
- **Monetary cost:** Not measured; deferred to Milestone 3.
- **Observations:**
  1. The complete real-search path worked end to end.
  2. Search was not the dominant latency source; decomposition and debate-model generation consumed approximately 95% of total runtime.
  3. Research utilization fell substantially because nine provider snippets contained 1,582 words, approximately 23.6 times the baseline research length, while the generated arguments and synthesis remained comparatively concise.
  4. The lower utilization score must not be interpreted as a direct factual quality score because metric v1.0 measures lexical overlap.
  5. Synthesis representation balance remained high, but mean argument coverage decreased.
  6. Team novelty reached its maximum because the arguments shared no measured three-token shingles.
  7. The retrieved source set was heterogeneous, including academic, policy/advocacy, aggregator, journal, and Wikipedia sources.
  8. Source-quality and citation-quality scoring remain deferred.
  9. Token use and monetary cost were not measured and remain deferred to Milestone 3.
- **Decision:**
  - **KEEP:** The real Tavily-over-MCP grounding architecture.
  - **CHANGE:** In a later milestone, test concise claim extraction or snippet compression before debate generation, without modifying metric v1.0 here.
  - **REMOVE:** Nothing at this checkpoint.

## M2.2-UBI-001

- **Milestone:** 2.2B2
- **Comparison:** Controlled replay of the frozen one-shot team graph against the hierarchical team-subgraph graph
- **Replay run timestamp (UTC), both variants:** `2026-08-05T16:48:22.554708Z`
- **Source run timestamp (UTC):** `2026-08-05T11:37:55.647723Z`
- **Implementation commit:** `280fb0a14992bdcc5ce923ae17b3f1de9230e913`
- **Implementation tag:** `debate-m2-2b2-replay-cli`
- **Run-record version:** `1.1`
- **Architectures:**
  - Control: `milestone-2.2b-control-replay`
  - Hierarchical: `milestone-2.2b-hierarchical-replay`
- **Source architecture:** `milestone-2.1b2b`
- **Source implementation commit:** `4a3eb4a4e05a882dd77eb5511e7a245a04fa3e83`
- **Question:** Should governments provide universal basic income to all adult citizens?
- **Stances:**
  - Team A: Support the proposition stated in the question.
  - Team B: Oppose the proposition stated in the question.
- **Model configuration:**
  - Model: `qwen3:8b`
  - Temperature: `0.0`
  - Reasoning output: Disabled
  - Structured-output method: `json_schema`
- **Metric version:** `1.0`
- **Decomposition:** `complex`
- **Ordered subquestions:**
  1. What are the economic impacts of implementing a universal basic income program in different countries?
  2. How does universal basic income affect labor market participation and employment rates?
  3. What are the fiscal implications of funding a universal basic income for governments?
- **Replay source:** `experiments/runs/m2_1b2b_ubi_qwen3_8b.json`
- **Replay source SHA-256:** `6d88274129f2ae6d1f708847d300a00f76f953bc1d42fbd493bef7f190c506c7`
- **Replay controls:** The question, stances, complexity classification, three ordered subquestions, nine raw findings, nine consolidated findings, formatted evidence, model settings, synthesis implementation, and metric version were identical between variants. Replay decomposition and replay research made zero model calls. No Tavily, MCP, HTTP, or live decomposition was used.
- **Artifacts:**
  - Control run record: `experiments/runs/m2_2b_control_ubi_qwen3_8b.json`
  - Control run SHA-256: `67fd98e98ce5156e488dfea47ae6fc453a012cd74427ecba325f23e06535e0df`
  - Control scorecard: `experiments/scores/m2_2b_control_ubi_qwen3_8b.metrics-v1.0.json`
  - Control scorecard SHA-256: `08544bea4800a6bba4ce66a87fb6313adc4edb9dab26493107f6f835ba702981`
  - Hierarchical run record: `experiments/runs/m2_2b_hierarchical_ubi_qwen3_8b.json`
  - Hierarchical run SHA-256: `7e05e8cc6f02af2a35616a5a0c2332361658f9bcfeec3ae2084586b788ba4e1c`
  - Hierarchical scorecard: `experiments/scores/m2_2b_hierarchical_ubi_qwen3_8b.metrics-v1.0.json`
  - Hierarchical scorecard SHA-256: `6c23f5d214244586e3e7d6fdba891d7fbe7203d49995c4053f6c57108b33b2ae`
- **Replay reproduction:** The control scorecard SHA-256 is identical to the prior M2.1 grounded scorecard SHA-256, confirming that deterministic control replay reproduced the prior metric output exactly.
- **Metric comparison:**

  | Metric                    |            Control |       Hierarchical |                 Delta |
  | ------------------------- | -----------------: | -----------------: | --------------------: |
  | Research utilization      | 0.2290656413443515 | 0.2204469505932126 | -0.008618690751138913 |
  | Team novelty              |                1.0 | 0.9738562091503268 |  -0.02614379084967322 |
  | Synthesis representation  | 0.5413676705034739 | 0.5195121671258034 | -0.021855503377670504 |
  | Summary-policy compliance |                  1 |                  1 |                     0 |

- **Control metric components:**
  - Research-utilization findings: `0.19444444444444445`, `0.23622047244094488`, `0.12037037037037036`, `0.23404255319148937`, `0.2465753424657534`, `0.1509433962264151`, `0.21518987341772153`, `0.25396825396825395`, `0.4098360655737705`
  - Team-novelty Jaccard similarity: `0.0`
  - Team-novelty shingles: Team A `90`; Team B `43`; shared `0`; union `133`
  - Synthesis Team A coverage: `0.5421686746987951`
  - Synthesis Team B coverage: `0.5581395348837209`
  - Synthesis mean coverage: `0.550154104791258`
  - Synthesis balance: `0.9840291398150742`
  - Summary-policy matched patterns: none
- **Hierarchical metric components:**
  - Research-utilization findings: `0.125`, `0.2440944881889764`, `0.17592592592592593`, `0.26595744680851063`, `0.2602739726027397`, `0.18867924528301888`, `0.189873417721519`, `0.20634920634920634`, `0.32786885245901637`
  - Team-novelty Jaccard similarity: `0.026143790849673203`
  - Team-novelty shingles: Team A `89`; Team B `68`; shared `4`; union `153`
  - Synthesis Team A coverage: `0.65`
  - Synthesis Team B coverage: `0.5303030303030303`
  - Synthesis mean coverage: `0.5901515151515151`
  - Synthesis balance: `0.8803030303030303`
  - Summary-policy matched patterns: none
- **Absolute output word counts:**
  - Control research: `1,582` (`200`, `227`, `193`, `178`, `141`, `187`, `187`, `114`, `155`)
  - Hierarchical research: `1,582` (`200`, `227`, `193`, `178`, `141`, `187`, `187`, `114`, `155`)
  - Control Team A argument: `126`; hierarchical Team A argument: `119`
  - Control Team B argument: `64`; hierarchical Team B argument: `98`
  - Control synthesis: `245`; hierarchical synthesis: `194`
  - Control total output: `2,017`; hierarchical total output: `1,993`
- **Word-count deltas, hierarchical minus control:**
  - Research total: `0`
  - Team A argument: `-7`
  - Team B argument: `+34`
  - Synthesis: `-51`
  - Total output: `-24`
- **Control timings:**
  - Team A argument: `19.739783375000115` seconds
  - Team B argument: `15.772302541998215` seconds
  - Neutral synthesis: `21.652570957958233` seconds
  - Total replay: `57.1693194169784` seconds
- **Hierarchical timings:**
  - Team A evidence analysis: `25.993490875000134` seconds
  - Team A argument draft: `18.64040199998999` seconds
  - Team A review and revision: `27.023711082991213` seconds
  - Team A total: `71.65844537504017` seconds
  - Team B evidence analysis: `28.31464929203503` seconds
  - Team B argument draft: `18.620678375009447` seconds
  - Team B review and revision: `26.720203417004086` seconds
  - Team B total: `73.65635479101911` seconds
  - Neutral synthesis: `7.379324458015617` seconds
  - Total replay: `152.70408820797456` seconds
- **Latency comparison:** Hierarchical execution added `95.53476879099616` seconds and took approximately `2.671` times as long as control.
- **Control model calls:** `3` total
  - Team A argument: `1`
  - Team B argument: `1`
  - Neutral synthesis: `1`
  - Replay decomposition: `0`
  - Replay research: `0`
- **Hierarchical model calls:** `7` total
  - Team A analysis: `1`
  - Team A draft: `1`
  - Team A review: `1`
  - Team B analysis: `1`
  - Team B draft: `1`
  - Team B review: `1`
  - Neutral synthesis: `1`
  - Replay decomposition: `0`
  - Replay research: `0`
- **Model-call delta:** `+4`
- **Bounded team-work-product inspection:**
  - Team A's analyst selected supporting evidence `(1, 2)`, `(1, 3)`, and `(3, 1)`, and limitations or conflicts `(1, 1)`, `(2, 2)`, and `(3, 3)`. It identified three claims to make, one unsupported claim to avoid, and no opponent point, as required for the opening team.
  - Team A's critic identified weak integration of conflicting labor and fiscal evidence, insufficient qualification of uncertain evidence, limited counterargument handling, and opportunities to improve focus and concision. The final argument was textually identical to the draft, so none of those requested revisions produced a textual change.
  - Team A's final argument used the analyst-selected pilot-outcome, macroeconomic, fiscal, and labor-market evidence. Within the supplied snippets, however, its linkage of successful outcomes specifically to Finland and Canada was more specific than the evidence, which mentions those countries and positive pilot outcomes without directly tying the outcomes together. Its fiscal-sustainability phrasing was also an inference from an advocacy source's proposed financing mechanism rather than an explicit finding in the snippet.
  - Team B's analyst selected supporting evidence `(2, 2)`, `(3, 3)`, and `(3, 2)`, and limitations or conflicts `(2, 3)`, `(1, 1)`, and `(1, 3)`. It identified three claims to make, one unsupported claim to avoid, and one Team A point concerning welfare gains, inequality, and macroeconomic effects.
  - Team B's critic found the stance consistent and the draft clear, while requesting fuller integration of fiscal implications, limitations, and a more direct response to Team A. The final differed from the draft: it expanded the opening acronym, streamlined the quantified participation wording, and replaced a self-referential closing about what opponents must address with a neutral call for careful consideration.
  - Team B received Team A's exact final argument and used the analyst-selected labor and fiscal evidence. It addressed Team A only indirectly by challenging universal benefit claims; it did not directly rebut the analyst-identified welfare-gain and inequality point. Its final quantified labor and fiscal claims appear in the supplied snippets, and no additional clearly unsupported factual claim was identified in this bounded inspection.
  - This inspection assesses support only against the replayed evidence snippets. It is not external factual verification and does not establish that any claim is globally true or false.
- **Token usage:** Not measured; deferred to Milestone 3.
- **Monetary cost:** Not measured; deferred to Milestone 3.
- **Observations:**
  1. The controlled replay successfully isolated the team architecture by holding question, stances, decomposition, evidence, model settings, synthesis implementation, and metric version constant.
  2. The control reproduced the prior M2.1 metric scorecard exactly.
  3. The hierarchical graph executed correctly with scoped team state, explicit analysis, drafting, criticism, and revision.
  4. Hierarchical execution increased model calls from `3` to `7` and latency by approximately `2.67` times.
  5. None of the four metric-v1.0 scores improved in this single run.
  6. Hierarchical output was `24` words shorter overall, despite Team B's final argument becoming `34` words longer.
  7. Metric v1.0 does not directly measure factual correctness, usefulness of critique, citation quality, or quality of reasoning.
  8. The work-product inspection must therefore be reported separately from the automated metrics.
  9. One run at temperature zero is still insufficient to establish a general architecture-quality conclusion.
  10. Token counts and monetary cost were not measured and remain deferred to Milestone 3.
- **Decision:**
  - **KEEP:** The hierarchical team-subgraph implementation as an explicit, inspectable architecture demonstrating agent specialization, scoped state, and subgraph-as-agent composition.
  - **DO NOT PROMOTE:** It to the default debate path based on this experiment. The one-shot control was faster and scored slightly better on every nonbinary metric-v1.0 measure.
  - **CHANGE:** In a later cost/quality milestone, test selective specialization, such as enabling analysis or criticism only when task complexity or draft quality justifies the additional model calls.
  - **REMOVE:** Nothing at this checkpoint.

## M2.3-UBI-001

- **Milestone:** 2.3
- **Run timestamp (UTC):** `2026-08-05T18:11:59.308887Z`
- **Implementation commit:** `472495ccad469c96054c4ba1b0ba512c5ac2139c`
- **Implementation tag:** `debate-m2-3-final-product`
- **Run-record version:** `2.0`
- **Architecture:** `final-multi-round-grounded`
- **Mode:** `multi-round`
- **Question:** Should governments provide universal basic income to all adult citizens?
- **Stances:**
  - Team A: Support the proposition stated in the question.
  - Team B: Oppose the proposition stated in the question.
- **Model configuration:**
  - Model: `qwen3:8b`
  - Temperature: `0.0`
  - Reasoning output: Disabled
  - Structured-output method: `json_schema`
- **Decomposition:** `complex`
- **Ordered subquestions:**
  1. What are the economic impacts of implementing a universal basic income program in different countries?
  2. How does universal basic income affect labor market participation and employment rates?
  3. What are the fiscal implications of funding a universal basic income for governments?
- **Research configuration:**
  - Provider: Tavily
  - Endpoint: `https://api.tavily.com/search`
  - MCP tool: `search_web`
  - Search depth: `basic`
  - Maximum results per query: `3`
  - Include answer: `false`
  - Include raw content: `false`
  - Include images: `false`
  - Execution: one decomposition and research phase, `3` provider calls, and `9` normalized sources; no research call occurred inside either debate round.
- **Ordered source information:**
  1. Subquestion 1, rank 1 — “Countries with Universal Basic Income 2026”; `https://worldpopulationreview.com/country-rankings/countries-with-universal-basic-income`; provider score `0.75616`; source type `web`.
  2. Subquestion 1, rank 2 — “Short-term impacts of Universal Basic Income on population mental health inequalities in the UK: A microsimulation modelling study”; `https://pmc.ncbi.nlm.nih.gov/articles/PMC10947674`; provider score `0.7099254`; source type `web`.
  3. Subquestion 1, rank 3 — “Universal basic income”; `https://en.wikipedia.org/wiki/Universal_basic_income`; provider score `0.6445166`; source type `web`.
  4. Subquestion 2, rank 1 — JSON-escaped title `"The Impact of Universal Basic Income on Labor Market Participation\n\t\t\t\t\t\t\t| International Journal of Economic Policy"`; `https://carijournals.org/journals/IJECOP/article/view/1761`; provider score `0.9323611`; source type `web`.
  5. Subquestion 2, rank 2 — “Study: Recipients of universal basic income work fewer hours, are less productive”; `https://www.carolinajournal.com/study-recipients-of-universal-basic-income-work-fewer-hours-are-less-productive`; provider score `0.86331123`; source type `web`.
  6. Subquestion 2, rank 3 — “Cogent | Blog | Universal Basic Income And The Future Of Work”; `https://cogentinfo.com/resources/universal-basic-income-and-the-future-of-work-3460c`; provider score `0.8047902`; source type `web`.
  7. Subquestion 3, rank 1 — “Tax policy reform and universal basic income effectiveness in a currency union: Implications for long-term growth, inequality, and welfare”; `https://www.sciencedirect.com/science/article/pii/S2667319323000186`; provider score `0.8170061`; source type `web`.
  8. Subquestion 3, rank 2 — “How to Fund a Universal Basic Income”; `https://citizen-network.org/library/how-to-fund-a-universal-basic-income.html`; provider score `0.72768575`; source type `web`.
  9. Subquestion 3, rank 3 — “Universal basic income - Wikipedia”; `https://en.wikipedia.org/wiki/Universal_basic_income`; provider score `0.680136`; source type `web`.
- **Debate policy and result:**
  - Configured maximum rounds: `2`
  - Novelty threshold: `0.15`
  - Completed rounds: `2`
  - Round 1 novelty: Team A `n/a`; Team B `n/a` because no prior same-team arguments existed.
  - Round 1 decision: `continue`
  - Round 2 Team A novelty: `0.9789473684210527`
  - Round 2 Team B novelty: `0.8301886792452831`
  - Final termination reason: `max_rounds`
- **Timings:**
  - Decomposition: `3.2749559590010904` seconds
  - Research 1: backend `2.4302038329769857` seconds; complete MCP research `2.447246333002113` seconds
  - Research 2: backend `1.5513250409858301` seconds; complete MCP research `1.5829742919886485` seconds
  - Research 3: backend `1.6731452089734375` seconds; complete MCP research `1.6885679580154829` seconds
  - Round 1 Team A: `18.29813929204829` seconds
  - Round 1 Team B: `16.570643624989316` seconds
  - Round 1 termination assessment: `0.00006212497828528285` seconds
  - Round 2 Team A: `15.93984674999956` seconds
  - Round 2 Team B: `3.552823292033281` seconds
  - Round 2 termination assessment: `0.0002982920268550515` seconds
  - Neutral synthesis: `23.81592299998738` seconds
  - Total: `83.91222683398519` seconds
- **Model-call counts:** `6` total
  - Question decomposition: `1`
  - Team A arguments: `2`
  - Team B arguments: `2`
  - Neutral synthesis: `1`
  - Research-provider calls: `3`, recorded separately and excluded from the model-call total
- **Run artifact:** `experiments/runs/m2_3_final_multi_round_ubi_qwen3_8b.json`
- **Run artifact SHA-256:** `8db0b0780320562b5ef57474f068178a5b25b812384917d969db36a4caeddb35`
- **Metric version:** `1.0`
- **Scorecard:** `experiments/scores/m2_3_final_multi_round_ubi_qwen3_8b.metrics-v1.0.json`
- **Scorecard SHA-256:** `22aefc070fe3c3be8cfeb141127dc7349f2cffcdb00a03799ee02be0dc9979c9`
- **Final metric values:**
  - Research utilization: `0.30785956790033353`
  - Team novelty: `0.323943661971831`
  - Synthesis representation: `0.7578429775038579`
  - Summary-policy compliance: `1`
- **Metric components:**
  - Research-utilization findings: `0.31496062992125984`, `0.16666666666666666`, `0.20161290322580644`, `0.35106382978723405`, `0.3269230769230769`, `0.4444444444444444`, `0.2736842105263158`, `0.21518987341772153`, `0.47619047619047616`
  - Team-novelty Jaccard similarity: `0.676056338028169`
  - Team-novelty shingles: Team A `62`; Team B `57`; shared `48`; union `71`
  - Synthesis Team A coverage: `0.7894736842105263`
  - Synthesis Team B coverage: `0.7647058823529411`
  - Synthesis mean coverage: `0.7770897832817337`
  - Synthesis balance: `0.9752321981424148`
  - Summary-policy matched patterns: none
- **Comparison with the grounded single-round M2.1 result:**

  | Metric                    |      M2.1 grounded | M2.3 multi-round |                  Delta |
  | ------------------------- | -----------------: | ----------------: | ---------------------: |
  | Research utilization      | 0.2290656413443515 | 0.30785956790033353 | +0.07879392655598203 |
  | Team novelty              |                1.0 |   0.323943661971831 |   -0.676056338028169 |
  | Synthesis representation  | 0.5413676705034739 |  0.7578429775038579 |  +0.2164753070003840 |
  | Summary-policy compliance |                  1 |                   1 |                      0 |

- **Absolute output word counts:**
  - M2.1 research: `1,582` (`200`, `227`, `193`, `178`, `141`, `187`, `187`, `114`, `155`)
  - M2.3 research: `1,504` (`227`, `200`, `194`, `178`, `188`, `72`, `225`, `187`, `33`)
  - M2.1 final Team A argument: `126`; M2.3 final Team A argument: `86`
  - M2.1 final Team B argument: `64`; M2.3 final Team B argument: `82`
  - M2.1 synthesis: `245`; M2.3 synthesis: `323`
  - M2.1 total output: `2,017`; M2.3 total output: `1,995`
- **Word-count deltas, M2.3 minus M2.1:**
  - Research: `-78`
  - Final Team A argument: `-40`
  - Final Team B argument: `+18`
  - Synthesis: `+78`
  - Total output: `-22`
- **Bounded round and evidence inspection:**
  - Team A began by supporting UBI with pilot outcomes, welfare-state context, education choice, conditional labor effects, and possible administrative savings. In round two it did not defend that stance: it adopted Team B's labor-participation objection, focused on the `-3.2%` larger-program employment result, and its text was labeled “Team B's response.” This is a material stance and evidence-use change visible in the artifact.
  - Team B received Team A's exact round-two argument, but because that argument already advanced Team B's position, Team B mostly restated and shortened the same employment, tax-revenue, productivity, and economic-risk claims rather than rebutting an opposing point.
  - Both round-two novelty values were well above `0.15`, demonstrating substantial new measured shingle content. Qualitatively, however, Team A repeated Team B's core round-one objection and Team B then repeated that same core claim; lexical novelty therefore did not imply a substantively new rebuttal.
  - The synthesis covered the labor-participation concern shared by both final arguments and separately reconstructed the supporting UBI case from the common evidence and earlier exchange. Its final-argument coverage was high and balanced, but it did not surface that Team A's final turn had departed from its assigned supporting stance.
  - The `-3.2%` employment figure used by both final arguments appears directly in a retrieved snippet. A separate snippet records critics' concern that less work could reduce tax revenue. Characterizing the employment result as “significant,” extending it to reduced economic productivity, and describing “substantial economic risks” are not directly established by the supplied snippets and should be treated as argumentative inferences. This inspection concerns support within the stored snippets only; it is not external factual verification.
- **Token usage:** Not measured.
- **Monetary cost:** Not measured.
- **Observations:**
  1. The final product completed the full live path: decomposition, Tavily-over-MCP research, two debate rounds, deterministic stopping, synthesis, metrics, and atomic artifacts.
  2. Research executed once, and the same consolidated evidence was reused across both rounds.
  3. The system stopped at the configured two-round limit rather than through low novelty.
  4. Both teams added substantial new measured wording in round two because their novelty scores were well above `0.15`.
  5. Research utilization and synthesis representation improved compared with the grounded single-round run.
  6. Team novelty decreased substantially, likely because direct rebuttal framing made the teams share more language and concepts.
  7. Metric v1.0 cannot determine factual correctness, persuasiveness, or whether convergence is desirable.
  8. The final product used six model calls: one decomposition, four argument turns, and one synthesis.
  9. Token counts and monetary cost were not measured.
  10. The cosmetic CLI output `Team A=None, Team B=None` for round one is known and does not affect the structured run artifact or experiment result.
- **Decision:**
  - **KEEP:** The two-round grounded debate as the default final product path.
  - **KEEP:** Deterministic low-novelty stopping and the hard maximum-round bound.
  - **KEEP:** The hierarchical team system as an optional experimental mode, not the default.
  - **CHANGE:** Only minor presentation and documentation issues during final polish; do not begin another architecture milestone.
  - **REMOVE:** Nothing from the frozen experimental history.

## EFF-UBI-001

- **Evaluation:** Final cost and efficiency comparison of three frozen debate architectures
- **Run timestamp (UTC):** `2026-08-06T05:59:38.242408Z`
- **Implementation commit:** `4c79e6bebcf314e798cfcd306d42675fe0ff52fa`
- **Efficiency-report version:** `1.0`
- **Usage-extraction version:** `1.0`
- **Metric version:** `1.0`
- **Model configuration:**
  - Model: `qwen3:8b`
  - Temperature: `0.0`
  - Reasoning output: Disabled
- **Pricing:**
  - Input cost per million tokens: `0`
  - Output cost per million tokens: `0`
  - Pricing basis: `local_zero_default`
- **Question:** Should governments provide universal basic income to all adult citizens?
- **Stances:**
  - Team A: Support the proposition stated in the question.
  - Team B: Oppose the proposition stated in the question.
- **Decomposition:** `complex`
- **Ordered subquestions:**
  1. What are the economic impacts of implementing a universal basic income program in different countries?
  2. How does universal basic income affect labor market participation and employment rates?
  3. What are the fiscal implications of funding a universal basic income for governments?
- **Replay source:** `experiments/runs/m2_1b2b_ubi_qwen3_8b.json`
- **Replay source SHA-256:** `6d88274129f2ae6d1f708847d300a00f76f953bc1d42fbd493bef7f190c506c7`
- **Replay source run-record version:** `1.0`
- **Replay source run timestamp (UTC):** `2026-08-05T11:37:55.647723Z`
- **Replay source architecture:** `milestone-2.1b2b`
- **Replay source implementation commit:** `4a3eb4a4e05a882dd77eb5511e7a245a04fa3e83`
- **Decomposition SHA-256:** `85c6ec20f592ea214a6286c0797ad233ea32ab0873e9f089f0f4b87ff4354b97`
- **Consolidated-evidence SHA-256:** `b536b315b4c136f1456a8f8470f1abe3f6205ee24db9f811f37077a169296c4f`
- **Efficiency artifact:** `experiments/efficiency/final_ubi_efficiency.json`
- **Efficiency artifact SHA-256:** `be47d4fa983489d95112069b65c1c2e5c36419aac380d59915ddc2b1e8536aa5`
- **Replay controls:** All variants reused the exact source question, stances, complexity classification, three ordered subquestions, consolidated evidence, formatted evidence, model, temperature, reasoning setting, and metric version. They ran sequentially. Replay decomposition and replay research made zero model calls; no Tavily, MCP, HTTP, or live decomposition was used.
- **Metric and operational comparison:**

  | Value | Single-round one-shot | Two-round one-shot | Hierarchical single-round |
  | --- | ---: | ---: | ---: |
  | Architecture | `one-shot-single-round-replay` | `one-shot-two-round-replay` | `hierarchical-single-round-replay` |
  | Completed rounds | 1 | 2 | 1 |
  | Termination | `max_rounds` | `max_rounds` | `n/a` |
  | Model calls | 3 | 5 | 7 |
  | Input tokens | 15,941 | 26,392 | 40,459 |
  | Output tokens | 513 | 670 | 2,229 |
  | Total tokens | 16,454 | 27,062 | 42,688 |
  | Tokens per call | 5484.666666666667 | 5412.4 | 6098.285714285715 |
  | Visible output words | 435 | 387 | 411 |
  | All generated words | 435 | 577 | 1,371 |
  | Tokens per visible output word | 37.82528735632184 | 69.92764857881137 | 103.86374695863746 |
  | Latency (seconds) | 58.60167895897757 | 32.37672441598261 | 151.36596191703575 |
  | Research utilization | 0.2290656413443515 | 0.24913793202981818 | 0.2204469505932126 |
  | Team novelty | 1.0 | 0.9655172413793104 | 0.9738562091503268 |
  | Synthesis representation | 0.5413676705034739 | 0.555408662673851 | 0.5195121671258034 |
  | Summary-policy compliance | 1 | 1 | 1 |
  | Research utilization per 1,000 tokens | 0.013921577813562142 | 0.009206190674370637 | 0.005164143332861989 |
  | Team novelty per 1,000 tokens | 0.06077549532028686 | 0.03567797063703017 | 0.02281334822784686 |
  | Synthesis representation per 1,000 tokens | 0.03290188832523847 | 0.02052356302837377 | 0.012169981426297867 |
  | Estimated cost | `0` | `0` | `0` |

- **Single-round one-shot stage and token usage:** All `3` calls reported `usage_metadata`; `3` were available and `0` unavailable.

  | Stage | Round | Input tokens | Output tokens | Total tokens | Duration (seconds) |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Team A argument | 1 | 5,188 | 148 | 5,336 | 20.892746832978446 |
  | Team B argument | 1 | 5,336 | 76 | 5,412 | 16.040547624987084 |
  | Neutral synthesis | 1 | 5,417 | 289 | 5,706 | 21.656592792016454 |

  - Model-call breakdown: Team A argument `1`; Team B argument `1`; neutral synthesis `1`; replay decomposition `0`; replay research `0`; total `3`.
  - Generated stage words: Team A argument `126`; Team B argument `64`; neutral synthesis `245`; all generated `435`.
  - Output word counts: research `1,582` (`200`, `227`, `193`, `178`, `141`, `187`, `187`, `114`, `155`); Team A `126`; Team B `64`; synthesis `245`; total output `2,017`.
  - Research-utilization findings: `0.19444444444444445`, `0.23622047244094488`, `0.12037037037037036`, `0.23404255319148937`, `0.2465753424657534`, `0.1509433962264151`, `0.21518987341772153`, `0.25396825396825395`, `0.4098360655737705`.
  - Team-novelty Jaccard similarity: `0.0`; Team A shingles `90`; Team B shingles `43`; shared `0`; union `133`.
  - Synthesis representation: Team A coverage `0.5421686746987951`; Team B coverage `0.5581395348837209`; mean coverage `0.550154104791258`; balance `0.9840291398150742`.
  - Summary-policy matched patterns: none.

- **Two-round one-shot stage and token usage:** All `5` calls reported `usage_metadata`; `5` were available and `0` unavailable.

  | Stage | Round | Input tokens | Output tokens | Total tokens | Duration (seconds) |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Team A argument | 1 | 5,188 | 148 | 5,336 | 4.044590750010684 |
  | Team B argument | 1 | 5,336 | 76 | 5,412 | 2.120858957991004 |
  | Team A argument | 2 | 5,263 | 74 | 5,337 | 15.133412166964263 |
  | Team B argument | 2 | 5,262 | 76 | 5,338 | 2.5355845839949325 |
  | Neutral synthesis | 2 | 5,343 | 296 | 5,639 | 8.53071295801783 |

  - Model-call breakdown: Team A argument `2`; Team B argument `2`; neutral synthesis `1`; replay decomposition `0`; replay research `0`; total `5`.
  - Generated stage words: round 1 Team A `126`; round 1 Team B `64`; round 2 Team A `64`; round 2 Team B `64`; neutral synthesis `259`; all generated `577`.
  - Output word counts: research `1,582` (`200`, `227`, `193`, `178`, `141`, `187`, `187`, `114`, `155`); final Team A `64`; final Team B `64`; synthesis `259`; total output `1,969`.
  - Research-utilization findings: `0.125`, `0.28346456692913385`, `0.17592592592592593`, `0.3723404255319149`, `0.3698630136986301`, `0.2169811320754717`, `0.21518987341772153`, `0.25396825396825395`, `0.22950819672131148`.
  - Team-novelty Jaccard similarity: `0.034482758620689655`; Team A shingles `45`; Team B shingles `45`; shared `3`; union `87`.
  - Synthesis representation: Team A coverage `0.5555555555555556`; Team B coverage `0.5581395348837209`; mean coverage `0.5568475452196382`; balance `0.9974160206718347`.
  - Summary-policy matched patterns: none.

- **Hierarchical single-round stage and token usage:** All `7` calls reported `usage_metadata`; `7` were available and `0` unavailable.

  | Stage | Team | Input tokens | Output tokens | Total tokens | Duration (seconds) |
  | --- | --- | ---: | ---: | ---: | ---: |
  | Evidence analysis | Team A | 5,481 | 441 | 5,922 | 25.7951591250021 |
  | Argument draft | Team A | 5,697 | 145 | 5,842 | 18.5616069160169 |
  | Review and revision | Team A | 6,057 | 412 | 6,469 | 27.306432332959957 |
  | Evidence analysis | Team B | 5,625 | 508 | 6,133 | 27.920232209027745 |
  | Argument draft | Team B | 5,908 | 113 | 6,021 | 18.17364379100036 |
  | Review and revision | Team B | 6,236 | 376 | 6,612 | 26.28935033403104 |
  | Neutral synthesis | System | 5,455 | 234 | 5,689 | 7.303478709014598 |

  - Model-call breakdown: Team A analysis `1`; Team A draft `1`; Team A review `1`; Team B analysis `1`; Team B draft `1`; Team B review `1`; neutral synthesis `1`; replay decomposition `0`; replay research `0`; total `7`.
  - Generated stage words: Team A analysis `174`; Team A draft `119`; Team A review and revision `291`; Team B analysis `230`; Team B draft `96`; Team B review and revision `267`; neutral synthesis `194`; all generated `1,371`.
  - Output word counts: research `1,582` (`200`, `227`, `193`, `178`, `141`, `187`, `187`, `114`, `155`); final Team A `119`; final Team B `98`; synthesis `194`; total output `1,993`.
  - Research-utilization findings: `0.125`, `0.2440944881889764`, `0.17592592592592593`, `0.26595744680851063`, `0.2602739726027397`, `0.18867924528301888`, `0.189873417721519`, `0.20634920634920634`, `0.32786885245901637`.
  - Team-novelty Jaccard similarity: `0.026143790849673203`; Team A shingles `89`; Team B shingles `68`; shared `4`; union `153`.
  - Synthesis representation: Team A coverage `0.65`; Team B coverage `0.5303030303030303`; mean coverage `0.5901515151515151`; balance `0.8803030303030303`.
  - Summary-policy matched patterns: none.

- **Two-round minus single-round raw deltas:**
  - Model calls: `+2`
  - Input tokens: `+10,451`
  - Output tokens: `+157`
  - Total tokens: `+10,608`, approximately `+64.4706454357603%`
  - Research utilization: `+0.020072290685466676`
  - Team novelty: `-0.03448275862068961`
  - Synthesis representation: `+0.014040992170377131`
  - Summary-policy compliance: `0`
  - Latency: `-26.224954542994965` seconds
  - Estimated cost: `0`
- **Hierarchical minus single-round raw deltas:**
  - Model calls: `+4`
  - Input tokens: `+24,518`
  - Output tokens: `+1,716`
  - Total tokens: `+26,234`, approximately `+159.43843442324055%`
  - Research utilization: `-0.008618690751138913`
  - Team novelty: `-0.02614379084967322`
  - Synthesis representation: `-0.021855503377670504`
  - Summary-policy compliance: `0`
  - Latency: `+92.76428295805817` seconds
  - Estimated cost: `0`
- **Hierarchical minus two-round raw deltas:**
  - Model calls: `+2`
  - Input tokens: `+14,067`
  - Output tokens: `+1,559`
  - Total tokens: `+15,626`, approximately `+57.74148252161703%`
  - Research utilization: `-0.02869098143660559`
  - Team novelty: `+0.008338967771016392`
  - Synthesis representation: `-0.035896495548047636`
  - Summary-policy compliance: `0`
  - Latency: `+118.98923750105314` seconds
  - Estimated cost: `0`
- **Interpretation:**
  1. Single-round one-shot used the fewest model calls and tokens.
  2. Two-round one-shot used approximately `64.5%` more tokens than single-round while producing modest improvements in research utilization and synthesis representation in this controlled run.
  3. Hierarchical single-round used approximately `159.4%` more tokens than single-round while scoring slightly worse on research utilization and synthesis representation.
  4. The hierarchical architecture was the most token-intensive and slowest variant in this run.
  5. Two-round one-shot remains the final quality-oriented default.
  6. Single-round remains the economical mode.
  7. Hierarchical teams remain an optional architecture demonstration rather than the cost-efficient default.
  8. The measured latency of two-round execution was lower than single-round, but this is not evidence of an architectural speed advantage.
  9. The variants ran sequentially, so Ollama warm-up, caching, machine load, or execution-order effects may have influenced latency.
  10. Reliable latency conclusions require repeated, randomized, or counterbalanced runs.
  11. Estimated monetary cost was zero only because both user-configured API rates were zero.
  12. Zero estimated API price excludes local hardware, electricity, cooling, maintenance, and personnel costs.
  13. Token-efficiency ratios are descriptive and do not measure factual correctness or persuasiveness.
  14. This is one controlled topic and is not statistically representative.
- **Practical recommendation:**
  - Use single-round mode when minimizing token usage matters most.
  - Use two-round mode as the final default when the modest measured quality gains justify approximately `64.5%` more tokens.
  - Keep hierarchical mode for architecture experiments rather than routine use.
  - Do not infer an aggregate winner or weighted quality score from these separate measurements.
