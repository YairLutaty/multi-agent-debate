# Deterministic Metrics

## Purpose

Milestone 1 adds a deterministic measurement layer to the debate system.

A metric is deterministic when the same saved debate output always receives the
same score. The scoring layer must not call an LLM, search engine, retrieval
system, or external service.

The metrics are proxies rather than complete measures of debate quality. They
are intended to support repeatable comparisons between architectural versions.

No weighted overall score is defined at this stage. Each metric will be reported
separately so that improvements in one area do not hide degradation in another.

---

## Text tokenization and normalization

Metric version 1.0 uses the exact Unicode-aware word regex:

```text
[^\W_]+
```

`tokenize_words(text)` applies Unicode NFKC normalization, case folding, and
the word regex. It removes punctuation through token extraction but retains all
word tokens, including stop words. Operational word counts use this function.

`normalize_text(text)` calls the same tokenization logic and then removes the
following exact frozen stop-word list:

```text
a, an, and, are, as, at, be, been, being, but, by, can, could, d, did, do,
does, for, from, had, has, have, he, her, him, his, i, if, in, into, is, it,
its, ll, m, may, might, of, on, or, our, re, s, she, should, t, than, that,
the, their, them, they, this, those, to, ve, was, we, were, will, with, would,
you, your
```

Lexical quality metrics use `normalize_text`. Operational word counts use
`tokenize_words`, so their counts include stop words. Negations such as `not`,
`no`, and `nor` are intentionally retained.

Neither function uses an LLM, embeddings, stemming, lemmatization, or an
external NLP model. The regex and stop-word list must remain unchanged after the
Milestone 0 baseline is scored unless the change is recorded as a new metric
version.

---

## Metric 1 — Research utilization

### Question answered

How much of the shared research pool is reflected in the downstream debate?

### Formula

For each research finding `F_i` and downstream output `O`:

```text
coverage(F_i, O) =
    number of normalized tokens shared by F_i and O
    divided by
    number of normalized tokens in F_i
```

Both inputs are sets of distinct normalized tokens:

```text
coverage(F_i, O) =
    size(set(F_i) intersect set(O))
    divided by
    size(set(F_i))
```

This prevents repeated words from inflating coverage. An empty normalized
finding receives coverage `0.0`. If the findings collection is empty, the final
research-utilization score is `0.0`.

The downstream outputs are:

* Team A's argument;
* Team B's argument;
* the neutral synthesis.

For each finding, retain the highest coverage obtained by any downstream output:

```text
finding_utilization(F_i) =
    max(
        coverage(F_i, Team A),
        coverage(F_i, Team B),
        coverage(F_i, synthesis)
    )
```

The final score is:

```text
research_utilization =
    mean(finding_utilization(F_i))
```

### Range

`0.0` to `1.0`; higher is better.

### Limitation

Lexical overlap does not detect all paraphrases and does not prove that a
finding was interpreted correctly.

---

## Metric 2 — Team novelty

### Question answered

Does Team B contribute materially different language rather than merely
repeating Team A?

### Formula

1. Normalize each team's argument.
2. Construct sets of consecutive three-token shingles.
3. Compute Jaccard similarity:

```text
similarity =
    size(shingles_A intersect shingles_B)
    divided by
    size(shingles_A union shingles_B)
```

4. Convert similarity to novelty:

```text
team_novelty = 1 - similarity
```

### Range

`0.0` to `1.0`; higher indicates less direct repetition.

"Empty" means empty after lexical normalization. If either normalized argument
is empty, similarity is `1.0` and novelty is `0.0`.

If both arguments are nonempty but neither is long enough to produce a
three-token shingle, similarity is `1.0` and novelty is `0.0`. If one nonempty
argument produces no shingles and the other produces at least one, their union
is nonempty, similarity is `0.0`, and novelty is `1.0`.

### Limitation

High novelty does not necessarily mean that Team B provides a strong rebuttal.
A completely irrelevant response could also receive a high novelty score.
When one nonempty argument is too short to form a three-token shingle and the
other is long enough, the resulting novelty of `1.0` is a length artifact and
should be interpreted cautiously.

---

## Metric 3 — Synthesis representation

### Question answered

Does the final synthesis represent both teams, rather than mainly repeating one
side?

### Formula

Calculate the proportion of each team's normalized content tokens represented
in the synthesis:

```text
coverage_A =
    size(tokens_A intersect tokens_synthesis)
    divided by
    size(tokens_A)

coverage_B =
    size(tokens_B intersect tokens_synthesis)
    divided by
    size(tokens_B)
```

Calculate:

```text
mean_coverage = (coverage_A + coverage_B) / 2

balance = 1 - abs(coverage_A - coverage_B)

synthesis_representation = mean_coverage * balance
```

This rewards both overall representation and similarity in how much attention
the synthesis gives each side.

### Range

`0.0` to `1.0`; higher is better.

All intersections and denominators use sets of distinct normalized tokens. If
either normalized team argument or the normalized synthesis is empty, both
coverage components, mean coverage, balance, and the final score are `0.0`.

### Limitation

Lexical representation does not guarantee conceptual fairness or factual
accuracy.

---

## Metric 4 — Summary-policy compliance

### Question answered

Does the neutral summarizer avoid explicitly selecting a winner or issuing a
recommendation?

### Formula

Apply Unicode NFKC normalization, then use the following exact fixed collection
of case-insensitive regular expressions. Preserve the pattern IDs and report all
matches in the listed order:

```text
team_a_wins:
    \bteam\s+a\s+wins\b
team_b_wins:
    \bteam\s+b\s+wins\b
team_a_more_convincing:
    \bteam\s+a\s+is\s+more\s+convincing\b
team_b_more_convincing:
    \bteam\s+b\s+is\s+more\s+convincing\b
stronger_argument:
    \bthe\s+stronger\s+argument\s+is\b
winning_side:
    \bthe\s+winning\s+side\s+is\b
my_verdict:
    \bmy(?:\s+final)?\s+verdict\s+is\b
i_conclude:
    \bi(?:\s+therefore)?\s+conclude\s+that\b
i_recommend:
    \bi(?:\s+would)?\s+recommend\b
```

Return:

```text
summary_policy_compliance = 1
```

when no prohibited pattern is found, and:

```text
summary_policy_compliance = 0
```

when at least one pattern is found.

An empty synthesis has no prohibited match and therefore receives compliance
`1`. Transcript parsing independently rejects an empty required synthesis.

### Range

`0` or `1`; `1` is compliant.

### Limitation

A fixed pattern list cannot detect every indirect verdict and may miss subtle
bias.

---

## Operational measurement — Output word count

Count full tokens from `tokenize_words`, including stop words, separately for:

* research findings;
* Team A argument;
* Team B argument;
* synthesis.

Also report:

```text
total_output_words
```

This is not treated as a quality score. It provides an initial efficiency
measurement that can later be compared with token usage, latency, and monetary
or compute cost.

The research count is also retained per finding. `total_output_words` is the
sum of all research-finding words, Team A argument words, Team B argument words,
and synthesis words. It excludes the question, stances, metadata, and timing
footer.

---

## Deterministic JSON serialization

Scorecard JSON contains no timestamp, source path, or runtime metadata. It uses:

```python
json.dumps(
    scorecard_data,
    sort_keys=True,
    indent=2,
    ensure_ascii=False,
    allow_nan=False,
) + "\n"
```

The serialized document ends with exactly one newline. Repeated scoring of the
same saved transcript must therefore produce byte-for-byte identical JSON.

---

## Deferred metrics

Citation density and academic-source proportion are intentionally deferred.

Milestone 0 uses simulated model-internal research and explicitly prohibits
fabricated citations. Citation-related metrics would therefore be trivially zero
and would not provide a meaningful quality comparison yet.

These metrics should be introduced when externally grounded research is added.

---

## Metric version

This initial metric specification is:

```text
metric_version = "1.0"
```

The formulas, full-token and lexical-normalization rules, word regex, stop-word
list, set-overlap semantics, empty and short-input rules, verdict patterns, and
JSON serialization rules must be versioned if they change after the baseline is
scored.
