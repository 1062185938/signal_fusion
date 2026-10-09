---
name: signal-fusion-review-experiment
description: Blindly adjudicate one LTE/WiFi/DVB-T case where an IQ classifier and a 64-feature probe disagree. Use for the cross-location A/B review experiment only, not for training, evidence generation, production inference, or audit evaluation.
---

# Signal Fusion Review Experiment

Analyze exactly one disagreement case. Decide whether to keep the IQ branch,
change to another supplied class, or abstain. This is an experimental decision,
not a production rule.

## Blind boundary

Accept exactly one schema-version-2 JSON file with:

- `bundle_type=signal_fusion_cross_location_review_input`;
- `evidence_type=blind_iq_feature_disagreement_adjudication`;
- an opaque `analysis_id` matching `review_<digits>` and the filename;
- one `iq_branch` and one `feature_probe_branch` whose Top1 labels differ.

Read only that case and the files listed in `reference_document_paths`. Do not
inspect sibling cases, private audit files, datasets, outputs, provenance,
command history, or previous answers. Reject input that exposes ground truth,
correctness, selection cohort, source identity, location, fold, receiver gain,
center frequency, hidden noise settings, or an existing fused decision.

## Evidence variants

Both variants contain:

- IQ-branch Top3 probabilities and uncertainty summaries;
- feature-probe Top3 probabilities and uncertainty summaries;
- the LTE/WiFi/DVB-T technology reference.

One variant additionally contains `physical_feature_evidence`: exactly 64 named
features from one complete region, plus the three feature-definition documents.

Treat the 64-dimensional feature vector as the physical basis of the feature-probe
branch, not as a third independent classifier or an additional vote.

The physical features may nevertheless be used to assess whether the
feature-probe decision has a coherent physical interpretation. They may support,
weaken, or leave unchanged the credibility of the feature-probe hypothesis.

Do not count correlated physical features as independent pieces of evidence.
When several features describe the same underlying phenomenon, such as
continuous occupancy, burstiness, spectral flatness, or envelope variability,
treat them as one physical evidence pattern.

Inspect the complete vector, but cite only observations that are material to the
current disagreement and supported by the reference documents.

A physical observation does not need to uniquely identify a protocol in order to
be useful. It may provide directional evidence when it is more compatible with
one supplied class than with the competing explanation in the observed region.
However, shared OFDM properties that are similarly compatible with all supplied
classes are non-discriminative and must not influence the decision.

If `physical_feature_evidence` is absent, return an empty
`physical_feature_evidence` list. Do not speculate about unobserved features.

Do not infer which A/B set the case belongs to or compare it with another case.

## Reasoning rules

The purpose of the review is to determine whether the available evidence gives a
coherent reason to retain the IQ label, overturn it in favor of another supplied
class, or abstain.

The two branch outputs are not calibrated against each other. Therefore:

- do not compare raw probability magnitudes across the two branches as if they
  shared a common probability scale;
- do not average branch probabilities;
- do not invent fusion weights;
- do not create class-specific numerical thresholds or fixed probability-based
  if/else rules.

Branch-internal uncertainty may still be examined. Within each branch, use the
Top1/Top2 relationship, margin, entropy, and the shape of the branch distribution
to describe whether that branch is internally concentrated, competitive, or
near-tied. This is an assessment of the structure of one branch and is not a
cross-branch probability comparison.

Evaluate the evidence in the following order.

### 1. Characterize the IQ branch

Identify:

- the IQ Top1 label;
- whether the IQ branch is internally concentrated or has a meaningful competing
  class;
- which class is the principal alternative to the IQ Top1.

A substantial IQ runner-up is important evidence. In particular, if the IQ
runner-up is the same class as the feature-probe Top1, treat this as cross-branch
agreement on the main competing hypothesis rather than as a completely unrelated
model conflict.

### 2. Characterize the feature-probe branch

Identify:

- the feature-probe Top1 label;
- whether the feature-probe branch is internally concentrated, competitive, or
  near-tied;
- whether its Top1 represents a meaningful alternative to the IQ label.

A weak or near-tied feature-probe output should not overturn a clearly supported
IQ decision merely because its Top1 label differs.

A concentrated feature-probe output is stronger evidence for review, but its raw
probability magnitude must not be directly compared with the IQ probability.

### 3. Examine physical support when available

Use `physical_feature_evidence` to determine whether the feature-probe hypothesis
has a coherent physical basis.

The physical vector is not a third vote. Instead ask:

- which physical phenomenon or signal behavior the material features describe;
- whether that behavior is directionally more compatible with the
  feature-probe label, the IQ label, both, or neither;
- whether a plausible competing explanation substantially weakens that
  interpretation.

Multiple correlated features that describe the same phenomenon count as one
coherent physical pattern, not multiple independent votes.

Protocol-specific measurements such as synchronization structures, pilots,
preambles, carrier spacing, TPS, or guard intervals are strong evidence when
they were actually measured, but they are not mandatory for `change`.

Do not require an observation to completely exclude every competing class before
it can provide useful directional evidence.

### 4. Build a coherent evidence chain

A coherent evidence chain may combine different roles of evidence without
treating them as independent votes.

For example, the following pattern may support overturning the IQ label:

- the IQ branch is not internally decisive and already assigns meaningful support
  to a competing class;
- that competing class is the feature-probe Top1;
- the feature-probe branch is internally coherent rather than near-tied;
- available physical observations provide a plausible and document-supported
  explanation for why that class is favored;
- no comparably strong observation contradicts that interpretation.

The physical observations in such a chain support the interpretation of the
feature-probe branch; they must not be counted as an additional numerical vote.

A coherent evidence chain may also exist without raw physical features when the
IQ branch itself shows substantial ambiguity toward the same class selected by a
clearly structured feature-probe branch. In such cases, absence of the raw
64-dimensional vector is a limitation, but it does not automatically require
abstention.

### 5. Make the decision

Use `keep` when the IQ label remains better supported after reviewing the
conflicting branch. Typical evidence for `keep` includes:

- the IQ branch is internally coherent;
- the feature-probe branch is weak, near-tied, or internally ambiguous;
- the feature-probe hypothesis lacks coherent physical support when physical
  evidence is available;
- the available physical observations are shared or non-discriminative rather
  than supportive of the competing label.

Use `change` when a coherent evidence chain supports another supplied class
strongly enough to make retaining the IQ label less well supported. `change`
does not require a protocol-exclusive signature. It may be justified when:

- the IQ branch itself shows meaningful competition involving the alternative
  class;
- the feature-probe branch consistently favors that same alternative;
- and, when physical evidence is available, the material physical observations
  provide a compatible explanation without equally strong contradictory
  evidence.

Do not use `change` merely because the feature-probe probability is numerically
larger than the IQ probability.

Use `abstain` when the evidence remains genuinely unresolved, including cases
where:

- both branches are internally strong and directly opposed, with no material
  evidence capable of favoring one interpretation;
- the relevant physical observations are equally compatible with both competing
  classes;
- the available evidence is contradictory in different directions;
- or neither `keep` nor `change` has a coherent evidence chain.

Do not use `abstain` merely because no protocol-exclusive structure was measured.
Absence of such a structure is a limitation, not by itself a reason to abstain.

Shared OFDM traits such as high PAPR, broad occupancy, high spectral entropy,
spectral flatness, or noise-like envelopes do not by themselves distinguish
LTE, WiFi, and DVB-T.

Do not claim protocol synchronization, standard channel bandwidth, or a
protocol-specific structure unless it was actually measured.

Never force a label merely to complete the experiment. Because every input is a
branch conflict, confidence may be `medium` or `low`, never `high`.

## Output contract

Return exactly one JSON object in the user's language:

```json
{
  "analysis_id": "review_0001",
  "decision": "keep | change | abstain",
  "recommended_label": "LTE | WiFi | DVB-T | null",
  "confidence_level": "medium | low",
  "summary": "concise blind adjudication or reason for abstention",
  "iq_evidence": [],
  "feature_probe_evidence": [],
  "physical_feature_evidence": [],
  "conflicting_evidence": [],
  "limitations": []
}
```

The relationships are strict:

- `keep`: `recommended_label` equals `iq_branch.top1.label`;
- `change`: `recommended_label` is a valid supplied class and differs from the
  IQ Top1 label;
- `abstain`: `recommended_label=null` and `confidence_level=low`.

For a case without the raw 64-dimensional vector,
`physical_feature_evidence` must be empty. Do not reveal reference paths.

Once returned, the answer is frozen. Only the separate evaluator may join it
with private truth and selection metadata. Never revise the blind response after
hidden information becomes known.
