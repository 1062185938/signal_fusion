---
name: signal-fusion-analysis
description: Interpret an existing blind signal_fusion evidence JSON containing an IQ classifier, a region-level feature classifier, and one global 62-dimensional feature vector. Do not use for evidence generation, dataset preparation, training, or evaluation.
---

# Signal Fusion Analysis

Produce one explainable signal-class decision from an existing blind evidence
file. The project package generates evidence; this skill only validates and
interprets the public JSON.

## Blind-input boundary

Accept exactly one JSON file whose `bundle_type` is
`hermes_signal_fusion_input`, whose `analysis_id` matches `case_<digits>`, and
whose filename is the same neutral case identifier. Read only that case file.

Do not inspect sibling files, audit directories, source datasets, command
history, or provenance. Do not generate the evidence in the same reasoning
session. Reject the input as non-blind if it exposes hidden experiment
conditions, ground truth, source identifiers, source paths, or a class-bearing
case name.

Allowed context includes sampling rate, duration, sample count, window length,
window count, model observations, and extracted feature observations. Infer
signal quality only from those observations; do not assume an undisclosed
capture or experiment condition.

## Evidence contract

Validate these fields before reasoning:

- `iq_model_evidence.region_top3`: probabilities averaged across selected local
  windows;
- `iq_model_evidence.window_predictions`: each window's Top1 label and
  confidence;
- `iq_model_evidence.window_agreement`: agreement with the IQ region Top1;
- `feature_model_evidence.region_top3`: probabilities from classifying the one
  complete-region 62-dimensional feature vector;
- `fusion_result`: the frozen weighted-probability result and final label;
- `feature_evidence`: one 62-dimensional vector from the complete continuous
  region, not statistics across windows;
- `feature_evidence.input`: original length, actual feature-input length, and
  whether the 16384-point feature limit caused truncation.

## Feature documentation

Resolve feature references relative to `feature_evidence.reference_root`.
Before interpreting the values, read all three documents completely:

- `time_domain_iq_features.md`;
- `frequency_domain_iq_features.md`;
- `time_frequency_iq_features.md`.

Continue through pagination until each document ends. The documents define the
features and their physical meaning; they are not class templates. Do not
invent class thresholds, class ranges, or independent votes from correlated
features. Do not call a value high, low, narrowband, wideband,
constant-envelope, or class-supporting unless the referenced documentation
provides the necessary interpretation boundary.

## Fusion decision

Treat `iq_model_evidence` and `feature_model_evidence` as two separate
classifier branches. Window Top1 values describe only the IQ branch's local
stability; they are not another classifier. Treat disagreement within or
between branches as uncertainty.

Use the global feature vector to describe the physical behavior of the complete
region and explain why the feature classifier may agree or disagree with the IQ
classifier. Raw handcrafted values are explanations, while
`feature_model_evidence` is the trained class mapping for those values.

Use `fusion_result.final_label` as the final classification. Its probabilities
come from the frozen weights shown in `fusion_result.weights`; do not change the
weights, recompute an alternative result, or replace the final label. Use the
two branches and raw features to explain agreement, conflict, and limitations.

If the feature input was truncated, state that its global description covers
only the recorded leading portion. Lower confidence when both classifier
branches are weak or disagree materially.

## Output

Return exactly one JSON object with the fields specified by `required_output`:

```json
{
  "final_label": "exactly fusion_result.final_label",
  "confidence_level": "high | medium | low",
  "summary": "concise fusion conclusion",
  "model_evidence": [],
  "feature_evidence": [],
  "conflicting_evidence": [],
  "limitations": []
}
```

Write explanations in the user's language. Name every feature used and ground
its interpretation in the feature documentation. If evidence cannot reliably
distinguish classes, choose the most plausible Top3 label with `low` confidence
and explain the limitation.

Only after returning and freezing the blind result, and only when the user
separately asks for evaluation, may an evaluator read the matching private
audit file. Never revise the original result after hidden information is read.
