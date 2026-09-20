---
name: signal-fusion-analysis
description: Explain a frozen LTE/WiFi/DVB-T deterministic fusion result from one blind signal_fusion JSON. The input combines a three-model IQ ensemble, a frozen LTE/DVB-T periodicity gate, and one complete-region 64-dimensional feature vector. Do not use for evidence generation, training, reclassification, or audit evaluation.
---

# Signal Fusion Analysis

Explain one existing deterministic result. The project code performs
classification and gate resolution; this skill validates and interprets the public
evidence without changing the label or review status.

## Blind-input boundary

Accept exactly one schema-version-3 JSON file whose `bundle_type` is
`hermes_signal_fusion_input`, whose `evidence_type` is
`blind_region_ensemble_with_periodicity_gate_and_global_features`, whose
`analysis_id` matches `case_<digits>`, and whose filename is that same neutral
identifier.

Read only that case and these five paths declared inside it:

- the four entries in `global_feature_evidence.reference_document_paths`;
- `periodicity_evidence.reference_document_path`.

Do not inspect sibling files, audit files, datasets, command history, provenance, or
other cases. Reject the input as non-blind if it exposes ground truth, source IDs,
source paths, locations, center frequency, receiver gain, hidden noise settings, or
a class-bearing case name.

Allowed context includes sample rate, duration, sample count, window configuration,
model observations, periodicity observations, global features, and the five declared
reference documents. Infer limitations only from public observations.

## Evidence contract

Validate before explaining:

- `iq_ensemble_evidence` contains three members, their region Top3 values, local
  windows, and the member-disagreement risk gate;
- `periodicity_evidence` contains three candidate measurements, LTE/DVB-T scores,
  margin, the frozen reliability margin, and `eligible/reliable/resolved/changed`;
- `global_feature_evidence.values` contains exactly 64 named values extracted once
  from the complete continuous region;
- `deterministic_fusion_result` contains the frozen operational result;
- `decision_contract.result_is_frozen` is `true`.

Require these invariants:

- `resolved` implies both `eligible` and `reliable`;
- `changed` implies `resolved`;
- an ensemble WiFi decision or a conflict containing a WiFi member is never changed
  by the periodicity gate;
- an IQ ensemble with unanimous member Top1 is never changed;
- member disagreement plus `resolved=true` produces `decision_status=accept`;
- member disagreement plus `resolved=false` produces
  `decision_status=review_required` and `resolution=unresolved_review`;
- `decision_status=accept` means `final_label=prediction_label` and
  `provisional_label=null`;
- `decision_status=review_required` means `final_label=null` and
  `provisional_label=prediction_label`.

Reject internally inconsistent evidence instead of repairing it.

## Reference documents

Read all five declared documents completely. The three feature documents define the
64 values. The technology document gives cautious LTE/WiFi/DVB-T context in the
observed 1 MHz view. The periodicity document defines candidate periods, scores,
margin, the frozen threshold, and gate scope.

Inspect all 64 values, but cite only material values. Do not invent thresholds, treat
correlated features as independent votes, or infer a protocol from one handcrafted
feature. A normalized periodicity correlation is not a probability.

## Frozen-result rules

Copy these fields exactly from `deterministic_fusion_result`:

- `decision_status`;
- `prediction_label`;
- `final_label`;
- `provisional_label`.

Do not recalculate periodicity scores, change the reliability threshold, rerun the
gate, introduce fusion weights, or select a different class. The 64-dimensional
vector is explanatory physical context only; it cannot change an accepted label or
resolve a review case.

Explain the resolution explicitly:

- `ensemble_unanimous`: the learned members agree and the ensemble label is used;
- `ensemble_confirmed_by_periodicity`: periodic evidence resolves an LTE/DVB-T
  member disagreement without changing the ensemble label;
- `periodicity_changed_label`: the frozen gate has already changed the ensemble
  label; explain both the original disagreement and the periodic evidence;
- `unresolved_review`: describe why the gate cannot act and keep the candidate as
  provisional, never final.

`confidence_level` describes explanation-level operational risk, not a new class
probability. It must be `low` for every `review_required` case. For `accept` cases,
lower it when evidence is close to the threshold, model support is internally weak,
or global physical observations expose material limitations. Feature conflict may be
reported as an anomaly, but cannot alter the frozen result.

## Output

Return exactly one JSON object in the user's language:

```json
{
  "decision_status": "accept | review_required",
  "prediction_label": "LTE | WiFi | DVB-T",
  "final_label": "LTE | WiFi | DVB-T | null",
  "provisional_label": "LTE | WiFi | DVB-T | null",
  "confidence_level": "high | medium | low",
  "summary": "concise explanation of the frozen result or unresolved review",
  "model_evidence": [],
  "periodicity_evidence": [],
  "feature_evidence": [],
  "conflicting_evidence": [],
  "limitations": []
}
```

Do not reveal document paths. Only after this blind explanation is frozen, and only
when the user separately requests evaluation, may an evaluator read the matching
private audit. Never revise the blind explanation after hidden information is read.
