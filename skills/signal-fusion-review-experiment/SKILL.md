---
name: signal-fusion-review-experiment
description: Produce a blind, non-operational LTE/WiFi/DVB-T recommendation or abstention for one unresolved signal_fusion review experiment case. Use only for P6-B review cases, not for production fusion, frozen-result explanation, training, evidence generation, or audit evaluation.
---

# Signal Fusion Review Experiment

Analyze one unresolved review case as a blind experiment. The output is a
recommendation, never an operational final label.

## Blind boundary

Accept exactly one schema-version-1 JSON file with:

- `bundle_type=signal_fusion_review_experiment_input`;
- `evidence_type=blind_unresolved_review_recommendation`;
- a neutral `analysis_id` matching `review_<digits>` and the filename.

Read only that case and the paths explicitly listed in
`reference_document_paths`. Do not inspect sibling files, audit files, other
cases, datasets, provenance, command history, or previous conversations.
Reject a case that exposes ground truth, correctness, source identifiers,
locations, receiver gain, center frequency, hidden noise settings, or a
class-bearing filename.

## Evidence variants

Every valid case contains:

- three-member `iq_ensemble_evidence`;
- unresolved `periodicity_evidence`;
- `review_context` with a provisional, non-final label;
- LTE/WiFi/DVB-T technology and periodicity references.

Some cases additionally contain `global_feature_evidence` with one complete-region
64-dimensional vector and three feature references. Use only evidence present in
the current case:

- without `global_feature_evidence`, do not seek, infer, or discuss missing feature
  values; return an empty `feature_evidence` list;
- with `global_feature_evidence`, verify exactly 64 named values, inspect all of
  them, and cite only material observations grounded in the declared feature
  references.

Do not compare variants or infer which experimental set the case belongs to.

## Review invariants

Validate before reasoning:

- the three IQ member Top1 labels disagree;
- `review_context.decision_status` is `review_required`;
- `periodicity_evidence.frozen_gate.resolved` and `changed` are both false;
- no `final_label` or ground truth is present;
- normalized periodicity correlations are measurements, not probabilities;
- the provisional label is a baseline candidate, not an answer to copy.

Reject inconsistent input rather than repairing it.

## Recommendation

Use IQ model behavior and periodicity evidence in every case. When the optional
64-dimensional vector exists, use it as additional physical context without
double-counting correlated measurements or inventing class thresholds.

Return `recommend` only when the supplied evidence supports one label coherently.
The recommendation may retain or change the provisional label. Return `abstain`
when evidence remains ambiguous, mutually conflicting, or physically
non-discriminative. Do not force a label merely to complete the experiment.

Do not create new fusion weights, refit the periodicity threshold, treat member
votes as calibrated independent evidence, infer standard channel bandwidth from
the 1 MHz view, or claim protocol-specific synchronization that was not measured.
Because every input was rejected by the deterministic system, confidence may be
`low` or `medium`, never `high`.

## Output

Return exactly one JSON object in the user's language:

```json
{
  "analysis_id": "review_0001",
  "recommendation_status": "recommend | abstain",
  "recommended_label": "LTE | WiFi | DVB-T | null",
  "confidence_level": "medium | low",
  "summary": "concise blind recommendation or reason for abstention",
  "model_evidence": [],
  "periodicity_evidence": [],
  "feature_evidence": [],
  "conflicting_evidence": [],
  "limitations": []
}
```

`recommendation_status=recommend` requires a non-null label.
`recommendation_status=abstain` requires `recommended_label=null` and
`confidence_level=low`. Do not reveal reference paths.

Once returned, the recommendation is frozen. Only a separate evaluator may later
read the private audit; never revise the blind response after hidden information is
known.
