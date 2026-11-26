# Static q¹ low-resistance capacity check

Status: `completed_static_conditional_capacity_check`.

No q¹ trajectory or fit was run. The shared-parameter statements below concern the bare 1R gate only.

## rectangle

```json
{
  "qcap": 0.21050001227684284,
  "fcap": 0.7222498212074407,
  "selected_a_upper_s": 0.0009538570932927105,
  "negative_protocol_node_envelope_max_a": 0.04286674573247503,
  "negative_max_node_indices": [
    300,
    376
  ],
  "negative_voltage_columns_inclusive": [
    0,
    376
  ],
  "negative_envelope_upper_node_voltage_v": 2.506119985502019,
  "negative_envelope_all_gamma_rows": true,
  "negative_envelope_includes_nearzero_linear_extension": true,
  "negative_envelope_includes_right_bracketing_column": true,
  "reset_max_v": 2.5,
  "negative_boundary_rhs_lower_a": 0.10746761150882182,
  "selected_negative_boundary_lhs_upper_a": 0.0452513884657068,
  "selected_negative_boundary_margin_a": 0.06221622304311502,
  "selected_joint_rectangle_sufficient_check_passed": true,
  "negative_check_is_sufficient_not_necessary": true
}
```

## positive_readout

```json
{
  "read_voltage_v": 0.0500000007450581,
  "soft_at_qcap_a": 6.195017703301536e-07,
  "gamma_lower_index": 219,
  "voltage_lower_index": 307,
  "gamma_fraction": 0.21032236717194108,
  "voltage_fraction": 0.9915045471691661,
  "feature_id": "1r.set.return.R_0p05V",
  "original_R_gate_upper_ohm": 187.67999555827214,
  "original_observed_R_eligible_ohm": [
    206.80533691830792,
    143.0541990515218,
    83.03088743487612,
    97.00418227845475,
    79.46145923429663,
    76.32068864351967,
    85.16647266938544,
    66.0849306287595,
    106.50617861420281,
    315.5812528248462,
    103.63195633285355,
    101.13390595888092,
    61.54003510602705,
    78.33259121986214
  ],
  "original_observed_R_median_ohm": 91.0853274739201,
  "fcap": 0.7222498212074407,
  "negative_protocol_node_envelope_max_a": 0.04286674573247503,
  "current_required_for_upper_R_a": 0.0002664109224658084
}
```

## selected_capacity

```json
{
  "selected_A_s": 0.0013206747378601963,
  "total_read_current_upper_a": 4.8312357145644636e-05,
  "upper_below_compliance": true,
  "unconstrained_read_resistance_lower_ohm": 1034.9319242347422,
  "current_capacity_less_than_required": true,
  "all_positive_kappa_excluded_for_original_R_gate": true,
  "applies_to_all_original_eligible_records_hence_median": true
}
```

## necessary_A

```json
{
  "raw_A_threshold_s": 0.007360096435622046,
  "positive_threshold": true,
  "verified_necessary_lower_bound_s": 0.007360096435622046,
  "attainability_or_sufficiency_claim": false,
  "threshold_negative_boundary_lhs_upper_a": 0.05615631656921889,
  "threshold_negative_boundary_margin_a": 0.05131129493960293,
  "threshold_joint_rectangle_sufficient_check_passed": true,
  "bound_scope": "All A below threshold, fixed J/b/rho/lambda/readout/entries/protocol; any kappa>0 and kappa_a>=0"
}
```

The selected A is capacity-excluded from the original bare low-resistance median gate for every positive kappa under the joint invariant rectangle. The positive A threshold is a verified necessary lower bound under the recorded fixed conditions; reaching it does not guarantee a valid fit.

A necessary bound is not an attainable-state claim or a sufficient fit criterion.
All statements retain the fixed J, b, rho, lambda, readout, entries and original protocol.
Changing these conditions requires rechecking the inequalities.

```json
{
  "saved_protocol_checks": 14,
  "negative_stored_node_maxima": 1,
  "positive_monotonicity_audits": 1,
  "positive_readouts_attempted": 1,
  "positive_readouts_completed": 1,
  "negative_forward_readouts": 0,
  "ODEs": 0,
  "fits": 0,
  "inverses": 0,
  "roots": 0,
  "quadratures": 0
}
```

Inputs unchanged: `True`. Ordinary floating point, not interval arithmetic.
