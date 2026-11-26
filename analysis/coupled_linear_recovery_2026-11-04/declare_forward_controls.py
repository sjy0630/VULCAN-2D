"""Declare finite capacity-adjusted q1 controls using saved arithmetic only."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'analysis/coupled_training_calibration_2026-11-04/calibration_run/RESULT.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = HERE / 'FORWARD_DECLARATION.json'
    if destination.exists():
        raise RuntimeError('Final declaration exists; do not overwrite or rerun automatically')
    seed = json.loads((HERE / 'SEED_DERIVATION.json').read_text())
    capacity = json.loads((HERE / 'CAPACITY_CHECK.json').read_text())
    parent = json.loads(PARENT.read_text())
    assert parent['numerical_final_candidate_verified'] and parent['sources_unchanged']
    assert seed['status'] == 'completed_reference_seed_declaration_no_forward_evaluation'
    assert capacity['status'] == 'completed_static_conditional_capacity_check'
    assert capacity['selected_capacity']['all_positive_kappa_excluded_for_original_R_gate']
    sources = {}
    for result in (seed, capacity):
        assert result['inputs_unchanged'] and result['input_sha256_before'] == result['input_sha256_after']
        for filename, digest in result['input_sha256_before'].items():
            if filename in sources:
                assert sources[filename] == digest
            sources[filename] = digest
    for path in (PARENT, HERE / 'SEED_DERIVATION.json', HERE / 'CAPACITY_CHECK.json',
                 HERE / 'CAPACITY_SEED_PLAN.md', Path(__file__).resolve()):
        name, digest = str(path.resolve()), sha(path)
        if name in sources:
            assert sources[name] == digest
        sources[name] = digest
    assert all(sha(path) == digest for path, digest in sources.items())

    selected = parent['last_accepted_center']
    assert selected['request_id'] == 26
    parameters = selected['parameters']
    entries = parent['fixed_segment_entries']
    assert entries == seed['initial_states']
    read, rectangle = capacity['positive_readout'], capacity['rectangle']
    fraction = 0.9
    a_seed = (1 / read['original_observed_R_median_ohm'] - read['soft_at_qcap_a'] / read['read_voltage_v']) / (fraction * rectangle['fcap'])
    lower = capacity['necessary_A']['verified_necessary_lower_bound_s']
    assert lower is not None and 0 < lower < a_seed and math.isfinite(a_seed)
    negative_bound = rectangle['negative_protocol_node_envelope_max_a'] + a_seed * rectangle['fcap'] * rectangle['reset_max_v']
    assert negative_bound <= rectangle['negative_boundary_rhs_lower_a']
    read_ceiling = read['soft_at_qcap_a'] + read['read_voltage_v'] * a_seed * rectangle['fcap']
    assert read_ceiling < 0.001
    reference = seed['q1_kappa_reference']
    assert reference == seed['kappa1_reference_per_v2'] and reference > 0
    candidates = []
    tuples = [('q1_order_only', dict(parameters), 'Same numerical parameters; only q recovery order differs. Known low-resistance capacity exclusion.')] + [
        (f'q1_capacity_rate_x{multiplier}', dict(parameters, a_max_s=a_seed, kappa_positive_per_v2=multiplier * reference),
         f'Prospective capacity-adjusted q1 seed, common rate reference multiplied by {multiplier}; not an order-only comparison.')
        for multiplier in (1, 3, 10)
    ]
    for label, candidate, role in tuples:
        changed = sorted(key for key in parameters if parameters[key] != candidate[key])
        assert set(changed) <= {'a_max_s', 'kappa_positive_per_v2'}
        candidates.append(dict(label=label, role=role, parameters=candidate,
            changed_parameters=changed, initial_states=entries, fixed_recovery_order=1,
            forward_evaluated=False, only_recovery_order_changed_from_selected_q2=not bool(changed)))
    result = dict(status='final_predeclared_q1_controls_no_forward_evaluation',
        parent_selected_request_id=26, parent_result_sha256=sha(PARENT), fixed_recovery_order=1,
        initial_states=entries, allowed_changed_parameters=['a_max_s', 'kappa_positive_per_v2'],
        input_sha256=sources, candidates=candidates,
        capacity_seed_derivation=dict(a_seed_s=a_seed, verified_necessary_A_lower_s=lower,
            assumed_fraction_of_invariant_a_ceiling=fraction, fraction_is_measured=False,
            observed_bare_SET_R_median_ohm=read['original_observed_R_median_ohm'],
            soft_current_at_qcap_a=read['soft_at_qcap_a'], native_read_voltage_v=read['read_voltage_v'],
            fcap=rectangle['fcap'], negative_boundary_upper_a=negative_bound,
            negative_boundary_lower_rhs_a=rectangle['negative_boundary_rhs_lower_a'],
            new_capacity_joint_rectangle_verified=True, read_current_ceiling_a=read_ceiling,
            read_ceiling_below_1mA=True, attainable_state_claim=False),
        q1_kappa_reference=reference, ODEs=0, new_readouts=0, inverses=0,
        fits=0, default_model_replaced=False, source_inputs_unchanged=True)
    assert all(sha(path) == digest for path, digest in sources.items())
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(declaration_sha256=sha(destination), a_seed_s=a_seed,
        kappa_values=[row['parameters']['kappa_positive_per_v2'] for row in candidates],
        negative_boundary_margin_a=rectangle['negative_boundary_rhs_lower_a']-negative_bound,
        new_model_evaluations=0)))


if __name__ == '__main__':
    main()
