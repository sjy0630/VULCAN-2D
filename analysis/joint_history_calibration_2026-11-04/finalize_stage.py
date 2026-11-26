"""Close this diagnostic's evidence manifest after all writers stop; no model run."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / 'JOINT_HISTORY_CALIBRATION_RESULTS_2026-11-04.md'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((HERE / 'bounded_joint_calibration.json').read_text())
    review = json.loads((HERE / 'ACTUAL_RESULT_REVIEW.json').read_text())
    diagnosis = json.loads((HERE / 'SAVED_RESULT_DIAGNOSTICS.json').read_text())
    assert result['status'] == 'completed_bounded_joint_diagnostic'
    assert review['status'].startswith('passed') and not review['failures']
    assert review['terminal_result_sha256'] == digest(HERE / 'bounded_joint_calibration.json')
    assert review['objective_requests'] == 85 and review['selected_request'] == 86
    assert review['unique_saved_phases_audited'] == 265
    assert review['independent_saved_native_points'] == diagnosis['independent_native_points'] == 3710
    assert result['final_precision_comparison']['passed']
    assert not result['full_unified_model_achieved']
    assert not result['continuation_used_for_selection']
    assert all(not r['scipy_success'] for r in result['optimizers'])
    assert review['source_provenance_qualification']['normal_numerical_path_unchanged']
    assert not review['source_provenance_qualification']['entire_startup_source_frozen_claim']
    assert (HERE / 'figures/fig_continuous_joint_calibration.png').exists()
    assert (HERE / 'figures/fig_continuous_joint_calibration.pdf').exists()
    root = {
        'status': 'passed_limited_diagnostic_with_explicit_source_timing_qualification',
        'executed_session': 10395, 'exit_code': 0,
        'actual_numerical_run_completed': True, 'model_rerun_for_audit': False,
        'actual_result_review_sha256': digest(HERE / 'ACTUAL_RESULT_REVIEW.json'),
        'independent_checks': review['checks_completed'],
        'saved_phase_count': 265, 'independent_native_points': 3710,
        'source_timing_qualification': review['source_provenance_qualification'],
        'same_precision_control_to_refined_scores': diagnosis['aggregate_calibration_scores'],
        'selected_candidate_is_not_uniform_phase_improvement': True,
        'both_optimizers_reached_evaluation_cap_without_convergence': True,
        'search_only_request_57_has_not_received_final_independent_validation': True,
        'root_visually_inspected_figure': {
            'path': str(HERE / 'figures/fig_continuous_joint_calibration.png'),
            'passed': True,
            'checks': ['All five I-V panels readable with original scored observations.',
                       'Calibration/continuation roles, old/new curves and branches clear.',
                       'Low-current tails retained and shared-state history shown.',
                       'No overlapping titles, labels, legends or footer text.',
                       'Full unified 1R/1T1R validation explicitly outstanding.']},
        'strict_frozen_table_domain_scope': 'Original saved phase_integrity uses actual kernel.lg; independent audit uses a broader Gamma-domain check, not an independent old-table boundary reconstruction.',
        'original_holdout_accessed': False, 'new_default_model_selected': False,
        'push_performed': False, 'full_unified_model_achieved': False,
        'goal_status': 'active', 'next_options_executed': False,
    }
    (HERE / 'ROOT_VALIDATION.json').write_text(json.dumps(root, indent=2, ensure_ascii=False)+'\n')
    # The independent reviewer just reread every catalog byte. Reuse those
    # verified hashes rather than hydrate hundreds of artifacts a second time.
    known = {item['path']: {'path': item['path'], 'sha256': item['sha256'],
             'verification': 'reread_by_independent_saved_result_audit'}
             for item in review['artifact_hash_checks']}
    assert len(known) == 622 and all(item['matched'] for item in review['artifact_hash_checks'])
    paths = [p for p in HERE.rglob('*') if p.is_file()
             and p.name not in ('completed_stage_manifest.json',)
             and '__pycache__' not in p.parts]
    paths += [REPORT]
    additional = [p for p in paths if str(p) not in known]
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(digest, additional))
    for p, h in zip(additional, hashes):
        known[str(p)] = {'path': str(p), 'sha256': h, 'verification': 'read_at_stage_close'}
    manifest = {
        'status': 'completed_limited_diagnostic_evidence_package',
        'full_scientific_goal_complete': False, 'files': sorted(known.values(), key=lambda r:r['path']),
        'file_count': len(known), 'reused_independently_verified_artifact_hashes': 622,
        'newly_hashed_files': len(additional),
        'manifest_self_excluded': True,
        'mutable_cross_stage_project_guidance_not_frozen': str(HERE.parents[1] / 'CLAUDE.md'),
        'authoritative_result': str(HERE / 'bounded_joint_calibration.json'),
        'progress_snapshot_is_not_terminal_result': True,
        'source_timing_qualification_preserved': True,
    }
    (HERE / 'completed_stage_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'root_validation': root['status'], 'manifest_files': len(known),
                     'goal_status': 'active', 'full_unified_model_achieved': False}))


if __name__ == '__main__':
    main()
