"""Close this limited diagnostic with explicit targeted-review provenance."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent/'CONSTRAINED_HISTORY_CONVERGENCE_RESULTS_2026-11-04.md'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = json.loads((HERE/'bounded_coordinate_check.json').read_text())
    review = json.loads((HERE/'ACTUAL_RESULT_REVIEW.json').read_text())
    freeze = json.loads((HERE/'SOURCE_FREEZE.json').read_text())
    diagnostic = json.loads((HERE/'SAVED_RESULT_DIAGNOSTICS.json').read_text())
    integral = json.loads((HERE/'integral_selfcheck/SELF_CHECK_RESULT.json').read_text())
    assert result['status']=='completed_finite_coordinate_diagnostic'
    assert result['numerical_validation_passed'] and result['all_three_goals_passed']
    assert result['search_outcome']['status']=='request_budget_exhausted'
    assert result['search_outcome']['grid_stopped'] is False
    assert result['search_requests']==96 and result['selected_incumbent']['request_id']==93
    assert review['status']=='passed_targeted_saved_result_review' and review['all_checks_passed']
    assert len(review['checks'])==1786
    assert review['coverage']['unique_phase_NPZ_decoded']==45
    assert result['source_files_unchanged']
    for path, expected in freeze['source_sha256'].items():
        assert result['source_sha256'][path]==expected and digest(path)==expected
    reread=review['actual_files_reread_and_hash_checked']
    assert reread[str(HERE/'bounded_coordinate_check.json')]['sha256']==digest(HERE/'bounded_coordinate_check.json')
    assert integral['status']=='completed_known_model_discretization_selfcheck'
    assert all(integral[key]==0 for key in ('new_model_function_calls','new_roots','new_inversions','new_ODEs','new_optimizations'))
    assert not integral['experimental_mechanism_test']
    root={
        'status':'passed_limited_three_phase_diagnostic_not_optimization_convergence',
        'model_session':47591,'model_exit_code':0,'saved_arithmetic_session':32356,'arithmetic_exit_code':0,
        'source_freeze_before_model_launch_verified':True,'source_files_unchanged':True,
        'three_phase_RMSE_goals_both_precisions_passed':True,
        'phase_goal_is_not_noise_interval':True,'grid_stopped':False,
        'continuous_stationarity_or_global_optimum_proven':False,
        'targeted_review_checks':1786,'targeted_review_coverage':review['coverage'],
        'targeted_review_sha256':digest(HERE/'ACTUAL_RESULT_REVIEW.json'),
        'all1170_RMSE_comparison':diagnostic['aggregate_scores']['all1170_RMSE_decade'],
        'critical_features_remain_inaccurate':True,
        'root_visual_inspection':{'path':str(HERE/'figures/fig_calibration_iv_old_vs_final.png'),
            'passed':True,'scope':'Three original calibration phases and all scored points; old/final distinction, low-current tails, units and budget limitation readable.'},
        'root_integral_selfcheck_review':{'source_and_result_inspected':True,
            'known_model_sampling_bias_demonstrated':True,
            'experimental_mechanism_test_performed':False,
            'physical_interpolation_error_certificate':False},
        'full_original_contract_evaluated':False,'full_unified_1R_1T1R_model_achieved':False,
        'original_holdout_accessed':False,'default_model_changed':False,'push_performed':False,
        'goal_status':'active',
    }
    (HERE/'ROOT_VALIDATION.json').write_text(json.dumps(root,indent=2)+'\n')
    # Preserve the distinction between creation-time records and targeted
    # byte rechecks. Do not claim a full reread of every search artifact.
    known={}
    for path,record in result['artifact_catalog'].items():
        checked=path in reread
        if checked: assert reread[path]['sha256']==record['sha256']
        known[path]={'path':path,'sha256':record['sha256'],
            'verification':'reread_by_targeted_review' if checked else 'recorded_at_write_time_not_individually_reread'}
    assert len(known)==626
    paths=[p for p in HERE.rglob('*') if p.is_file() and p.name!='completed_stage_manifest.json'
           and '__pycache__' not in p.parts]+[REPORT]
    new=[p for p in paths if str(p) not in known]
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes=list(pool.map(digest,new))
    for p,h in zip(new,hashes):known[str(p)]={'path':str(p),'sha256':h,'verification':'read_at_stage_close'}
    manifest={'status':'completed_limited_diagnostic_evidence_package','file_count':len(known),
        'files':sorted(known.values(),key=lambda r:r['path']),
        'runtime_catalog_file_count':626,'catalog_is_not_a_full_current_file_rehash':True,
        'targeted_review_coverage':review['coverage'],'newly_hashed_files':len(new),
        'manifest_self_excluded':True,'progress_snapshot_is_not_authoritative_result':True,
        'mutable_project_guidance_not_frozen':str(HERE.parents[1]/'CLAUDE.md'),
        'full_scientific_goal_complete':False}
    (HERE/'completed_stage_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'root_validation':root['status'],'manifest_files':len(known),'goal_status':'active'}))


if __name__=='__main__':main()
