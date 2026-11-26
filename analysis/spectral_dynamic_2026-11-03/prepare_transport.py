"""Build and check a numerical table of the fixed shared spectral readout."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vulcan2d.shared_spectral import SpectralShape
from vulcan2d.shared_spectral_ivp import SpectralTable

HERE = Path(__file__).resolve().parent
TABLE = HERE/'transport_g010_v0025.npz'
RESULT = HERE/'transport_g010_v0025_validation.json'


def main():
    shape = SpectralShape.from_json(ROOT/'analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json')
    started = time.monotonic()
    table = SpectralTable.build(shape, progress=lambda x: print(json.dumps(x), flush=True))
    table.save(TABLE)
    rng = np.random.default_rng(43182)
    gammas = np.r_[rng.uniform(-23.9, 5.9, 500), [-18., -10., -6., -3., -.1, 0., 1., 3., 5.9]]*np.log(10.)
    volts = 10**rng.uniform(-8.9, np.log10(5.), len(gammas))
    signs = rng.choice([-1., 1.], len(gammas))
    exact = np.abs(shape.current_at_log_gamma(signs*volts, gammas, order=64))
    fast = np.array([np.exp(np.interp(np.log(v), table.log_voltage, table.profile(g, sign)))
                     for v, g, sign in zip(volts, gammas, signs)])
    errors = np.log10(fast/exact)
    j = {'status': 'table_built_and_pointwise_interpolation_audited_not_full_trajectory_certification',
         'shape': asdict(shape), 'metadata': table.metadata, 'n_validation_points': len(exact),
         'max_error_decade': float(max(abs(errors))), 'rms_error_decade': float(np.sqrt(np.mean(errors**2))),
         'voltage_V': (signs*volts).tolist(), 'ln_gamma': gammas.tolist(),
         'exact_current_magnitude_A': exact.tolist(), 'table_current_magnitude_A': fast.tolist(),
         'error_decade': errors.tolist(), 'elapsed_s': time.monotonic()-started,
         'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'vulcan2d/shared_spectral.py',
                                     ROOT/'vulcan2d/shared_spectral_ivp.py', TABLE]}}
    RESULT.write_text(json.dumps(j, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in j.items() if k not in ('voltage_V', 'ln_gamma', 'exact_current_magnitude_A',
                    'table_current_magnitude_A', 'error_decade', 'source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
