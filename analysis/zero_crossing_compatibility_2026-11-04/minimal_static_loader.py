"""Build the unchanged frozen static classes from only used input files.

Avoid the historical full-trajectory loader and vulcan2d package initializer.
No readout, circuit, state equation or interpolation method is reimplemented.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys

import numpy as np
from signed_forward_kernel import SignedSoftKernel, _load_frozen_module, PREVIOUS, ROOT


def load_minimal():
    cache = Path('/private/tmp/vulcan-runtime-cache')
    inputs = {
        PREVIOUS/'forward_kernel.py': '052971a4ee8e927769118a3bccac323c7d5ad9a81e08d0e5caf1fe7a38181302',
        PREVIOUS/'final_refined.json': '5fa0b38f637d8a47b6681283f6e9b5c898f0662fbcfe4abd72004c2f0154ea8c',
        ROOT/'analysis/signed_soft_history_2026-11-04/signed_forward_kernel.py': '30b3df1eeb9aa22e2f8d3e8827b2950b97a38e14a4700d829edcbae6b4c57d6e',
        ROOT/'vulcan2d/transistor.py': 'fd16203554103254688c6a23e90959cec1ac9e7fab63644f207da589fb13667d',
        ROOT/'analysis/reset_shape_2026-11-03/soft_field_mobility_probe.json': '02af7e28e1a92bc742d683733fefd7f04fb4404f80c16547da4e6e0a051a15de',
        cache/'native_observations.npz': '0aedbe51f41b4f4355e1de4d26048c6790a9d454b8b7d56f51bd76a4671465a8',
        cache/'transport_g010_v0025.npz': '00d6d9ea0b1dfe855181e7016afc6dd5e23d169f83f19cfcedfc9d3b35f9e4e2',
        cache/'fig2b_transistor_lookup.json': '1c9ecb7073f74e3db9f1c4a041d17630c652b1a12e4f33ff7388aaf5f0b16f21',
        cache/'bookoutput2_reverse_transistor_lookup.json': '28462c3dcb7e0184199635de72a4c5fb1832a9f6281f08900ede28198dfe1359',
    }
    for path, expected in inputs.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    positive = _load_frozen_module(PREVIOUS/'forward_kernel.py')
    name = 'vulcan_transistor_direct_for_compatibility'
    spec = importlib.util.spec_from_file_location(name, ROOT/'vulcan2d/transistor.py')
    transistor = importlib.util.module_from_spec(spec)
    sys.modules[name] = transistor
    spec.loader.exec_module(transistor)
    mapping = transistor.measured_transistor_mapping(
        positive_lookup=transistor.TransistorLookup.from_json(cache/'fig2b_transistor_lookup.json'),
        reverse_lookup=transistor.TransistorLookup.from_json(cache/'bookoutput2_reverse_transistor_lookup.json'))
    knots = {}
    for phase in ['set', 'reset']:
        lookup = mapping._active_lookup(phase)
        x = np.unique(np.concatenate([a['drain_voltage_v'] for a in lookup.curves()]))
        knots[phase] = (x, np.array([mapping.current(float(v), 1.1, phase) for v in x]))
    with np.load(cache/'transport_g010_v0025.npz') as archive:
        lg, lv, li = (archive[key].copy() for key in ['log_gamma', 'log_voltage', 'log_current'])
        r0 = json.loads(str(archive['metadata']))['shape']['ron_ohm']
    reference = json.loads((PREVIOUS/'final_refined.json').read_text())
    assert reference['fixed_recovery_order'] == 2
    kernel = SignedSoftKernel()
    kernel.cache, kernel.frozen_module = cache, positive
    kernel.parameters = positive.Parameters(**reference['parameters'])
    readout = json.loads((ROOT/'analysis/reset_shape_2026-11-03/soft_field_mobility_probe.json').read_text())['plan']
    kernel.ron, kernel.gc = float(readout['Ron_ohm']), float(readout['gamma_c_ev'])
    assert (kernel.ron, kernel.gc) == (58.463033350725695, 0.03127031375421752)
    kernel.lg, kernel.lv, kernel.vmin = lg, lv, math.exp(lv[0])
    kernel.li_negative, (kernel.negative_tx, kernel.negative_ti) = li[1], knots['reset']
    kernel.base = positive.ForwardKernel()
    base = kernel.base
    base.ron, base.gc, base.vmin = kernel.ron, kernel.gc, kernel.vmin
    base.lg, base.lv, base.li = lg, lv, li[0]
    base.tx, base.ti = knots['set']
    base.log_scale = math.log(r0/kernel.ron)
    for array in [lg, lv, li, base.tx, base.ti, kernel.negative_tx, kernel.negative_ti]:
        array.flags.writeable = False
    kernel.source_sha256 = {str(path): digest for path, digest in inputs.items()}
    kernel.source_sha256[str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return kernel
