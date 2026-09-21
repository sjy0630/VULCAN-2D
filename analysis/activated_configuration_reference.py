"""Exact phase-template error floor and training-only chronological reference.

This flexible data template is a diagnostic benchmark, not a physical model.
Its all-data optimum bounds any exactly repeating deterministic mean waveform
on this common applied-voltage protocol; it does not bound history dependence.
"""
import json
from pathlib import Path
import numpy as np
from compare_activated_configuration import assemble, scores


def run():
    d = assemble(); n = 53; points = len(d['voltage'])//n
    voltage = d['voltage'].reshape(n, points)
    np.testing.assert_allclose(voltage, np.broadcast_to(voltage[0], voltage.shape), rtol=0, atol=1e-12)
    y = d['observed'].reshape(n, points); w = d['weights'].reshape(n, points)
    logy = np.log10(np.maximum(y, np.finfo(float).tiny))
    def fit(train):
        total = w[train].sum(axis=0)
        template = np.divide((w[train]*logy[train]).sum(axis=0), total,
                             out=np.zeros(points), where=total > 0)
        return np.tile(10**template, n)
    all_pred = fit(np.arange(n)); chrono = fit(np.arange(35))
    result = {'scope': 'Exact weighted log-current least-squares template; phase-resolved and exactly repeating each pair',
              'no_history_class_only': True, 'identical_native_protocol_max_error_V': float(abs(voltage-voltage[0]).max()),
              'all_descriptive_exact_floor': scores(d, all_pred),
              'first35_template_training': scores(d, chrono, np.arange(1, 36)),
              'first35_template_target_last18': scores(d, chrono, np.arange(36, 54))}
    path = Path(__file__).with_suffix('.json'); path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v if not isinstance(v, dict) else v['rmse_decade'] for k, v in result.items()}, indent=2))


if __name__ == '__main__': run()
