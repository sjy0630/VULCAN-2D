"""Conditional zero-bias retention implications, not a matched retention test."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vulcan2d.stochastic_bottleneck import kinetic_parameters
from vulcan2d.bidirectional_state import log_rates


def run():
    source = ROOT / 'analysis/stochastic_reentry_probe.json'; report = json.loads(source.read_text())
    sets = [(name, row['fit']['parameters']) for name, row in report['candidates'].items()]
    sets += [('polarizable_saddle_peak_'+str(int(row['fixed_log_peak_rate_per_V'])), row['parameters'])
             for row in report['polarizable_rate_peak_profiles']]
    rows = []
    for tag, parameters in sets:
        name = 'polarizable_saddle' if tag.startswith('polarizable_saddle') else tag
        a, b = log_rates(np.array([0.]), kinetic_parameters(parameters, name), 'polarizable_saddle')
        a, b = float(np.exp(a[0])), float(np.exp(b[0])); stationary = a/(a+b)
        if stationary >= .95:
            maximum_rate = minimum_scan = None
        else:
            maximum_rate = -np.log((.95-stationary)/(1-stationary))/((a+b)*7*3600)
            minimum_scan = 3./maximum_rate
        rows.append({'candidate': tag, 'zero_build_per_V': a, 'zero_remove_per_V': b,
                     'zero_stationary_high_probability': stationary,
                     'hypothetical_retention_seconds': 7*3600, 'hypothetical_retained_high_probability': .95,
                     'conditional_maximum_sweep_rate_V_per_s': maximum_rate,
                     'conditional_minimum_duration_of_3V_path_days': minimum_scan/86400 if minimum_scan is not None else None})
        print(tag, 'zero loss per V', b, 'minimum 3V scan days', rows[-1]['conditional_minimum_duration_of_3V_path_days'])
    paths = [Path(__file__), source, ROOT / 'vulcan2d/stochastic_bottleneck.py', ROOT / 'vulcan2d/bidirectional_state.py']
    result = {'scope': 'Counterfactual mapping of voltage-path rates to absolute time for the SAME model state at zero bias',
              'formula': 'P_on(t)=p_eq+(1-p_eq)*exp[-(k_build_perV+k_remove_perV)*sweep_rate*t]',
              'rows': rows,
              'limitations': ['No scan speed was measured or supplied; no absolute lifetime is measured by this calculation.',
                              'The Nature retention result uses +0.1 V, other programmed states and potentially another cell; this is not a direct contradiction.',
                              'This test addresses an interpretation as the same nonvolatile structural configuration. A fast contact/charge readout variable need not itself have nonvolatile retention.',
                              'A proposed unified law must supply the missing state/bias/timing connection instead of silently extrapolating the fit.'],
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT / 'analysis/stochastic_retention_diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    run()
