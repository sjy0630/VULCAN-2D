"""Exact joint permutation probability when every qualifying subset is unique.

For each feature/cut/sign, sorted values give the extremal subset sum. The
second extremum certifies whether only one subset can reach the threshold.
When so, inclusion-exclusion over compatible prefix-set events is exact.
Otherwise this routine reports unresolved rather than claiming exactness.
"""
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from check_cycle_order_evidence import cusum, ROOT


def certify(values, threshold=None):
    x = np.asarray(values, float); n, m = x.shape; centered = x-x.mean(axis=0); sd = x.std(axis=0, ddof=1)
    threshold = float(cusum(x).max()) if threshold is None else threshold
    records = []; events = set(); unresolved = []
    for j in range(m):
        if sd[j] == 0:
            if threshold <= 0: unresolved.append({'feature': j, 'reason': 'constant feature reaches threshold'})
            continue
        for sign in (-1, 1):
            order = np.argsort(sign*centered[:, j])[::-1]; sorted_values = sign*centered[order, j]
            for k in range(1, n):
                scale = sd[j]*np.sqrt(k*(n-k)/n)
                maximum = sorted_values[:k].sum()/scale
                second = (sorted_values[:k].sum()-sorted_values[k-1]+sorted_values[k])/scale
                row = {'feature': j, 'cut_after': k, 'sign': sign, 'maximum_possible_statistic': float(maximum),
                       'second_largest_subset_statistic': float(second), 'can_reach_observed': bool(maximum >= threshold-1e-12)}
                records.append(row)
                if row['can_reach_observed']:
                    if second >= threshold-1e-12: unresolved.append(row)
                    else: events.add((k, tuple(sorted(order[:k].tolist()))))
    if unresolved or len(events) > 16:
        return {'certified': False, 'threshold': threshold, 'unresolved': unresolved, 'events': len(events),
                'exact_probability': None, 'extrema': records}
    events = sorted(events); probability = Fraction(0, 1)
    # Prefix sets at different cuts must be nested. Their block permutations
    # have count product(block_size!); incompatible events have zero intersection.
    for size in range(1, len(events)+1):
        for group in combinations(events, size):
            group = sorted(group); previous_k = 0; previous = set(); ways = 1; compatible = True
            for k, items in group:
                selected = set(items)
                if not previous <= selected: compatible = False; break
                ways *= math.factorial(k-previous_k); previous_k, previous = k, selected
            if compatible:
                ways *= math.factorial(n-previous_k)
                probability += (-1 if size % 2 == 0 else 1)*Fraction(ways, math.factorial(n))
    return {'certified': True, 'threshold': threshold, 'exact_probability': float(probability),
            'probability_numerator': probability.numerator, 'probability_denominator': probability.denominator,
            'qualifying_prefix_set_events': [{'cut_after': k, 'cycle_ids_in_prefix': [i+1 for i in items]} for k, items in events],
            'extrema': records}


def run():
    source = ROOT/'analysis/cycle_order_evidence.json'; data = json.loads(source.read_text()); records = {}
    for name, row in data['analyses'].items():
        records[name] = certify(row['features']); assert records[name]['certified']
        print(name, records[name]['exact_probability'], len(records[name]['qualifying_prefix_set_events']), flush=True)
    files = [Path(__file__), source, ROOT/'analysis/check_cycle_order_evidence.py', ROOT/'tests/test_cycle_order_certificate.py']
    result = {'scope': 'Exact probability for the specified exploratory joint feature/cut statistic under exchangeable cycle order',
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'analyses': records}
    (ROOT/'analysis/cycle_order_exact_certificate.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': run()
