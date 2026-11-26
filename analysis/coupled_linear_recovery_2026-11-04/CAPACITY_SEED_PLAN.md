# Final finite q1 controls after the capacity exclusion

The one-readout capacity audit completed in root session 21194, exit 0. With
the final q2 A, the q1 joint invariant rectangle implies bare SET return
R >= 1034.93 ohm for any positive kappa, failing the original median upper
limit 187.68 ohm. The verified necessary A lower bound is 0.00736010 S,
under the recorded fixed J, b, rho, lambda, readout and bare protocol.
The three earlier kappa-only proposals were never simulated.

Do not use those excluded proposals as prospective improved candidates.
Retain one same-numeric-parameters q1 case only as an explicitly labelled
order-only control. It is already capacity-excluded, and any numerical
table-domain failure must not be interpreted as physical family rejection.

For exactly three prospective candidates, choose one common capacity seed:

    A_seed = (1 / R_observed_median - Fsoft(vread,qcap) / vread)
             / (0.9 * fcap).

Use the saved original 14-record bare SET return resistance median, saved
native vread, and the capacity audit's one evaluated positive readout.
The factor 0.9 is an explicit initialization assumption: 90% of the invariant
a ceiling, with the optimistic q=qcap soft contribution. It is not a measured
occupancy, inferred trajectory, fitted fraction or guarantee that this state
will be reached. Do not infer q1 RESET entrances from q2 states.

Before declaring candidates, check A_seed is finite and above the verified
necessary lower bound, and recheck the joint rectangle's negative boundary
using Mnegative + A_seed * fcap * Vreset_max <= Icc * exp(4b). Check also the
read-current ceiling remains below Icc. These are saved-value arithmetic;
do not perform more table readouts, inverses, roots or ODEs here.

The three common kappa values remain [1,3,10] times 0.05765319667504992 V^-2,
from the existing explicitly conditional loss-reference derivation. All
three use A_seed; all other material parameters are the final q2 selection,
and all three original segment entries remain unchanged. Scaling kappa
changes capture and recovery together in both polarities. These comparisons
change recovery order and two numerical coefficients, not recovery order
alone; they are not a full q1 optimization or rejection test for the family.

`declare_forward_controls.py` performs only the above arithmetic, writes
the four complete candidates and input hashes to FORWARD_DECLARATION.json,
and refuses to overwrite an existing declaration. Root and independent
review of that declaration, then source freeze, precede the unique run of
run_rate_controls.py. Its existing six-evaluation/552-phase/600+300-second
limits, complete original task, failure retention and final two-precision
verification remain unchanged. Original holdouts are not accessed.
