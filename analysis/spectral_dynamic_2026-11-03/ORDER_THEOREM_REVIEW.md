# Independent review: same-state SET residence/crossing comparison

Date: 2026-11-03. Reviewer scope: `finite_rate_order_audit.py` section 3 and
its generated report. No core, source-data, optimizer, or other-agent file edits.
This is a conditional mathematical audit, not a claim of measured scan rates.

## Verdict

The inequality direction is correct for the current shared spectral SET law.
A soft trajectory need only remain in the state band; it does not have to
traverse the entire band. A bare trajectory that crosses the entire band at
uniformly lower actual material voltage supplies the comparison time.

The general theorem needs a positive-rate or uniqueness qualification missing
from the shortest statement. Nonnegative rate and voltage monotonicity alone
are insufficient if a non-Lipschitz zero-rate point permits arbitrary waiting.
The current positive-part logarithmic rate is locally Lipschitz on the relevant
finite state band, so that pathology does not invalidate this model audit.

## Precise sufficient assumptions and proof

Write the common physical-time law on the positive branch as

`du/dt = -h(V_h,u)`, with `h >= 0`.

Let the common state band be `[a,b]`, `a < b`, and suppose:

1. The same scalar deterministic material law and parameters apply to both
   trajectories. For fixed `u`, `h(V_h,u)` is nondecreasing in positive actual
   material voltage, and the initial-value problem is unique (for example,
   local Lipschitz continuity in `u`, uniformly over the relevant voltages).
2. Throughout the observed soft segment, `u_s(t)` belongs to `[a,b]` and
   `V_s(t) >= V_*`.
3. The bare trajectory passes continuously from above `b` to below `a` in a
   finite observed time bracket, and throughout its crossing
   `0 <= V_b(t) <= V_b,max < V_*`.
4. There is no extra hidden state, rate rescaling, reset, jump, or unrecorded
   local-voltage transient invalidating those premises.

Set `g(u)=h(V_*,u)`. A zero of `g` within `[a,b]` is also a zero of every
nonnegative bare rate at voltage `<=V_b,max`. With uniqueness it is an
uncrossable equilibrium barrier. Thus the finite crossing from outside the
band implies `g(u)>0` throughout this compact band. Equivalently, one may
assume this positivity and finiteness of the integral directly.

Define `Phi(u)=integral_a^u dz/g(z)`. For the soft segment,

`dPhi(u_s)/dt = -h(V_s,u_s)/g(u_s) <= -1`,

hence

`T_soft <= Phi(u_s,start)-Phi(u_s,end) <= Phi(b)-Phi(a)`.

For the bare crossing from `b` to `a`,

`dPhi(u_b)/dt = -h(V_b,u_b)/g(u_b) >= -1`,

hence

`Phi(b)-Phi(a) <= T_bare,cross <= T_bare,bracket`.

Combining gives the claimed result. The middle inequality permits pauses in
the bare trajectory. The soft path may cover only a proper subinterval.
Equality is not generally excluded when voltage monotonicity is non-strict;
using a non-strict final bound is safe.

For constant commanded-voltage scan speeds in these two segments,
`T_soft=DeltaV_soft/r_series` and
`T_bare,bracket=DeltaV_bare/r_bare`. Therefore

`alpha = r_series/r_bare >= DeltaV_soft/DeltaV_bare`.

A 1 V soft segment and nominal 0.05 V bare bracket imply `alpha >= about 20`.
The numerical bracket values in the saved audit give individual bounds
19.9999237063 to 20.0000190735; precision beyond about 20 is a stored-grid
artifact. A common clock must satisfy every applicable bracket. If the model
uses one shared command-path rate, it implicitly sets alpha=1 and violates
this necessary condition whenever the stated trajectory/data premises hold.

## Why regularity cannot be omitted

Let `[a,b]=[0,2]` and `h(V,u)=sqrt(abs(u-1))`, independent of voltage. This is
nonnegative and voltage-monotone. A bare solution can travel from 2 to 1 in
2 time units and from 1 to 0 in another 2, while a soft solution can stay at
`u=1` for an arbitrarily long time. Both solve the same scalar ODE. The
residence inequality then fails, despite the soft voltage being higher.
The failure is nonuniqueness at the non-Lipschitz zero, not a reversed
integral comparison. This counterexample does not apply to the current core.

## Mapping the theorem to the saved audit

### State band and endpoint quantifiers

The reported gamma interval `[0.0024282213,0.0027003124] eV` is the **interval
hull** of the 5 V and 4 V endpoint intervals, rather than their set union.
Gamma increases as `u` decreases, so a monotone SET trajectory with both
endpoints in the hull remains there for the entire segment.

For the unexpanded empirical endpoint intervals,
`gamma_5V,min=0.0025577842 > gamma_4V,max=0.0025004827 eV`.
No monotone-closing trajectory satisfies both intervals exactly. Consequently
the original endpoint-constrained family is already empty; the order bound
must not be advertised as an independently non-vacuous impossibility proof
for that family. The enlarged endpoint-band sensitivity being prepared by
the design agent should explicitly check that `gamma_5V <= gamma_4V` is
possible before interpreting the timing bound.

The Q10–Q90 intervals summarize different cycles. They are not confidence
intervals, universal per-cycle envelopes, or automatically the full-fit
acceptance gates. The theorem concerns any particular model trajectory that
satisfies the declared endpoint intervals. It does not establish that all
32 observed series trajectories do so or that an existing RMSE gate logically
imposes these endpoints. Different endpoint cycles cannot be spliced into one
physical trajectory to assert a measured residence time.

### Actual voltage and ideal compliance

For the soft branch, passive monotone material and transistor I–V relations
make the material voltage increase with command voltage and decrease with
gamma at fixed command. Thus the minimum on command `[4,5] V` and the gamma
hull is the KCL solution at `command=4 V, gamma=gamma_max`; the saved value is
`3.1429477176 V`. This uses the fixed measured-transistor mapping and specified
gate voltage; unmeasured calibration changes or capacitive transients are not
covered by that bound.

For bare records the after-crossing implication is conservative even at
compliance: if `V_h <= V_command` and measured material current is at least
`F(V_command,gamma_max)`, monotonicity implies `gamma >= gamma_max`.

The before-crossing implication is stronger: `I < F(V_command,gamma_min)`
does **not** imply `gamma < gamma_min` from passivity alone. It requires
`V_h=V_command` (or a verified lower bound close enough to it) at that sample.
The saved before currents range from about 0.022 to 1.94 microampere, far
below 1 milliampere compliance; the current ideal bare circuit is therefore
unlimited there and does satisfy `V_h=V_command`. Unknown series drops,
instrument settling, or a voltage column representing a different quantity
would require a new bracket calculation. The after currents are approximately
1 milliampere, so their command voltages should not be asserted to equal the
actual hBN voltages. Only their upper-bound role is used.

No between-sample voltage excursion above the recorded bracket ceiling is
allowed in this static argument. A capacitor/current-compliance transient is
one physical way that assumption could fail, and cannot be excluded by these
quasistatic arrays alone.

Independent numerical recomputation with the exact current core reproduces
the soft minimum as `3.1429477175978198 V`. Across all 14 saved brackets,
`F(V_before,gamma_min)/I_before >= 19.14006979` and
`I_after/F(V_after,gamma_max) >= 23.25736181`. Thus the classification margins
are substantial under the stated voltage mapping. The maximum before current
is only `0.00193724486` of the 1 mA compliance value. These are numerical
checks of the conditional bracket, not a validation of its instrument premises.

### Scan-time claim

There are no measured timestamps in the current records. Voltage-point spacing
is not evidence of relative elapsed time. The result is a necessary relative
time/scan-speed condition under constant scans, not a measurement of alpha.
For step-and-settle or nonuniform scans the direct elapsed-time inequality is
the valid statement; one must use measured segment times before converting it.

## Consequence for the proposed RESET-initiation term

`A_RESET` changes only negative-current dynamics. It changes neither the
spectral readout, the positive-current material rate, nor the voltage/state
ordering used above. It can affect the initial state of a later SET segment,
but any resulting trajectory that satisfies these same endpoint and bracket
premises remains subject to the same inequality.

Therefore optimizing `A_RESET` at shared path rates can quantify the best
compromise or repair a RESET-specific mismatch; it cannot remove this SET
necessary condition. A better scalar loss alone is not evidence that all
declared endpoint/order gates have passed. A proposed escape must identify
which premise changed and have independent support; it must not silently fit
an unknown 20-fold scan-speed ratio.
