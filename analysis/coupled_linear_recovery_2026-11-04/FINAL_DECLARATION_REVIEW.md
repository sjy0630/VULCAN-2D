# Independent review of the final finite q¹ declaration

Status: **passed for the stated finite forward-control purpose; no blocking
change requested**. This review did not run the declaration script, load or
evaluate the transport table, integrate a trajectory, solve a root, perform
an inverse or fit a model. It used source inspection, saved-value arithmetic
and SHA-256 comparisons only. The author-owned files were not modified.

## Reviewed artifacts

| Artifact | SHA-256 |
|---|---|
| `CAPACITY_SEED_PLAN.md` | `660487a007fcac1e126449b530f9c1a49d2435d8437232a794f15b0b67796dcd` |
| `declare_forward_controls.py` | `71dc065a4f98d1cc040108e61c2f154ffae8c20c76430ebf0b8cfa2049c5619e` |
| `FORWARD_DECLARATION.json` | `a0e4f9bf6ff99b2bc06f4064c86e3ee9a299bd5136c45a5eb27e1cef2c91f956` |
| `CAPACITY_CHECK.json` | `689e34480c5187a39685b33d8f5982b056913e5c982ec6b3206bf5c56f135b22` |
| `SEED_DERIVATION.json` | `147ac6f439bddd3c416c4a5d17c9e868fa8cd8a2c4fd80234a2202733c2b9f98` |

Every current file listed in the declaration's `input_sha256` mapping
matches its recorded digest. The final declaration records zero ODEs,
new readouts, inverses and fits, consistently with the inspected script.

## Capacity arithmetic and invariant rectangle

An independent recomputation from the saved quantities gives

- qcap = 0.21050001227684284;
- fcap = 0.7222498212074407;
- original eligible observed resistance median = 91.0853274739201 Ω;
- original median gate upper limit = 187.67999555827214 Ω;
- native read command = 0.0500000007450581 V;
- frozen positive soft current at qcap = 6.195017703301536e−7 A;
- Aseed = **0.016870626999145072 S**.

The Aseed calculation reproduces the saved value exactly in ordinary
floating point. Its numerator is a conductance difference in siemens;
both 0.9 and fcap are dimensionless, so Aseed has the required units S.
The shared κ remains in V⁻² under the command-voltage path clock, not s⁻¹.

The conditional rectangle q ≤ qcap, a ≤ Aseed fcap has negative-boundary
upper estimate 0.07332876406694991 A against the lower right-hand-side
bound 0.10746761150882182 A. The positive margin is
**0.034138847441871914 A**. This verifies the same sufficient negative
boundary inequality for the new A. Positive q is bounded by bare compliance;
both a upper boundaries point inward because b ≥ 0 and the positive
equilibrium fraction is the larger one. The two bare segment entries are
inside the rectangle. This proof retains the original finite table domain,
original bare protocol and fixed J, λ, b, ρ and readout.

The full read-current ceiling is 0.000609859877538217 A, below the original
0.001 A compliance. Consequently the read-point capacity calculation uses
the command voltage directly without needing an extra assumption about
voltage monotonicity. Aseed exceeds the previously verified necessary
A lower bound 0.007360096435622046 S. This removes that particular capacity
exclusion; it does not establish reachability or satisfaction of the gate.

## Meaning of the 90% construction

The construction inserts a hypothetical read state q = qcap,
a = 0.9 Aseed fcap **only into the seed algebra**. The resulting current is
0.0005489358399614284 A, exactly the saved native read voltage divided by
the original observed median resistance. The 0.9 factor is a chosen
parameter-seed assumption about proximity to the a ceiling. It is not an
observed occupation, fitted fraction, inferred q¹ state or a new simulator
initial condition. The files state that no attainable-state claim is made.
The optimistic soft contribution also assumes q = qcap; an actual
trajectory can remain below either assumed value.

## Candidates and initial-state audit

Exactly four candidates are declared:

| Role | A / S | κ / V⁻² | Changed numerical parameters from selected q² |
|---|---:|---:|---|
| Explicit order-only control | 0.0013206747378601963 | 57.67798793605963 | none |
| Prospective ×1 seed | 0.016870626999145072 | 0.05765319667504992 | A and κ |
| Prospective ×3 seed | 0.016870626999145072 | 0.17295959002514977 | A and κ |
| Prospective ×10 seed | 0.016870626999145072 | 0.5765319667504992 | A and κ |

All four explicitly set fixed recovery order to 1. The order-only control
is already excluded from the bare low-resistance gate and is labelled as
such; it is not offered as a prospective improved fit. Each prospective
candidate changes only A and κ numerically, and its metadata correctly
denies that it is an order-only comparison. All other seven-parameter
entries match the selected q² values. Scaling κ changes capture and
recovery in both polarities, as the plan states.

All candidate initial-state dictionaries exactly equal the parent result
and seed derivation:

    bare1   = [4.298219923945171e−05, 0.0]
    bare5   = [0.00025057076016083026, 0.0]
    series4 = [7.755279151592979e−05, 0.0]

No candidate installs q² RESET entrance states as q¹ boundaries. The old
q² entrances are used only in the previously declared conditional κ scale
reference. The 90% fraction does not alter any of the three segment entries.
Every candidate remains marked `forward_evaluated=false` in the declaration.

## Scope of this approval

The declaration is internally consistent with its finite comparison plan.
Source freeze and the separate driver review remain the root agent's
responsibility before its unique forward run. This review does not claim
that a declared candidate will complete, improve error, satisfy the 20
gates, describe physical-time kinetics or establish a microscopic mechanism.
A trajectory or domain failure must be retained and cannot reject the full
q¹ family. Likewise, success of a seed would require the planned complete
training evaluation and independent precision checks before interpretation.
