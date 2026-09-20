import unittest

import numpy as np

from vulcan2d import cycle_link as CL
from vulcan2d import hrs_transport as HT
from vulcan2d import model_v7 as M7
from vulcan2d import model_v8 as M8
from vulcan2d import model_v11 as M11
from vulcan2d.reset_identification import ReconstructedReset


def _synthetic_pf_cycle(beta, amplitude):
    forward = np.linspace(0.08, 1.2, 24)
    voltage = np.concatenate([forward, forward[-2::-1]])
    current = amplitude * HT.effective_pf_basis(voltage, beta)
    return ReconstructedReset(
        gate_voltage_v=1.1,
        voltage_app_v=voltage.copy(),
        current_a=current,
        voltage_ds_v=np.zeros_like(voltage),
        voltage_hbn_v=voltage,
        valid=np.ones_like(voltage, dtype=bool),
        apex_index=len(forward) - 1,
    )


class ModelV11Tests(unittest.TestCase):
    def test_effective_pf_basis_matches_the_documented_equation(self):
        voltage = np.array([0.0, 0.1, 0.4, 1.0])
        beta = 2.5
        expected = voltage * np.exp(beta * np.sqrt(voltage))

        np.testing.assert_allclose(
            HT.effective_pf_basis(voltage, beta), expected)

    def test_shared_pf_fit_profiles_out_cycle_amplitude(self):
        beta = 5.75
        cycles = [
            _synthetic_pf_cycle(beta, amplitude)
            for amplitude in (1e-10, 4e-10, 2e-9, 7e-9)
        ]

        law = HT.fit_effective_pf_law(
            cycles, gate_voltage=1.1, protocol_label="synthetic RESET")

        self.assertAlmostEqual(law.beta_per_sqrt_v, beta, places=4)
        for cycle, amplitude in zip(
                cycles, (1e-10, 4e-10, 2e-9, 7e-9)):
            self.assertAlmostEqual(
                HT.estimate_hrs_amplitude(cycle, law),
                amplitude,
                delta=amplitude * 1e-6,
            )

    def test_leave_one_out_transport_excludes_the_target_curve(self):
        beta = 6.0
        cycles = [
            _synthetic_pf_cycle(beta, amplitude)
            for amplitude in (1e-10, 2e-10, 3e-10, 4e-10)
        ]
        changed = list(cycles)
        changed[0] = _synthetic_pf_cycle(11.0, 9e-7)

        original_law = HT.leave_one_out_pf_laws(
            cycles, 1.1, "synthetic RESET")[0]
        changed_law = HT.leave_one_out_pf_laws(
            changed, 1.1, "synthetic RESET")[0]

        self.assertAlmostEqual(
            original_law.beta_per_sqrt_v,
            changed_law.beta_per_sqrt_v,
            places=10,
        )

    def test_legacy_sinh_transport_remains_the_default(self):
        p = M7.Params()
        voltage = 0.35

        self.assertAlmostEqual(
            M7.reset_hrs_basis(voltage, p),
            np.sinh(p.reset_alpha_hrs_identified * voltage),
        )

    def test_pf_parameter_object_selects_the_new_hrs_branch(self):
        law = HT.EffectivePooleFrenkelLaw(
            beta_per_sqrt_v=6.2,
            residual_mad_decade=0.05,
            gate_voltage_v=1.1,
            protocol_label="synthetic RESET",
            n_training_cycles=10,
        )
        p = M11.params_with_transport_law(M8.Params(), law)
        voltage = 0.4

        self.assertEqual(
            p.reset_hrs_transport_model, "poole_frenkel_effective")
        self.assertAlmostEqual(p.reset_pf_beta_per_sqrt_v, 6.2)
        self.assertAlmostEqual(
            M7.reset_hrs_basis(voltage, p),
            voltage * np.exp(6.2 * np.sqrt(voltage)),
        )

    def test_refined_cycle_requires_no_target_reset_current(self):
        transfer = CL.SetResetTransferLaw(
            log10_exposure_ref=0.0,
            log10_g_ref=-7.0,
            slope=0.0,
            residual_sigma_decade=0.2,
            n_training=10,
        )
        endpoint = CL.ResetEndpointAttractorLaw(
            gate_voltage_v=1.1,
            protocol_label="synthetic RESET",
            log10_g_hrs_ref=-9.0,
            residual_sigma_decade=0.15,
            n_training=10,
            reset_peak_abs_v=1.5,
        )
        transport = HT.EffectivePooleFrenkelLaw(
            beta_per_sqrt_v=6.2,
            residual_mad_decade=0.05,
            gate_voltage_v=1.1,
            protocol_label="synthetic RESET",
            n_training_cycles=10,
            reset_peak_abs_v=1.5,
        )
        set_voltage = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        set_current = np.array([0.0, 0.5, 1.5, 8.0, 7.0])
        reset_voltage = np.concatenate([
            np.arange(0.0, -1.5 - 1e-12, -0.02),
            np.arange(-1.48, 0.0 + 1e-12, 0.02),
        ])

        result = M11.run_refined_set_conditioned_cycle(
            set_voltage, set_current, reset_voltage, 1.1,
            transfer, endpoint, transport,
            base_params=M8.Params(reset_e50_sigma_mvc=0.0),
        )

        self.assertEqual(len(result["Is"]), len(reset_voltage))
        self.assertEqual(result["hrs_transport_model"],
                         "poole_frenkel_effective")
        self.assertEqual(result["endpoint_source"],
                         "protocol_conditioned_pf_amplitude_attractor")
        self.assertFalse(result["target_reset_current_used_for_prediction"])


if __name__ == "__main__":
    unittest.main()
