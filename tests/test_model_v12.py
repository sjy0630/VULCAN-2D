import inspect
import unittest

import numpy as np

from vulcan2d import model_v12 as M12
from vulcan2d.hrs_transport import EffectivePooleFrenkelLaw, effective_pf_basis
from vulcan2d.reset_identification import (
    ReconstructedReset,
    identify_pf_reset_state,
    logistic_state,
)
from vulcan2d.set_identification import (
    SetTransitionLaw,
    SetTransportLaw,
    connected_basis,
    direct_set_transistor_mapping,
    effective_pre_basis,
)


class _LinearMapping:
    def current(self, terminal_vds, gate_voltage, side="set"):
        del gate_voltage, side
        return 8e-5 * abs(float(terminal_vds))


def _transport_law():
    return SetTransportLaw(
        beta_pre_per_sqrt_v=2.0,
        connected_exponent=1.25,
        log10_a_pre_ref=-8.0,
        log10_a_connected_ref=-4.7,
        pre_amplitude_sigma_decade=0.1,
        connected_amplitude_sigma_decade=0.1,
        pre_shape_rmse_decade=0.1,
        connected_shape_rmse_decade=0.1,
        gate_voltage_v=1.1,
        protocol_label="synthetic SET",
        n_training_cycles=10,
        set_peak_abs_v=2.0,
    )


def _transition_law():
    return SetTransitionLaw(
        e50_v2=0.35,
        width_v2=0.04,
        e50_sigma_v2=0.05,
        width_sigma_v2=0.01,
        residual_rms_state=0.03,
        gate_voltage_v=1.1,
        protocol_label="synthetic SET",
        n_training_cycles=10,
        set_peak_abs_v=2.0,
    )


class ModelV12Tests(unittest.TestCase):
    def test_set_envelopes_match_documented_equations(self):
        voltage = np.array([0.0, 0.1, 0.5, 1.0])
        np.testing.assert_allclose(
            effective_pre_basis(voltage, 2.5),
            voltage * np.exp(2.5 * np.sqrt(voltage)),
        )
        np.testing.assert_allclose(
            connected_basis(voltage, 1.3), voltage ** 1.3)

    def test_literal_set_mapping_does_not_shift_gate_voltage(self):
        mapping = direct_set_transistor_mapping()
        self.assertEqual(mapping.effective_gate_voltage(1.1, "set"), 1.1)
        self.assertEqual(mapping.gate_gain, 1.0)
        self.assertEqual(mapping.gate_offset_v, 0.0)

    def test_link_fit_recovers_known_cross_protocol_factor(self):
        connected = np.array([8e-6, 1e-5, 1.2e-5, 1.4e-5])
        exponent = 1.25
        alpha = 8.0
        factor = 0.27
        reference = 0.1
        reset = (
            factor * connected * reference ** exponent
            / np.sinh(alpha * reference)
        )
        law = M12.fit_set_reset_link_law(
            connected, reset, exponent, alpha,
            "synthetic SET", "synthetic RESET", reference)
        self.assertAlmostEqual(law.retention_factor, factor, places=12)
        self.assertAlmostEqual(law.residual_sigma_decade, 0.0, places=12)

    def test_autonomous_set_needs_voltage_not_target_current(self):
        voltage = np.concatenate([
            np.linspace(0.0, 2.0, 101),
            np.linspace(1.98, 0.0, 100),
        ])
        result = M12.run_autonomous_set(
            voltage, 1.1, _transport_law(), _transition_law(),
            mapping=_LinearMapping())

        self.assertEqual(len(result["Is"]), len(voltage))
        self.assertFalse(result["target_set_current_used_for_prediction"])
        self.assertTrue(np.all(np.diff(result["xs"][:101]) >= -1e-12))
        self.assertGreater(result["xs"][100], 0.95)
        self.assertAlmostEqual(result["xs"][-1], result["xs"][100])
        self.assertTrue(np.all(np.diff(
            result["set_exposure_v2"][:101]) >= -1e-12))

        # Every stored point also satisfies the transistor load line.
        mapping = _LinearMapping()
        for index in (10, 50, 100, 150):
            transistor_current = mapping.current(
                abs(result["Vs"][index]) - abs(result["Vhs"][index]),
                1.1,
            )
            self.assertAlmostEqual(
                abs(result["Is"][index]), transistor_current,
                delta=max(1e-13, transistor_current * 1e-8),
            )

    def test_full_cycle_api_cannot_receive_target_currents(self):
        parameters = inspect.signature(
            M12.run_autonomous_full_cycle).parameters
        self.assertNotIn("set_current", parameters)
        self.assertNotIn("reset_current", parameters)

    def test_default_calibration_matches_reported_central_values(self):
        calibration = M12.default_calibration()
        self.assertAlmostEqual(
            calibration.set_transition.e50_v2, 1.1350926441681073)
        self.assertAlmostEqual(
            calibration.set_reset_link.retention_factor,
            0.2575148876811263,
        )
        self.assertEqual(calibration.set_transport.n_training_cycles, 53)
        self.assertEqual(calibration.reset_transport.n_training_cycles, 51)
        self.assertAlmostEqual(
            calibration.reset_transition.e50_mvc,
            0.7926771526923162,
        )

    def test_calibrated_cycle_exposes_both_reset_transition_modes(self):
        set_voltage = np.r_[
            np.linspace(0.0, 5.0, 11), np.linspace(4.5, 0.0, 10)]
        reset_voltage = np.r_[
            np.linspace(0.0, -1.7, 9), np.linspace(-1.5, 0.0, 8)]
        local = M12.simulate_calibrated_cycle(
            set_voltage, reset_voltage,
            reset_transition_mode="same_protocol")
        prior = M12.simulate_calibrated_cycle(
            set_voltage, reset_voltage,
            reset_transition_mode="cross_gate_prior")

        self.assertEqual(
            local["reset_transition_source"],
            "protocol_fitted_PF_envelope_state",
        )
        self.assertEqual(
            prior["reset_transition_source"],
            "base_parameter_or_explicit_override",
        )
        self.assertNotEqual(
            local["reset"]["e50_mvc"], prior["reset"]["e50_mvc"])

    def test_pf_reset_state_recovers_a_synthetic_transition(self):
        alpha_lrs = 8.0
        beta_pf = 6.0
        g_lrs = 2e-7
        a_hrs = 8e-10
        e50 = 1.0
        width = 0.12
        forward = np.linspace(0.08, 1.2, 80)
        returning = forward[-2::-1]
        voltage = np.concatenate([forward, returning])
        state_forward = logistic_state(
            10.0 * forward / 6.0, e50, width)
        i_lrs = g_lrs * np.sinh(alpha_lrs * forward)
        i_hrs = a_hrs * effective_pf_basis(forward, beta_pf)
        current_forward = state_forward * i_lrs + (1.0 - state_forward) * i_hrs
        current_return = a_hrs * effective_pf_basis(returning, beta_pf)
        cycle = ReconstructedReset(
            gate_voltage_v=1.1,
            voltage_app_v=voltage,
            current_a=np.concatenate([current_forward, current_return]),
            voltage_ds_v=np.zeros_like(voltage),
            voltage_hbn_v=voltage,
            valid=np.ones_like(voltage, dtype=bool),
            apex_index=len(forward) - 1,
        )
        law = EffectivePooleFrenkelLaw(
            beta_per_sqrt_v=beta_pf,
            residual_mad_decade=0.0,
            gate_voltage_v=1.1,
            protocol_label="synthetic RESET",
            n_training_cycles=10,
            reset_peak_abs_v=1.2,
        )

        identified = identify_pf_reset_state(cycle, alpha_lrs, law)

        self.assertAlmostEqual(identified.e50_mvc, e50, delta=0.08)
        self.assertAlmostEqual(identified.width_mvc, width, delta=0.04)
        self.assertLess(identified.state_fit_rms, 0.04)


if __name__ == "__main__":
    unittest.main()
