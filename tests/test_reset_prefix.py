import unittest

import numpy as np

from vulcan2d import reset_prefix as prefix


class ResetPrefixTest(unittest.TestCase):
    def test_features_require_only_the_outgoing_prefix(self):
        voltage = -np.linspace(0., .23, 24)
        grid = np.linspace(0., 1.5, 151)
        transistor = (grid, grid * 1e-3)
        measured = np.abs(voltage) * 2e-6
        result = prefix.features(voltage, measured, transistor, .23)
        self.assertEqual(result.shape, (10,))
        self.assertTrue(np.isfinite(result).all())
        with self.assertRaisesRegex(ValueError, "outgoing RESET prefix"):
            prefix.features(np.r_[voltage, -.24], np.r_[measured, 2e-7], transistor, .23)
        with self.assertRaisesRegex(ValueError, "outgoing RESET prefix"):
            prefix.features(np.r_[voltage, -.22], np.r_[measured, 2e-7], transistor, .23)

    def test_masked_training_targets_cannot_change_predictions(self):
        x = np.arange(20., dtype=float).reshape(10, 2)
        y = np.stack((x[:, 0]*.03, x[:, 1]*.02), axis=1)
        mask = np.ones_like(y, dtype=bool)
        mask[::2, 0] = False
        changed = np.where(mask, y, -100.)
        for method, size in (("neighbors", 6.), ("linear", .15), ("quadratic", .15)):
            fit = prefix.fit(x, y, mask, method, size)
            alternate = prefix.fit(x, changed, mask, method, size)
            np.testing.assert_allclose(prefix.predict(fit, x[:3]),
                                       prefix.predict(alternate, x[:3]))


if __name__ == "__main__":
    unittest.main()
