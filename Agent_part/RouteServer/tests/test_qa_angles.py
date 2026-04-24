from __future__ import annotations

import unittest

from mcp_server.qa.angles import (
    bev_theta_deg,
    coarse_bin_from_theta,
    fpv_theta_from_u,
    theta_bin_residual,
    u_from_fpv_theta,
    wrap_angle_deg,
)


class QAAnglesTest(unittest.TestCase):
    def test_bev_theta_convention(self) -> None:
        self.assertAlmostEqual(bev_theta_deg(0.0, 1.0), 0.0)
        self.assertAlmostEqual(bev_theta_deg(1.0, 0.0), 90.0)
        self.assertAlmostEqual(bev_theta_deg(-1.0, 0.0), -90.0)

    def test_fpv_u_theta_roundtrip(self) -> None:
        width = 1000
        for theta in [-50.0, 0.0, 50.0]:
            u = u_from_fpv_theta(theta, width)
            self.assertAlmostEqual(fpv_theta_from_u(u, width), theta)

    def test_bins_and_wrap(self) -> None:
        self.assertEqual(coarse_bin_from_theta(-20.0), "left_front")
        self.assertEqual(coarse_bin_from_theta(0.0), "front")
        self.assertEqual(coarse_bin_from_theta(20.0), "right_front")
        bin_index, residual = theta_bin_residual(0.0)
        self.assertEqual(bin_index, 6)
        self.assertAlmostEqual(residual, -7.5)
        self.assertAlmostEqual(wrap_angle_deg(190.0), -170.0)


if __name__ == "__main__":
    unittest.main()
