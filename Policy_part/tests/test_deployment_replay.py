"""Clock-boundary checks; no model execution or external services."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbannav_deploy.replay import plan_clip


def frames(stamps, prefix):
    return [{'timestamp_ns': t, 'path': f'{prefix}/{i}.jpg'} for i, t in enumerate(stamps)]


class CausalReplayTests(unittest.TestCase):
    def test_separate_rates_and_future_bev_exclusion(self):
        fpv = frames(range(0, 2_000_000_001, 100_000_000), 'fpv')
        bev = frames([0, 510_000_000, 1_010_000_000, 1_510_000_000], 'bev')
        events = plan_clip(fpv, bev, start_ns=0, end_ns=2_000_000_000)
        predictions = [e for e in events if e['kind'] == 'predict']
        self.assertEqual([e['status'] for e in predictions], ['rejected'] * 2 + ['eligible'] * 3)
        self.assertEqual(predictions[2]['bev']['timestamp_ns'], 510_000_000)
        self.assertEqual(len([e for e in events if e['kind'] == 'fpv']), 11)
        self.assertTrue(all(e['bev']['timestamp_ns'] <= e['observation_ns']
                            for e in predictions if e['status'] == 'eligible'))

    def test_gap_clears_history_instead_of_padding(self):
        events = plan_clip(frames([0, 200_000_000, 800_000_000, 1_000_000_000], 'f'),
                           frames([0], 'b'), start_ns=0, end_ns=1_000_000_000)
        self.assertFalse(any(e['status'] == 'eligible' for e in events))
        self.assertTrue(any(e.get('reason') == 'missing_or_stale_fpv' for e in events))

    def test_invalid_source_clock_fails(self):
        with self.assertRaises(ValueError):
            plan_clip(frames([0, 0], 'f'), frames([0], 'b'), start_ns=0, end_ns=1)


if __name__ == '__main__':
    unittest.main()
