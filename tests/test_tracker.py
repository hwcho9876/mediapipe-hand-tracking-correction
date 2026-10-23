import unittest
from types import SimpleNamespace as S
import numpy as np
from handlab.tracker import convert_result, HandTracker
from handlab.geometry import to_normalized
from handlab.capture import VideoClock


class TrackerTests(unittest.TestCase):
    def test_output_contract(self):
        pts = [S(x=.5, y=.6, z=-.1)] * 21
        r = S(hand_landmarks=[pts], hand_world_landmarks=[pts],
              handedness=[[S(category_name='Left', score=.9)]])
        obs = convert_result(r, 640, 480, 42)
        np.testing.assert_allclose(obs.points[0], [.5, .45, -.1])
        np.testing.assert_allclose(to_normalized(obs.points, 640, 480)[0], [.5, .6, -.1])
        self.assertEqual(obs.handedness_score, .9)
        self.assertFalse(hasattr(obs, 'landmark_confidence'))

    def test_missing(self):
        self.assertIsNone(convert_result(S(hand_landmarks=[]), 640, 480, 0).points)
        with self.assertRaises(FileNotFoundError):
            HandTracker('/nonexistent/model.task')

    def test_video_timestamps(self):
        clock = VideoClock(25)
        values = [clock.timestamp(x) for x in (0, 40, 40, 20, float('nan'), 240)]
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertEqual(clock.fallbacks, 3)
