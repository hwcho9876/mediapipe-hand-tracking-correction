import unittest
import numpy as np
from handlab.config import Config
from handlab.pipeline import Pipeline
from handlab.synthetic import hand, sequence


class PipelineTests(unittest.TestCase):
    def test_loss_recovery_and_independent_oef(self):
        a=Pipeline(); b=Pipeline(Config(ema_enabled=False))
        states=[]; recovery_end=False
        for t,x,_ in sequence('occlusion'):
            out=a.step(x,t); other=b.step(x,t)
            np.testing.assert_allclose(a.oef.value,b.oef.value)
            np.testing.assert_allclose(a.oef.derivative,b.oef.derivative)
            states.append(out.state)
            if out.event=='RECOVERING->NORMAL':
                recovery_end=True
                self.assertLessEqual(out.metrics['recovery_elapsed'],1.)
                self.assertLess(out.metrics['recovery_gap'],.04)
        self.assertIn('LOST',states); self.assertIn('RECOVERING',states)
        self.assertTrue(recovery_end); self.assertEqual(states[-1],'NORMAL')

    def test_bad_joint_does_not_pollute_filter(self):
        p=Pipeline(); x=hand(); p.step(x,0)
        y=x.copy(); y[8]+=[.5,-.3,.4]
        out=p.step(y,.033)
        np.testing.assert_equal(p.oef.raw[8],x[8])
        self.assertEqual(out.state,'SUSPECT')
        self.assertLess(np.linalg.norm(out.points[8]-x[8]),.01)

    def test_whole_jump_and_confirmed_move(self):
        p=Pipeline(); x=hand(); p.step(x,0)
        y=x+[.5,0,0]
        out=p.step(y,.033)
        self.assertEqual(out.state,'SUSPECT')
        np.testing.assert_allclose(p.oef.raw,x)
        states=[]
        for i in range(2,32):
            out=p.step(y,i*.033)
            states.append(out.state)
        self.assertIn('RECOVERING',states)
        self.assertEqual(out.state,'NORMAL')

    def test_long_loss_hides_prediction_and_resyncs(self):
        p=Pipeline(); x=hand(); p.step(x,0)
        self.assertIsNotNone(p.step(None,.04).points)
        self.assertIsNone(p.step(None,.4).points)
        out=p.step(x+.3,2)
        self.assertTrue(out.resynced)
        self.assertEqual(out.state,'NORMAL')
        np.testing.assert_allclose(out.points,x+.3)

    def test_missing_before_first_and_invalid_times(self):
        p=Pipeline(); self.assertIsNone(p.step(None,0).points)
        p.step(hand(),.1)
        state=p.oef.value.copy()
        for t in (.1,0,float('nan')):
            self.assertEqual(p.step(hand()+.2,t).reason,'invalid_timestamp')
            np.testing.assert_equal(p.oef.value,state)

    def test_irregular_timing_timeout_handoff(self):
        c=Config(ema_tau=1.,recovery_timeout=.15,recovery_distance=.00001,recovery_velocity=.00001)
        p=Pipeline(c); x=hand(); p.step(x,0); p.step(None,.05)
        p.step(x+.05,.1)
        states=[]; t=.1
        for dt in [.01,.05,.017,.08,.03,.02]*6:
            t+=dt
            out=p.step(x+[.05+.02*np.sin(t*7),0,0],t)
            states.append(out.state)
            self.assertTrue(np.isfinite(out.points).all())
            if out.event=='RECOVERING->NORMAL':
                self.assertAlmostEqual(out.metrics['recovery_gap'],0)
        self.assertEqual(states[-1],'NORMAL')

    def test_fast_motion_not_removed(self):
        p=Pipeline(); rejected=0
        for t,x,_ in sequence('fast'):
            out=p.step(x,t)
            rejected+=out.predicted
        self.assertEqual(rejected,0)

    def test_local_prediction_expires(self):
        p=Pipeline(); x=hand(); p.step(x,0); y=x.copy(); y[8]+=[.5,-.3,.4]
        for i in range(1,15): out=p.step(y,i/30)
        self.assertIsNone(out.points)
        self.assertEqual(out.state,'LOST')
