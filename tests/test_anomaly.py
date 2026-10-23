import unittest
import numpy as np
from handlab.anomaly import AnomalyDetector
from handlab.config import Config
from handlab.correction import Corrector
from handlab.synthetic import hand,sequence


class AnomalyTests(unittest.TestCase):
    def test_fast_motion_and_fingers_preserved(self):
        for scenario in ('fast','fingers'):
            d=AnomalyDetector(Config()); count=0
            for t,x,_ in sequence(scenario):
                a=d.assess(x,t)
                count += a.kind != 'good'
                if a.kind == 'good': d.accept(x,t)
            self.assertLessEqual(count,2, scenario)

    def test_local_jump_only_bad_joint(self):
        c=Config(); d=AnomalyDetector(c); r=Corrector(c); x=hand()
        d.accept(x,0); r.accept(x,0)
        y=x.copy(); y[8]+=[.4,-.4,.3]
        a=d.assess(y,1/30)
        self.assertEqual(a.kind,'local'); self.assertTrue(a.bad[8])
        self.assertLessEqual(a.bad.sum(),2)
        repaired=r.repair(y,a.bad,1/30)
        np.testing.assert_allclose(repaired,x)
        np.testing.assert_equal(repaired[~a.bad],y[~a.bad])

    def test_rigid_jump_is_uncertain_and_confirmed(self):
        d=AnomalyDetector(Config()); x=hand(); d.accept(x,0)
        self.assertEqual(d.assess(x+[.5,0,0],1/30).kind,'uncertain')
        self.assertEqual(d.assess(x,2/30).kind,'good')
        d.accept(x,2/30)
        for i in range(3,5):
            self.assertEqual(d.assess(x+[.5,0,0],i/30).kind,'uncertain')
        self.assertEqual(d.assess(x+[.5,0,0],5/30).kind,'good')

    def test_prediction_expires(self):
        c=Corrector(Config()); c.accept(hand(),0); c.accept(hand()+.01,.03)
        p=c.predict(.1)
        self.assertTrue(np.isfinite(p).all())
        self.assertIsNone(c.predict(.5))
