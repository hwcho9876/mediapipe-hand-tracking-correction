import unittest
import numpy as np
from handlab.filters import OneEuro, alpha
from handlab.synthetic import hand, sequence


class FilterTests(unittest.TestCase):
    def test_static_noise(self):
        f=OneEuro(beta=0)
        raw=[]; out=[]
        for t,x,_ in sequence():
            raw.append(x); out.append(f.update(x,t))
        self.assertLess(np.std(np.array(out)[30:],axis=0).mean(), .5*np.std(np.array(raw)[30:],axis=0).mean())

    def test_fast_beta_reduces_lag(self):
        slow=OneEuro(beta=0); fast=OneEuro(beta=10)
        errors=[[],[]]
        for t,x,_ in sequence('fast'):
            for idx,f in enumerate((slow,fast)):
                errors[idx].append(np.mean((f.update(x,t)-x)**2))
        self.assertLess(np.mean(errors[1]),np.mean(errors[0]))

    def test_default_tradeoff_on_synthetic_motion(self):
        from handlab.config import Config
        from handlab.geometry import center
        c=Config()
        f=OneEuro(c.min_cutoff,c.beta,c.d_cutoff)
        samples=[(x,f.update(x,t),truth) for t,x,truth in sequence('fast',frames=180)]
        err=np.sqrt(np.mean([(y-truth)**2 for x,y,truth in samples]))
        self.assertLess(err,.02)  # width units, only for this synthetic motion
        self.assertLess(np.mean([(y-x)**2 for x,y,_ in samples]), .001)

    def test_actual_dt(self):
        f=OneEuro(beta=0); x=hand()
        f.update(x,0)
        y=f.update(x+.1,.1)
        np.testing.assert_allclose(y,x+alpha(1.5,.1)*.1)

    def test_invalid_times_and_freeze(self):
        f=OneEuro(); x=hand(); f.update(x,1)
        for t in (1,.9,float('nan')):
            np.testing.assert_equal(f.update(x+1,t),x)
        mask=np.ones(21,bool); mask[8]=False
        y=f.update(x+.1,1.1,mask)
        np.testing.assert_equal(y[8],x[8])
        self.assertEqual(f.last_t[8],1)
        np.testing.assert_allclose(f.update(x+.2,3),x+.2)
