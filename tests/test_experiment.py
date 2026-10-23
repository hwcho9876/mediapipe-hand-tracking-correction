import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from handlab.config import Config
from handlab.experiment import Experiment
from handlab.evaluation import summarize, lag_estimate, analyze
from handlab.recording import Recorder,load_frames
from handlab.synthetic import sequence
from handlab.tracker import Observation
from handlab.render import comparison_image


class ExperimentTests(unittest.TestCase):
    def test_record_analyze_roundtrip(self):
        e=Experiment(); frames=[]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'session'
            with Recorder(directory,{'synthetic':True},e.pipelines) as r:
                for t,x,truth in sequence(frames=90):
                    f=e.step(Observation(x),t,scenario='stationary',truth=truth)
                    r.write(f); frames.append(f)
            saved=list(load_frames(directory))
            self.assertEqual(len(saved),90)
            report=analyze(directory)
            arms={r['arm']:r for r in report['summaries']}
            self.assertLess(arms['full']['stationary_std_width_units'],arms['raw']['stationary_std_width_units'])
            self.assertIsNotNone(arms['raw']['rmse_to_supplied_truth_width_units'])
            self.assertTrue((directory/'summary.csv').is_file())
            self.assertEqual(json.loads((directory/'metadata.json').read_text())['schema_version'],1)
        image=comparison_image(np.zeros((480,640,3),dtype=np.uint8),frames[-1])
        self.assertEqual(image.size,(1280,520))

    def test_relative_lag_and_constant_rejection(self):
        t=np.arange(300)/60
        x=np.sin(2*np.pi*.7*t); y=np.sin(2*np.pi*.7*(t-.1))
        self.assertAlmostEqual(lag_estimate(t,x,y)['milliseconds'],100,delta=17)
        self.assertIsNone(lag_estimate(t,x*0,y*0))

    def test_no_accuracy_claim_without_truth(self):
        e=Experiment(); rows=[e.step(Observation(x),t) for t,x,_ in sequence(frames=40)]
        for r in summarize(rows)['summaries']:
            self.assertIsNone(r['rmse_to_supplied_truth_width_units'])
            self.assertIsNone(r['stationary_std_width_units'])

    def test_comparisons_share_raw_and_separate_states(self):
        e=Experiment()
        for t,x,_ in sequence('occlusion',frames=100):
            frame=e.step(Observation(x),t)
            if x is None: self.assertIsNone(frame['outputs']['raw']['points'])
            else: np.testing.assert_equal(frame['outputs']['raw']['points'],x)
        np.testing.assert_allclose(e.pipelines['corrected'].oef.value,e.pipelines['full'].oef.value)

    def test_fast_motion_relative_delay_and_recovery_distance(self):
        e=Experiment(); rows=[]
        for t,x,truth in sequence('fast',frames=180):
            rows.append(e.step(Observation(x),t,scenario='fast',truth=truth))
        full={r['arm']:r for r in summarize(rows)['summaries']}['full']
        self.assertLessEqual(full['raw_relative_wrist_x_lag']['milliseconds'],34)
        self.assertEqual(full['predicted_frames'],0)
        e=Experiment(); rows=[]
        for t,x,truth in sequence('occlusion',frames=180):
            rows.append(e.step(Observation(x),t,scenario='occlusion',truth=truth))
        full={r['arm']:r for r in summarize(rows)['summaries']}['full']
        self.assertLess(full['recovery_seconds'][0],.65)
        recovered=rows[78]
        out=np.asarray(recovered['outputs']['full']['points'])
        truth=np.asarray(recovered['truth'])
        from handlab.geometry import max_distance,palm_scale
        self.assertLess(max_distance(out,truth)/palm_scale(truth),.15)
