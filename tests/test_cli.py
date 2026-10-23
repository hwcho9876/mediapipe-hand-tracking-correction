from contextlib import redirect_stderr,redirect_stdout
from io import StringIO
from pathlib import Path
import os
import tempfile
import unittest
from handlab.cli import main


class CliTests(unittest.TestCase):
    def test_synthetic_and_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            stdout=StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(['synthetic','--scenario','stationary','--frames','40','--output',d]),0)
            session=next(Path(d).iterdir())
            with redirect_stdout(stdout):
                self.assertEqual(main(['analyze',str(session)]),0)

    def test_errors_reported(self):
        err=StringIO()
        with redirect_stderr(err):
            self.assertEqual(main(['synthetic','--frames','0']),1)
            self.assertEqual(main(['video','missing.mp4']),1)
        self.assertIn('HandLab:',err.getvalue())

    @unittest.skipIf(bool(os.environ.get('DISPLAY')) or os.name=='nt','needs headless Linux')
    def test_gui_missing_display_is_actionable(self):
        err=StringIO()
        with redirect_stderr(err):
            self.assertEqual(main(['gui']),1)
        self.assertIn('DISPLAY',err.getvalue())
