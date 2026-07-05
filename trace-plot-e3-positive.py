import os
import runpy
from pathlib import Path


os.environ['E3_TRACE_SUFFIX'] = 'e3-positive'
os.environ['E3_TRACE_INPUT_SUFFIX'] = 'e3-positive'
runpy.run_path(str(Path(__file__).resolve().with_name('trace-plot-e3.py')), run_name='__main__')
