from danlp.datasets import DDT
import os
from pathlib import Path
cache_dir = os.path.join(str(Path.home()), '.nerda')
ner = DDT(cache_dir = cache_dir)
test = ner.load_as_simple_ner(True)