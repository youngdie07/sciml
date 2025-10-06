#!/usr/bin/env python
import json

# Read the corrupted notebook
with open('docs/03-function-encoder/03-function-encoders.ipynb', 'r') as f:
    nb = json.load(f)

# Fix code cells by adding proper newlines
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and isinstance(cell['source'], list):
        # Join and resplit properly
        source_text = ''.join(cell['source'])

        # Add newlines after common Python keywords
        import re

        # Fix imports
        source_text = source_text.replace('import numpy as npimport', 'import numpy as np\nimport')
        source_text = source_text.replace('import torchimport', 'import torch\nimport')
        source_text = source_text.replace('import torch.nn as nnimport', 'import torch.nn as nn\nimport')
        source_text = source_text.replace('import torch.optim as optimimport', 'import torch.optim as optim\nimport')
        source_text = source_text.replace('import torch.nn.functional as Ffrom', 'import torch.nn.functional as F\nfrom')
        source_text = source_text.replace('from torch.utils.data import Dataset, DataLoaderimport', 'from torch.utils.data import Dataset, DataLoader\nimport')
        source_text = source_text.replace('import matplotlib.pyplot as pltfrom', 'import matplotlib.pyplot as plt\nfrom')
        source_text = source_text.replace('from tqdm import tqdmimport', 'from tqdm import tqdm\nimport')
        source_text = source_text.replace('import warningswarnings', 'import warnings\nwarnings')
        source_text = source_text.replace('warnings.filterwarnings', 'warnings.filterwarnings')
        source_text = source_text.replace("filterwarnings('ignore')np.random", "filterwarnings('ignore')\n\nnp.random")
        source_text = source_text.replace('np.random.seed(42)torch.manual_seed', 'np.random.seed(42)\ntorch.manual_seed')
        source_text = source_text.replace('torch.manual_seed(42)device', 'torch.manual_seed(42)\n\ndevice')
        source_text = source_text.replace('"cpu")print', '"cpu")\nprint')

        # Fix function definitions
        source_text = source_text.replace('def compare_classical_bases():    x', 'def compare_classical_bases():\n    x')
        source_text = source_text.replace('def target(x):        return', 'def target(x):\n        return')
        source_text = source_text.replace('y_true = target(x)    n_basis', 'y_true = target(x)\n    n_basis')

        # Fix after common patterns
        source_text = re.sub(r'(\))([a-zA-Z_])', r'\1\n\2', source_text)
        source_text = re.sub(r'(    )([a-zA-Z_])', r'\n\1\2', source_text)

        # This is complex - let's just use the reference notebook instead
        cell['source'] = source_text

# Actually, let's just copy from the working function-encoder.ipynb
print("Reading reference notebook...")
with open('docs/03-function-encoder/function-encoder.ipynb', 'r') as f:
    ref_nb = json.load(f)

print(f"Reference notebook has {len(ref_nb['cells'])} cells")
print("This will be used as the base")
