from pathlib import Path


SCREENERS_LIST = ['toxicity', 'solubility']
DEVICE_OPTIONS = ['cpu','cuda','mps']

FOLDER_SIGNATURE = 'screening_run_XX'
OUTPUT_DIR = Path('static/runs')

toxicity_clf_path = Path('src/screeners/toxicity/RFC_esm.pkl')