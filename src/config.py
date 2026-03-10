from pathlib import Path

### ADD YOUR SCREENER TO THE 'SCREENER_LIST' WITH CUSTOM NAME (does not have to reflect python class name)
SCREENERS_LIST = ['toxicity', 'solubility_will', 'solubility_jana']

### ADD PATHS TO NEW SCREENER MODELS ###
toxicity_clf_path = Path('src/screeners/toxicity/RFC_esm.pkl')
solubility_will_clf_path = Path('src/screeners/solubility/williams_model.joblib')
solubility_jana_clf_path = Path('src/screeners/solubility/janas_model.pkl')

### DO NOT EDIT ###
DEVICE_OPTIONS = ['cpu','cuda','mps']
FOLDER_SIGNATURE = 'screening_run_XX'
OUTPUT_DIR = Path('static/runs')
EMBEDDER_OPTIONS = {'ESM2':'PLM', 'PBERT':'PLM', 'PCHEM':'PCHEM', 'CUSTOM_FEATURES':'CF'}