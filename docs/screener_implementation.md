# Peptide Screener – Implementation Guide

This guide explains how to **add a new screening tool** (e.g. toxicity, solubility, immunogenicity, etc.) to the **Peptide Screener** .

## Step 1 – Create Directory & Files

1. Create a new folder under  
   `src/screeners/`

   Name it meaningfully according to what the tool predicts, for example:
   - `toxicity`
   - `solubility`
   - `stability`
   - `binding_affinity`
   - etc.

   **→ If a suitable folder already exists, navigate to that folder.**

2. Copy the template file:  
   `src/screeners/screener_template.py`  
   → paste it into your folder

3. Rename the file and the class inside to match your tool, e.g.:
   - `toxicity_screener.py`
   - `class ToxicityScreener(Screener):`

4. Place any model files or other dependencies in the same folder:
   - `.pkl`, `.joblib`, `.h5`, `.pt`, `.onnx`, etc.

## Step 2 – Implement the Abstract `Screener` Class

Your screener must inherit from the abstract base class `Screener` and implement the required method.

Core method to implement:

```python
def run_screening(self, df: pd.DataFrame) -> pd.DataFrame:
    ...
```

**Rules:**
- Input: pandas DataFrame (contains a column with peptide sequences)
- Output: **the same DataFrame** with **one or more new columns** added containing your predictions / scores

Typical flow inside `run_screening()`:

1. Extract sequences (`df["sequence"]`)
2. Perform any necessary preprocessing / feature calculation
3. Run model inference
4. Add result column(s) back to the DataFrame
5. Return the modified DataFrame

<img src="imgs/abs_screener.png" alt="Abstract Screener class diagram" width="620">

<img src="imgs/example_screener.png" alt="Example implementation of run_screening method" width="620">

## Step 3 – Edit Configuration

Open `src/config.py`

1. Add your screener name to the list:

```python
SCREENER_LIST = [
    ...,
    "toxicity",
    "solubility",
    # your new name here
]
```

2. If your tool uses a trained model, add the path(s) (follow existing examples):

```python
toxicity_clf_path = Path('path/to/clf.pkl')
```

## Step 4 – Edit Screener Manager

Open `src/screener_manager.py`

1. Add import for your screener class:

```python
from screeners.toxicity.toxicity_screener import ToxicityScreener
```

2. (Optional) Import model if needed

```python
from src.config import toxicity_clf_path
```

3. Add your screener to the `_get_screener_instance()` method:

Follow the pattern of existing screeners.

## Step 5 – Add UI Control in Web Interface

Open `templates/main_page.html`

1. Find section **SCREENING OPTIONS** (~ line 101)
2. Locate the comment `<!-- ADD NEW SECTION -->`
3. Uncomment the block below it
4. Replace placeholder values:

```html
<div class="form-check">
    <input class="form-check-input" type="checkbox" name="screeners" value="toxicity" id="scr_toxicity">
    <label class="form-check-label" for="scr_toxicity">
        TOXICITY
    </label>
</div>
```

Make sure:
- `value="..."` matches the name in `SCREENER_LIST`
- `id="scr_screener"` and `for="scr_screener"` use the same name

## Done! 🎉

##  OPTIONAL Step 6 – Add documentation to your tool

Open `templates/documentation.html` and use the commented `<div>` block to add section for your tool !

## ADDITIONAL INFORMATION
if your tools needs <strong>python package<strong> which is not included, try to install that package via:

```bash
uv add package
```

if there is problem with package installation, <strong>contact us<strong> for the help with tool implementation