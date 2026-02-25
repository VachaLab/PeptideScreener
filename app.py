from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from src.screener_manager import ScreenerManager as SM
from src.utils import get_next_run_id
import pandas as pd

from src.config import SCREENERS_LIST, FOLDER_SIGNATURE, OUTPUT_DIR

from __version__ import __version__

app = Flask(__name__)
app.secret_key = "your-secret-key"

@app.context_processor
def inject_version():
    return dict(app_version=app.config['APP_VERSION'])

app.config['APP_VERSION'] = __version__

@app.route('/', methods=['GET','POST'])
def main_page():

    if request.method == 'POST':

        peptides_csv = request.files['PeptideCSV']
        selected = request.form.getlist('screeners')

        output_folder = OUTPUT_DIR / 'SCREENING_OUTPUT'
        run_id = get_next_run_id(base_dir=output_folder)
        run_name = FOLDER_SIGNATURE.replace('XX',str(run_id))
        output_folder = output_folder / run_name
        output_folder.mkdir(parents=True, exist_ok=True)

        screeners_dict = {
            opt: opt in selected
            for opt in SCREENERS_LIST 
        }

        sm = SM(screeners_dict)

        peptides_csv_df = pd.read_csv(peptides_csv)
        df_results = sm.run_complete_screening(peptides_csv_df)
        df_results.to_csv(output_folder / 'screening_results.csv', index=False)

        return render_template('results.html',
            run_name = run_name,
            output_dir=output_folder,
            screening_df = df_results,
            visualizations = None,
            results_file='screening_results.csv'
        )

    return render_template('main_page.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/<path:filename>')
def download_file(filename):
    return send_from_directory('./', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',port=6969)