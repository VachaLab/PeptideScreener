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
        manual_sequences = request.form.get('manualSequences', '').strip()
        selected = request.form.getlist('screeners')
        custom_header_name = request.form.get('customHeader', 'sequence').strip()

        if not peptides_csv and not manual_sequences:
            print('OHNO')
            flash("Please provide peptide sequences — either upload a CSV file or enter them manually.", "warning")
            return render_template('main_page.html')
        
        # Optional: treat empty or only-spaces as default
        if not custom_header_name:
            custom_header_name = 'sequence'

        output_folder = OUTPUT_DIR / 'SCREENING_OUTPUT'
        run_id = get_next_run_id(base_dir=output_folder)
        run_name = FOLDER_SIGNATURE.replace('XX',str(run_id))
        output_folder = output_folder / run_name
        output_folder.mkdir(parents=True, exist_ok=True)

        screeners_dict = {
            opt: opt in selected
            for opt in SCREENERS_LIST 
        }

        sm = SM(screeners_dict, custom_header_name)

        if manual_sequences:
            sequences = manual_sequences.split(',')
            sequences = [s.strip() for s in sequences]
            peptides_csv_df = pd.DataFrame(
                {
                    'sequence':sequences
                }
            )
        else:
            peptides_csv_df = pd.read_csv(peptides_csv)

        df_results, df_skipped = sm.run_complete_screening(peptides_csv_df)
        df_results.to_csv(output_folder / 'screening_results.csv', index=False)

        context = {
            'run_name': run_name,
            'output_dir': output_folder,
            'screening_df': df_results,
            'visualizations': None,
            'results_file': 'screening_results.csv'
        }

        # Only add skipped_file key when the file exists / was written
        if not df_skipped.empty:
            df_skipped.to_csv(output_folder / 'skipped.csv', index=False)
            context['skipped_file'] = 'skipped.csv'

        return render_template('results.html', **context)

    return render_template('main_page.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/<path:filename>')
def download_file(filename):
    return send_from_directory('./', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',port=6969)