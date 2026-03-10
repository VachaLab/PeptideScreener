from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from src.screener_manager import ScreenerManager as SM
from src.utils import get_next_run_id
import pandas as pd

### screener design ###

from src.screeners.screener_design.screener_pchem import PeptideScreenerPCHEM
from src.screeners.screener_design.screener_plm import PeptideScreenerPLM
from src.screeners.screener_design.screener_cf import PeptideScreenerCF

from src.config import SCREENERS_LIST, FOLDER_SIGNATURE, OUTPUT_DIR, EMBEDDER_OPTIONS

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
            flash("Please provide peptide sequences — either upload a CSV file or enter them manually.", "warning")
            return render_template('main_page.html')
        
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

        download_files = ['screening_results.csv']
        if not df_skipped.empty:
            download_files.append('skipped.csv')

        context = {
            'run_name': run_name,
            'output_dir': output_folder,
            'screening_df': df_results,
            'visualizations': None,
            'download_files': download_files
        }

        return render_template('results.html', **context)

    return render_template('main_page.html')

@app.route('/screener_design', methods=['GET','POST'])
def screener_design():

    if request.method == 'POST':

        train_csv = request.files['TrainCSV']
        val_csv = request.files['ValidationCSV']
        feature_generator = request.form.get('feature_generator')

        seq_header = request.form.get('custom_seq_header', 'sequence').strip()
        label_header = request.form.get('custom_label_header', 'label').strip()

        if not seq_header:
            seq_header = 'sequence'
        if not label_header:
            label_header = 'label'

        if not train_csv or not val_csv:
            flash("Provide training and validation datasets", "warning")
            return render_template('screener_design.html')
        
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        output_folder = OUTPUT_DIR / 'SCREENER_DESIGN'
        run_id = get_next_run_id(base_dir=output_folder)
        run_name = FOLDER_SIGNATURE.replace('XX',str(run_id))
        output_folder = output_folder / run_name
        output_folder.mkdir(parents=True, exist_ok=True)

        if EMBEDDER_OPTIONS[feature_generator] == 'PLM':
            peptide_screener = PeptideScreenerPLM(embedder_key=feature_generator, seq_header=seq_header,label_header=label_header)
        if EMBEDDER_OPTIONS[feature_generator] == 'PCHEM':
            peptide_screener = PeptideScreenerPCHEM(embedder_key=feature_generator, seq_header=seq_header,label_header=label_header)
        if EMBEDDER_OPTIONS[feature_generator] == 'CF':
            peptide_screener = PeptideScreenerCF(embedder_key=feature_generator, seq_header=seq_header,label_header=label_header)

        peptide_screener.design_screener(train_df, val_df, output_folder)

        visualizations = ['train.png','validation.png']
        if feature_generator == 'PCHEM' or feature_generator == 'CUSTOM_FEATURES':
            visualizations.append('feature_importance.png')


        download_files = ['clf.pkl', 'config.yaml']

        context = {
            'run_name': run_name,
            'output_dir': output_folder,
            'screening_df': None,
            'visualizations': visualizations,
            'download_files': download_files
        }

        return render_template('results.html', **context)

    feature_generators = list(EMBEDDER_OPTIONS.keys())
    return render_template('screener_design.html', feature_generators=feature_generators)

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/<path:filename>')
def download_file(filename):
    return send_from_directory('./', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',port=6969)