from pathlib import Path

from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / 'mymodel.pkl')
except (FileNotFoundError, EOFError, ValueError, ImportError):
    model = None

@app.route('/')
def dashboard():
    rules_path = BASE_DIR.parent / 'rules_human.txt'
    try:
        rules_text = rules_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        rules_text = 'Aturan belum digenerate. Jalankan cell IF-THEN pada EDA.ipynb.'
    return render_template('dashboard.html', rules_text=rules_text)

@app.route('/tree-image')
def tree_image():
    tree_path = BASE_DIR.parent / 'tree.png'
    return send_file(tree_path, mimetype='image/png', max_age=0)

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    hasil_prediksi = None
    if request.method == 'POST':
        if model is not None:
            data = pd.DataFrame({
                'Tren_Harga': [request.form['tren_harga']],
                'Broker_Summary': [request.form['broker_summary']],
                'Volume_Transaksi': [request.form['volume_transaksi']],
                'Status_Orderbook': [request.form['status_orderbook']]
            })
            hasil_prediksi = model.predict(data)[0]
        else:
            hasil_prediksi = "Model belum dimuat. Jalankan train_model.py terlebih dahulu."
            
    return render_template('prediction.html', hasil=hasil_prediksi)

@app.route('/tentang')
def tentang():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)