import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def main():
    print("Memulai proses training model ID3...")

    # 1. Load Data
    print("1. Memuat dataset...")
    df = pd.read_csv("Stock_Trading_Categorical_Dataset_Large.csv") # Sesuaikan nama file jika perlu
    
    # Preprocessing (Menghapus kolom No jika ada)
    if 'No' in df.columns:
        df = df.drop(columns=['No'])

    categorical_cols = ["Tren_Harga", "Broker_Summary", "Volume_Transaksi", "Status_Orderbook"]
    X = df[categorical_cols]
    y = df['Keputusan']

    # 2 & 3 & 4. Inisialisasi Encoder, ID3, dan Pipeline
    print("2. Membangun Pipeline (Encoder + Decision Tree)...")
    encoder = OneHotEncoder(sparse_output=False)
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
    
    model_pipeline = Pipeline([
        ('preprocessor', encoder),
        ('classifier', classifier)
    ])

    # 5. Training
    print("3. Melatih model...")
    model_pipeline.fit(X, y)

    # 6. Simpan Model (.pkl)
    print("4. Menyimpan model ke 'mymodel.pkl'...")
    joblib.dump(model_pipeline, 'mymodel.pkl')

    # 7. Ekstrak dan Simpan Rules (rules.txt)
    print("5. Menyimpan aturan IF-THEN ke 'rules.txt'...")
    tree_model = model_pipeline.named_steps['classifier']
    nama_fitur = model_pipeline.named_steps['preprocessor'].get_feature_names_out(categorical_cols).tolist()
    
    tree_rules = export_text(tree_model, feature_names=nama_fitur)
    with open("rules.txt", "w") as file:
        file.write(tree_rules)

    # 8. Simpan Visualisasi (tree.png)
    if plt is not None:
        print("6. Menyimpan visualisasi ke 'tree.png'...")
        plt.figure(figsize=(16, 10))
        plot_tree(tree_model,
                  feature_names=nama_fitur,
                  class_names=tree_model.classes_,
                  filled=True, rounded=True, fontsize=10)
        plt.title("Visualisasi Pohon Keputusan (ID3) - Analisis Saham", fontsize=14)
        plt.savefig('tree.png')
    else:
        print("6. matplotlib tidak tersedia; visualisasi tree.png dilewati.")
    print("Proses Selesai!")

if __name__ == "__main__":
    main()