import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time

# Panggil Data set Sesuai path
df = pd.read_csv('klasifikasi/breast-cancer.csv')
print(df.head(286))

label_encoder = LabelEncoder()

# Melakukan encoding pada fitur-fitur kategorikal
categorical_cols = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'breast', 'breast-quad', 'irradiat']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Memisahkan target variabel dan fitur-fitur lainnya
X = df.drop('Class', axis=1)  # Menghilangkan kolom 'Class' dari fitur-fitur
y = df['Class']  # Menggunakan kolom 'Class' sebagai target variabel

# Pembagian Data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Range nilai tetangga yang ingin diuji
neighbors = [1,3,5,7,9]
test_sizes = [0.2, 0.3, 0.4]

for test_size in test_sizes:
    train_size = 1 - test_size
    train_percentage = int(train_size * 100)
    test_percentage = int(test_size * 100)
    
    print(f"Split Percentage {train_percentage}%/{test_percentage}%:")
    
    # Pembagian Data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    start_time = time.time()  # Waktu awal proses
    
    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        end_time = time.time()  # Waktu akhir proses
        execution_time = end_time - start_time  # Menghitung waktu eksekusi
        execution_time_rounded = round(execution_time, 3)  # Membulatkan waktu ke tiga angka di belakang koma
        accuracy_rounded = round(accuracy, 3)  # Membulatkan akurasi ke tiga angka di belakang koma
        
        print(f"\tNilai K: {n}\n\tAccuracy: {accuracy_rounded}\n\tExecution Time: {execution_time_rounded} seconds\n")