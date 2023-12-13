import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time

# Panggil Data set Sesuai path
df = pd.read_csv('klasifikasi/breast-cancer.csv')

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
x = df.iloc[:,:-1] # Target
y = df.iloc[:,-1] # Feature
# Range nilai tetangga yang ingin diuji
neighbors = [3]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print("X train : ", x_train.shape)
print("X test : ", x_test.shape)
print("Y train : ", y_train.shape)
print("Y test : ", y_test.shape)

for n in neighbors:
    start_time = time.time()  # Waktu awal proses
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    end_time = time.time()  # Waktu akhir proses
    execution_time = end_time - start_time  # Menghitung waktu eksekusi
    execution_time_rounded = round(execution_time, 3)  # Membulatkan waktu ke tiga angka di belakang koma
    accuracy_rounded = round(accuracy, 3)  # Membulatkan akurasi ke tiga angka di belakang koma
    print(f"Neighbors: {n}\nAccuracy: {accuracy_rounded}\nExecution Time: {execution_time_rounded} seconds \n")