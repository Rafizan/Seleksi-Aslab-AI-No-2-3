# Supervised Learning & Unsupervised Learning

Kumpulan implementasi model *machine learning* untuk klasifikasi dan klastering data.

Proyek ini dibuat untuk seleksi asisten lab AI tahun 2025.

### Model yang Diimplementasikan

Supervised Learning (Bagian 2)
[v] KNN
[v] LogReg
[v] Gaussian Naive Bayes
[v] CART
[v] SVM
[v] ANN

Unsupervised Learning (Bagian 3)
[v] K-MEANS
[v] DBSCAN
[v] PCA

---
## Cara Penggunaan

#### 1. Persiapan

Install semua library yang dibutuhkan dan jalankan notebook preprocessing.
```bash
# Install library
pip install pandas numpy scikit-learn jupyterlab

# Jalankan notebook untuk membersihkan data
jupyter lab src/EDA-Preprocessing.ipynb

# Jalankan script .py yang diinginkan dari folder src/
python src/supervised/knn.py
```
**Penting:** pastikan input path dataset di dalam .py sudah benar mengarah ke file .csv yang telah diproses