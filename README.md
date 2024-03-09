# Logistic Regression from Scratch

## 1. Pengantar Regresi Logistik
Analisis regresi logistik merupakan suatu pendekatan untuk membuat 
model prediksi seperti halnya regresi linear atau yang biasa disebut dengan 
istilah Ordinary Least Squares (OLS) regression. Perbedaannya yaitu pada 
regresi logistik, peneliti memprediksi variabel terikat yang berskala dikotomi. 
Skala dikotomi yang dimaksud adalah skala data nominal dengan dua kategori, 
misalnya: besar dan kecil, baik dan buruk, atau berhasil dan gagal. Pada
Analisis OLS mewajibkan syarat atau asumsi bahwa error varians (residual) 
terdistribusi secara normal. Sebaliknya, pada regresi logistik tidak mensyaratkan
asumsi tersebut karena pada regresi logistik mengikuti distribusi logistik. Berikut 
syarat yang ada dalam regresi logistik yaitu:
1. Regresi logistik tidak membutuhkan hubungan linier antara variabel 
independen dengan variabel dependen.
2. Variabel independen tidak memerlukan asumsi multivariate normality.
3. Asumsi homokedastisitas tidak diperlukan
4. Variabel bebas tidak perlu diubah ke dalam bentuk skala interval atau ratio.
5. Variabel dependen harus bersifat dikotomi (2 kategori)
6. Variabel independen tidak harus memiliki varian yang sama antar kelompok
variabel
7. Kategori dalam variabel independen harus terpisah satu sama lain atau 
bersifat eksklusif
8. Sampel yang diperlukan dalam jumlah relatif besar, minimum dibutuhkan 
hingga 50 sampel data untuk sebuah variabel prediktor (independen).
9. Regresi logistik dapat menyeleksi hubungan karena menggunakan 
pendekatan non linier log transformasi untuk memprediksi odds ratio. Odd 
dalam regresi logistik sering dinyatakan sebagai probabilitas.

### **Fungsi Logistik**

Regresi logistik adalah model statistik yang menggunakan fungsi logistik, atau fungsi logit, dalam matematika sebagai persamaan antara x dan y. Fungsi logit memetakan y sebagai fungsi sigmoid dari x.

![Gambar fungsi logistik](https://d1.awsstatic.com/sigmoid.bfc853980146c5868a496eafea4fb79907675f44.png)

grafik sigmoid dalam regresi logistik adalah sebagai berikut

![Grafik Sigmoid](https://d1.awsstatic.com/S-curve.36de3c694cafe97ef4e391ed26a5cb0b357f6316.png)

### **Model Regresi Logistik**
Secara umum berikut Model persamaan regresi logistik dan perbedaannya dengan regresi linier:
![Grafik Sigmoid](https://miro.medium.com/v2/resize:fit:828/format:webp/1*OqMldkIfSI3DpTrCB2b2Bw.png)

## 2. Loss Function (Fungsi Biaya)
Loss function (fungsi biaya) pada regresi logistik, sering disebut sebagai log loss atau binary cross-entropy loss, digunakan untuk mengukur seberapa baik model regresi logistik memprediksi kelas aktual dari data. Fungsi ini mengukur penalty untuk klasifikasi yang salah, dengan penalty yang lebih besar diberikan untuk prediksi yang jauh dari nilai sebenarnya.

Menambahkan regularization pada loss function dalam regresi logistik bertujuan untuk mencegah overfitting, meningkatkan generalisasi model pada data yang belum dilihat, dan dalam beberapa kasus, mengatasi masalah kolinearitas antar fitur. Overfitting terjadi ketika model terlalu kompleks, menangkap detail dan noise dalam data latih hingga mengorbankan performa pada data uji atau data yang belum dilihat. Regularization mengurangi kompleksitas model tanpa secara signifikan meningkatkan bias, dengan cara menambahkan penalti terhadap besarnya koefisien regresi.

![Loss Function](https://analyticsindiamag.com/wp-content/uploads/2022/07/image-73.png)

## 3. Gradient Descent
Gradient Descent adalah teknik optimasi yang digunakan untuk menemukan nilai parameter (koefisien) model yang meminimalkan fungsi kerugian (loss function). Dalam konteks regresi logistik, tujuannya adalah untuk menemukan set koefisien yang membuat model paling akurat dalam memprediksi label kelas.

![Gradient Descent](https://i.stack.imgur.com/pYVzl.png)

## 4. Pseudo Code

```
class customlogisticregression():
    def __init__(self, learning_rate=0.01, iterations=1000, regularization_strength=0.01):
        initiate learning_rate
        initiate iterations
        initiate regularization_strength
        initiate Weights
        initiate bias
    
    def sigmoid(self, z):
        return sigmoid function
    
    def loss function(self, y_true, y_pred):
        # Avoid log(0) which is undefined
        determine epsilon
        
        # Compute binary cross entropy loss
        create loss function formula
        return loss
    
    def fit(self, X, y):
        
        # Gradient Descent
        for _ in range(iterasi):
            # Compute predictions
            Create logistic model

            # Compute loss
            loss = loss function
            
            # Compute gradients
            compute gradient for Weights
            compute gradient for bias
            
            # Update parameters
            update Weights
            update bias

            # Print loss
            print iteration and loss output
    
    def predict_proba(self, X):
        Calculate prediction probability from updated parameter
        return proba
    
    def predict(self, X):
        transform predict probability to 0 and 1 values
    
    def score(self, X, y=None):
        calculate accuracy score
        return accuracy
```
## 5. Code in Python

Import library
```
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
```

Langkah pertama yang dilakukan adalah dengan melakukan inisiasi pada learning rate, iteration dan regularization strength
```
    def __init__(self, learning_rate=0.01, iterations=1000, regularization_strength=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization_strength = regularization_strength
        self.weights = None
        self.bias = None
```
Langkah kedua adalah dengan membuat fungsi sigmoid, fungsi ini bertujuan untuk melakukan kalkulasi model logistik.
```
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
```

Langkah ketiga menghitung loss function sebagai berikut:
* Menghitunng loss function dengan binary cross entropy loss
* menambahkan epsilon untuk menghindari hasil log(0)
```
    def binary_cross_entropy_loss(self, y_true, y_pred):
        # Avoid log(0) which is undefined
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute binary cross entropy loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
```
Langkah keempat adalah melakukan optimasi melalui gradient descent denga tahapan sebagai berikut:
* Menghitung model prediksi
* menghitung loss function
* menghitung gradient descent dengan menambahkan regularisasi.
* update parameter
* cetak output iterasi dan loss valuenya
```     
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for _ in range(self.iterations):
            # Compute predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute loss
            loss = self.binary_cross_entropy_loss(y, y_predicted)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (self.regularization_strength / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print loss
            if _ % 100 == 0:
                print(f"Iteration {_}, Loss: {loss}")
```
Langkah kelima adalah dengan melakukan perhitungan prediksi probability dan hotung skor akurasinya
```
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        proba = self.sigmoid(linear_model)
        return proba
    
    def predict(self, X):
        return (self.predict_proba(X)>0.5).astype("int")
    
    def score(self, X, y=None):
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy
```

## 6. Implement with dataset
Data set yang digunakan adalah Heart Disease , kita akan lakukan prediksi dengan model regresi logistik yang telah dibuat dan kita juga akan melakukan hyperparameter tuning
dengan kode sebagai berikut:
```
#Implementation
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from customlogreg import customlogisticregression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# import dataset
df = pd.read_csv('D:/Advance Machine Learning/heart.csv')
df.head(6)

# drop missing values
df.dropna()

# Define X and y
X = df.drop('target',axis=1)
y = df['target']

# split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# train model with modification parameter
model = customlogisticregression()
model.fit(X_train,y_train)

# predict model
y_pred = model.predict(X_test)

# show accuracy
print(accuracy_score(y_test,y_pred))

## Hyperparameter TUning
param_grid = {
    'learning_rate': [0.001, 0.01],
    'iterations': [1000, 2000 ]
}

# GridSearchCV for hyperparameter tuning
grid_cv = GridSearchCV(model, param_grid=param_grid, cv=3)
grid_cv.fit(X_train, y_train)

print("Best hyperparameters (GridSearchCV):", grid_cv.best_params_)
print("Best score (GridSearchCV):", grid_cv.best_score_)

# Predict using the best model from GridSearchCV
best_lr_model = grid_cv.best_estimator_
predictions = best_lr_model.predict(X_test)

# Evaluate the model
accuracy = np.sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)
```

## 7. Result
Dengan hyperparameter tuning menggunakan GridSearchCV maka diperoleh best score sebesar 0.60 dan accuracy sebesar 0.56 (akurasi ini tergolong cukup rendah)

```
Best hyperparameters (GridSearchCV): {'iterations': 1000, 'learning_rate': 0.001}
Best score (GridSearchCV): 0.603343621399177
Accuracy: 0.5573770491803278
```

## 8. Conclusion and recommendation
**Konklusi**
* Regresi logistik adalah mdoel untuk memprediksi variabel terikat yang berskala dikotomi.
* Asumsi klasik pada regresi linier tidak diperlukan paa regresi lopgistik.
* Regresi logistik dapat dilakukan optimalisasi dengan cara memodifikasi pada loss function dengan menambahkan epsilon untuk menghindari hasil log0, maupun dengan penambahan penalty berupa regularization dengn tujuan untuk mencegah overfitting, meningkatkan generalisasi model pada data yang belum dilihat, dan dalam beberapa kasus, mengatasi masalah kolinearitas antar fitur.
* Prediksi Heart Disease data dengan hyperparameter tuning menghasilkan nilai akurasi yang cukup rendah.

**Rekomendasi**

Akurasi yang cukup rendah bisa disebabkas oleh beberapa alasan, termasuk pemilihan model maupu tuning hyper parameter tidak optimal. Oleh karena itu, diperlukan berbagai skenario optimasi parameter untuk dapat meningkatkan akurasi model.

Selain itu data juga memiliki nilai pengaruh, Eksplorasi data yang mendalam juga diperlukan untuk mendapatkan hasil yang optimal.

## 9. Referrence

https://www.analyticsvidhya.com/blog/2022/02/implementing-logistic-regression-from-scratch-using-python/

https://medium.com/@koushikkushal95/logistic-regression-from-scratch-dfb8527a4226

https://python.plainenglish.io/logistic-regression-from-scratch-7b707662c8b9

https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression
