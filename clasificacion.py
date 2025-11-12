from experimentacion_v2 import reporte_metricas, train_cv_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm
from sklearn import svm #svc.()
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd

#formato
#lista de modelos
modelos_base = {
    "KNN": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(random_state=42),
    "LightGBM": lgbm.LGBMClassifier(random_state=42, verbose=-1),
    "SVM": svm.SVC(kernel="linear",random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=10000, random_state=42),
    "NaiveBayes": GaussianNB()
}

models = { "KNN": None, "RandomForest": None, "LightGBM": None, "SVM": None, "LogisticRegression": None, "NaiveBayes": None
}
models_smote = { "KNN": None, "RandomForest": None, "LightGBM": None, "SVM": None, "LogisticRegression": None, "NaiveBayes": None
}
models_adasyn = { "KNN": None, "RandomForest": None, "LightGBM": None, "SVM": None, "LogisticRegression": None, "NaiveBayes": None
}
models_rus = { "KNN": None, "RandomForest": None, "LightGBM": None, "SVM": None, "LogisticRegression": None, "NaiveBayes": None
}
#función para recibir dict de modelos
#dict = crear_modelos(pipeline, df)

#función para generar reporte a partir de modelo
#for dict, index  in modelos:
#   resultados(dict, index)

df_final = pd.read_csv('datosFSCUV4_post2012PROCESADO.csv', engine= 'c', index_col=False)
df_final = df_final.drop('id', axis=1)


X = df_final.drop(columns="dejo_declarar")
y = df_final["dejo_declarar"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(X_train.shape, X_val.shape)