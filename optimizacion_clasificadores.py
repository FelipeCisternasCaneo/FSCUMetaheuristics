from mealpy import FloatVar, TransferBoolVar, IntegerVar, StringVar, PSO, Problem


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
    make_scorer
)

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.base import clone
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm
from sklearn import svm #svc.()
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB   
    
    
class seleccion_caracteristicas(Problem):
    
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)
        
    def obj_func(Self, x):
        
        datos = self.data["datos"]
        clases = self.data["clases"]
        clasificador = self.data["clasificador"]
        x_decoded = self.decode_solution(x)
        
        features = x_decoded["features"]
        if sum(features) == 0:
            features = np.random.randint(low=0, high=2, size = (datos.shape[1]))
        selection = np.where(features == 1)[0]
        datos = datos.iloc[:, selection]
        
        X_train, X_val, y_train, y_val = train_test_split(datos, clases, test_size=0.2, stratify=y, random_state=42)
        
        model = None 
        
        if clasificador == 'KNN':
            model = KNeighborsClassifier(
                n_neighbors=x_decoded["n_neighbors"], 
                weights=x_decoded["weights"], 
                p=x_decoded["p"]
                )
        elif clasificador == 'RandomForest':
            model = RandomForestClassifier(
                random_state=42,
                n_estimators=x_decoded["n_estimators_rf"],
                max_depth=x_decoded["max_depth_rf"],
                min_samples_split=x_decoded["min_samples_split"],
                min_samples_leaf=x_decoded["min_samples_leaf_rf"],
                max_features=x_decoded["max_features"]
                )
        elif clasificador == 'LightGBM':
            model =lgbm.LGBMClassifier(random_state=42, 
                                       verbose=-1,
                                       n_estimators=x_decoded["n_estimators_lgbm"],
                                       max_depth=x_decoded["max_depth_lgbm"],
                                       learning_rate=x_decoded["learning_rate"],
                                       min_samples_leaf=x_decoded["min_samples_leaf_lgbm"],
                                       num_leaves=x_decoded["num_leaves"]
                                       )
        elif clasificador == 'SVM':
            model = svm.SVC(kernel="linear",
                            random_state=42,
                            C=x_decoded["C_svm"],
                            loss=x_decoded["loss"],
                            penalty=x_decoded["penalty"]
                            )
        elif clasificador == 'LogisticRegression':
            model = LogisticRegression(max_iter=10000, 
                                       random_state=42,
                                       C=x_decoded["C_lr"],
                                       solver=x_decoded["solver"]
                                       )
        elif clasificador == 'NaiveBayes':
            model = GaussianNB(
                var_smoothing=x_decoded["var_smoothing"]
            )
        
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        mcc_scorer = make_scorer(matthews_corrcoef)
        
        recall = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall", n_jobs=-1)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        resultados_metricas = classification_report(y_val, y_pred, output_dict=True)
        
        return resultados_metricas['0']['recall']
        
#formato
#lista de modelos
clasificadores = ['KNN', 'RandomForest', 'LightGBM', 'SVM', 'LogisticRegression', 'NaiveBayes']
df_final = pd.read_csv('datosFSCUV4_post2012PROCESADO.csv', engine= 'c', index_col=False)
df_final = df_final.drop('id', axis=1)
# df_final

print(df_final["dejo_declarar"].value_counts())

X = df_final.drop(columns="dejo_declarar")
y = df_final["dejo_declarar"]

for clasificador in clasificadores:
    data = {
        "datos": X,
        "clases": y,
        "clasificador": clasificador
    }

    my_bounds = [
    
    # parametros a optimizar clasificadores
    # KNN
    IntegerVar(lb=1, ub=50, name="n_neighbors"),
    StringVar(valid_sets=('uniform', 'distance'), name="weights"),
    IntegerVar(lb=1, ub=2, name="p"),
    
    # RandomForest
    IntegerVar(lb=20, ub=800, name="n_estimators_rf"),
    IntegerVar(lb=1, ub=100, name="max_depth_rf"),
    IntegerVar(lb=2, ub=100, name="min_samples_split_rf"),
    IntegerVar(lb=1, ub=30, name="min_samples_leaf_rf"),
    StringVar(valid_sets=('sqrt', 'log2'), name="max_features_rf"),
    
    # seleccion de características
    TransferBoolVar(n_vars=X.shape[1], name="features", tf_func="sstf_04"),
]


resultados = {}

for nombre, modelo in modelos_base.items():
    models[nombre] = modelo  # sin pipeline

for nombre, pipeline in models.items():
    print(f"\n=== Entrenando {nombre} ===")
    models[nombre], resultados[nombre] = train_cv_model(pipeline, X_train, y_train, nombre)
    print(models[nombre], resultados[nombre])
    
for name,model in models.items():
    print("=========================")
    print(f'\033[1mMetricas de {name}\033[0m')
    y_pred = model.predict(X_val)
    if name != 'SVM':
        y_proba = model.predict_proba(X_val)[:,1]
    else:
        y_proba = None
    reporte_metricas(name, y_val, y_pred, y_prob=y_proba)
    print("\n")