from mealpy import FloatVar, TransferBoolVar, IntegerVar, StringVar, PSO, Problem, GWO, WOA
from pathlib import Path

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
    
    
def reporte_metricas(ejecucion, nombre, mh, y_true, y_pred, y_prob=None, pos_label=1):

    resultados_testing_generales = []
    resultados_testing_generales.append({
        "metric": 'Accuracy',
        "value": accuracy_score(y_true, y_pred)
    })
    resultados_testing_generales.append({
        "metric": 'Precision',
        "value": precision_score(y_true, y_pred, pos_label=pos_label)
    })
    resultados_testing_generales.append({
        "metric": 'Recall',
        "value": recall_score(y_true, y_pred, pos_label=pos_label)
    })
    resultados_testing_generales.append({
        "metric": 'F1 Score',
        "value": f1_score(y_true, y_pred, pos_label=pos_label)
    })

    if y_prob is not None:
        resultados_testing_generales.append({
            "metric": 'AUC',
            "value": roc_auc_score(y_true, y_prob)
        })
    else:
        resultados_testing_generales.append({
            "metric": 'AUC',
            "value": None
        })
    resultados_testing_generales.append({
        "metric": 'Cohen\'s Kappa',
        "value": cohen_kappa_score(y_true, y_pred)
    })
    resultados_testing_generales.append({
        "metric": 'Matthews CorrCoef',
        "value": matthews_corrcoef(y_true, y_pred)
    })

    resultados_testing_generales_df = pd.DataFrame(resultados_testing_generales)
    resultados_testing_generales_df.to_csv(f'./resultados/seleccion_caracteristicas/{mh}/{nombre}/{ejecucion}/resultados_generales_{nombre}_{mh}.csv', index=False)

    cm = confusion_matrix(y_true, y_pred)

    labels = ['Dejó de Declarar', 'Declaró']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')  # 'd' para enteros

    plt.title(f"{nombre} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f'./resultados/seleccion_caracteristicas/{mh}/{nombre}/{ejecucion}/confusion_matrix_{nombre}_{mh}.png')
    plt.close()

    resultados_testing_clases = []
    
    resultados_metricas = classification_report(y_true, y_pred, output_dict=True)
    
    resultados_testing_clases.append({
        "class": '0',
        "metric": 'presicion',
        "value": resultados_metricas['0']['precision']
    })
    resultados_testing_clases.append({
        "class": '1',
        "metric": 'presicion',
        "value": resultados_metricas['1']['precision']
    })
    resultados_testing_clases.append({
        "class": 'macro avg',
        "metric": 'presicion',
        "value": resultados_metricas['macro avg']['precision']
    })
    resultados_testing_clases.append({
        "class": 'weighted avg',
        "metric": 'presicion',
        "value": resultados_metricas['weighted avg']['precision']
    })
    

    resultados_testing_clases.append({
        "class": '0',
        "metric": 'recall',
        "value": resultados_metricas['0']['recall']
    })
    resultados_testing_clases.append({
        "class": '1',
        "metric": 'recall',
        "value": resultados_metricas['1']['recall']
    })
    
    resultados_testing_clases.append({
        "class": 'macro avg',
        "metric": 'recall',
        "value": resultados_metricas['macro avg']['recall']
    })
    resultados_testing_clases.append({
        "class": 'weighted avg',
        "metric": 'recall',
        "value": resultados_metricas['weighted avg']['recall']
    })
    
    
    
    resultados_testing_clases.append({
        "class": '0',
        "metric": 'f1-score',
        "value": resultados_metricas['0']['f1-score']
    })
    resultados_testing_clases.append({
        "class": '1',
        "metric": 'f1-score',
        "value": resultados_metricas['1']['f1-score']
    })
    resultados_testing_clases.append({
        "class": 'macro avg',
        "metric": 'f1-score',
        "value": resultados_metricas['macro avg']['f1-score']
    })
    resultados_testing_clases.append({
        "class": 'weighted avg',
        "metric": 'f1-score',
        "value": resultados_metricas['weighted avg']['f1-score']
    })
    
    resultados_testing_clases_df = pd.DataFrame(resultados_testing_clases)
    resultados_testing_clases_df.to_csv(f'./resultados/seleccion_caracteristicas/{mh}/{nombre}/{ejecucion}/resultados_clases_{nombre}_{mh}.csv', index=False)
    
class seleccion_caracteristicas(Problem):
    
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)
        
    def obj_func(self, x):
        
        datos = self.data["datos"]
        clases = self.data["clases"]
        clasificador = self.data["clasificador"]
        x_decoded = x > 0.5
        # x_decoded es un array booleano, np.where funciona igual
        selection = np.where(x_decoded)[0]
        # Validación de seguridad: Si no seleccionó ninguna, penalizamos fuerte
        if len(selection) == 0:
            return 0.0 # El peor caso (Recall 0 -> fitness 1)
        datos_filtrados = datos.iloc[:, selection]
        
        X_train, X_val, y_train, y_val = train_test_split(datos_filtrados, clases, test_size=0.2, stratify=clases, random_state=42)
        
        model = None 
        
        if clasificador == 'KNN':
            model = KNeighborsClassifier()
        elif clasificador == 'RandomForest':
            model = RandomForestClassifier(random_state=42)
        elif clasificador == 'LightGBM':
            model = lgbm.LGBMClassifier(random_state=42, verbose=-1)
        elif clasificador == 'SVM':
            model = svm.SVC(kernel="linear",random_state=42)
        elif clasificador == 'LogisticRegression':
            model = LogisticRegression(max_iter=10000, random_state=42)
        elif clasificador == 'NaiveBayes':
            model = GaussianNB()
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        resultados_metricas = classification_report(y_val, y_pred, output_dict=True)
        return resultados_metricas['0']['recall']
        
#formato
#lista de modelos
clasificadores = ['KNN', 'RandomForest', 'LightGBM', 'SVM', 'LogisticRegression', 'NaiveBayes']
mhs = ['PSO','GWO','WOA']
df_final = pd.read_csv('datosFSCUV4_post2012PROCESADO.csv', engine= 'c', index_col=False)
df_final = df_final.drop('id', axis=1)

print(df_final["dejo_declarar"].value_counts())

X = df_final.drop(columns="dejo_declarar")
y = df_final["dejo_declarar"]

for mh in mhs:
    carpeta_mh = Path(f"resultados/seleccion_caracteristicas/{mh}")
    carpeta_mh.mkdir(parents=True, exist_ok=True)
    for c in clasificadores:
        carpeta_clasificador = Path(f"resultados/seleccion_caracteristicas/{mh}/{c}")
        carpeta_clasificador.mkdir(parents=True, exist_ok=True)
        
        corridas = 1
        while corridas <= 5:
            carpeta_corrida = Path(f"resultados/seleccion_caracteristicas/{mh}/{c}/{corridas}")
            carpeta_corrida.mkdir(parents=True, exist_ok=True)
        
            data = {
                "datos": X,
                "clases": y,
                "clasificador": c
            }

            my_bounds = [   
                # seleccion de características
                FloatVar(lb=[0.0] * X.shape[1], ub=[1.0] * X.shape[1], name="features"),
            ]


            problem = seleccion_caracteristicas(bounds=my_bounds, minmax="max", data=data)
            optimizador = None
            if mh == 'PSO':
                optimizador = PSO.OriginalPSO(epoch=10, pop_size=5)
            elif mh == 'GWO':
                optimizador = GWO.OriginalGWO(epoch=10, pop_size=5)
            elif mh == 'WOA':
                optimizador = WOA.OriginalWOA(epoch=10, pop_size=5)
            optimizador.solve(problem)
        
            
            soluciones_reporte = []        
            best = optimizador.g_best.solution
            soluciones_reporte.append({
                "solucion": 'continua',
                "valor": best.tolist(),
            })
            x_decoded = best > 0.5
            soluciones_reporte.append({
                "solucion": 'binaria',
                "valor": x_decoded.tolist(),
            })
            
            soluciones_reporte_df = pd.DataFrame(soluciones_reporte)
            soluciones_reporte_df.to_csv(f'./resultados/seleccion_caracteristicas/{mh}/{c}/{corridas}/solucion_{c}_{mh}.csv', index=False)
            
            
            
            # x_decoded es un array booleano, np.where funciona igual
            selection = np.where(x_decoded)[0]
            X_filtrados = X.iloc[:, selection]
            
            X_train, X_val, y_train, y_val = train_test_split(X_filtrados, y, test_size=0.2, stratify=y, random_state=42)
            
            model = None 
                
            if c == 'KNN':
                model = KNeighborsClassifier()
            elif c == 'RandomForest':
                model = RandomForestClassifier(random_state=42)
            elif c == 'LightGBM':
                model = lgbm.LGBMClassifier(random_state=42, verbose=-1)
            elif c == 'SVM':
                model = svm.SVC(kernel="linear",random_state=42)
            elif c == 'LogisticRegression':
                model = LogisticRegression(max_iter=10000, random_state=42)
            elif c == 'NaiveBayes':
                model = GaussianNB()
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            
            reporte_metricas(corridas, c, mh, y_val, y_pred, y_prob=None, pos_label=1)
            
            corridas+=1
    