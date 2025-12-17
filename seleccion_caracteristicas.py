import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend no interactivo vital
import matplotlib.pyplot as plt
from pathlib import Path
from mealpy import FloatVar, PSO, GWO, WOA, Problem
from sklearn.metrics import recall_score, make_scorer, precision_score, accuracy_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import gc

# Definimos explícitamente cuál es tu clase minoritaria
# (Revisa tus datos: ¿es 0 o 1?)
CLASE_MINORITARIA = 0

# Creamos un "Scorer" personalizado
# pos_label le dice a la métrica qué clase mirar
scorer_minoritaria = make_scorer(recall_score, pos_label=CLASE_MINORITARIA)

# --- DEFINICIÓN DEL PROBLEMA ---
class SeleccionCaracteristicas(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        datos = self.data["datos"]
        clases = self.data["clases"]
        clasificador_nombre = self.data["clasificador"]
        
        # Decodificación determinista
        selection = np.where(x > 0.5)[0]
        
        if len(selection) == 0:
            return 0.0
        datos_filtrados = datos.iloc[:, selection]
        
        # IMPORTANTE: Reducir overhead en obj_func
        # Usar un split simple 80/20 dentro de la optimización es MUCHO más rápido que CV
        # El CV (Cross Validation) déjalo solo para la validación final.
        X_t, X_v, y_t, y_v = train_test_split(datos_filtrados, clases, test_size=0.2, stratify=clases, random_state=42)
        
        model = self._get_model(clasificador_nombre)
        
        cv_strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
        # Usamos nuestro scorer personalizado
        scores = cross_val_score(
            model, 
            X_t, 
            y_t, 
            cv=cv_strat, 
            scoring=scorer_minoritaria # <--- Aquí está la clave
        )
        return scores.mean()
        
        
        # model.fit(X_t, y_t)
        # y_pred = model.predict(X_v)
        
        # Optimizamos Recall de la clase minoritaria (0)
        # return recall_score(y_v, y_pred, pos_label=0)

    def _get_model(self, name):
        # Instanciar modelos ligeros para la optimización
        if name == 'KNN': return KNeighborsClassifier()
        if name == 'RandomForest': return RandomForestClassifier(random_state=42, n_jobs=-1) # Pocos arboles para ser rápido
        if name == 'LightGBM': return lgbm.LGBMClassifier(random_state=42, verbose=-1)
        if name == 'SVM': return svm.SVC(kernel="linear", random_state=42)
        if name == 'LogisticRegression': return LogisticRegression(max_iter=10000, random_state=42)
        if name == 'NaiveBayes': return GaussianNB()
        return None

# --- FUNCIÓN DE REPORTE ---
def generar_reporte(ejecucion, mh, nombre, y_true, y_pred, pos_label=1):

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
    plt.close("all")

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

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mh", type=str, required=True)
    parser.add_argument("--clf", type=str, required=True)
    parser.add_argument("--corrida", type=str, required=True)
    args = parser.parse_args()

    mh = args.mh
    clf = args.clf
    ejecucion = args.corrida

    print(f"--- WORKER INICIADO: {mh} + {clf} + corrida:{ejecucion} ---")
    
    # CARGA DE DATOS (Se hace fresca cada vez)
    try:
        df_final = pd.read_csv('datosFSCUV4_post2012PROCESADO.csv', index_col=False)
        if 'id' in df_final.columns: df_final = df_final.drop('id', axis=1)
        X = df_final.drop(columns="dejo_declarar")
        y = df_final["dejo_declarar"]
    except Exception as e:
        print(f"Error cargando datos: {e}")
        sys.exit(1)


    # CONFIGURAR MEALPY
    data_dict = {"datos": X, "clases": y, "clasificador": clf}
    bounds = [FloatVar(lb=[0.0]*X.shape[1], ub=[1.0]*X.shape[1], name="features")]
    problem = SeleccionCaracteristicas(bounds=bounds, minmax="max", data=data_dict)

    model_mh = None
    if mh == 'PSO': model_mh = PSO.OriginalPSO(epoch=100, pop_size=10) # Ajusta epoch/pop
    elif mh == 'GWO': model_mh = GWO.OriginalGWO(epoch=100, pop_size=10)
    elif mh == 'WOA': model_mh = WOA.OriginalWOA(epoch=100, pop_size=10)

    # RESOLVER
    model_mh.solve(problem)
    
    # VALIDACIÓN FINAL (Aquí sí usamos el modelo robusto)
    best_mask = model_mh.g_best.solution > 0.5
    selection = np.where(best_mask)[0]
    
    if len(selection) > 0:
    
        soluciones_reporte = []        
        soluciones_reporte.append({
            "solucion": 'continua',
            "valor": model_mh.g_best.solution.tolist(),
        })
        soluciones_reporte.append({
            "solucion": 'binaria',
            "valor": best_mask.tolist(),
        })
        soluciones_reporte_df = pd.DataFrame(soluciones_reporte)
        soluciones_reporte_df.to_csv(f'./resultados/seleccion_caracteristicas/{mh}/{clf}/{ejecucion}/solucion_{clf}_{mh}.csv', index=False)
        
        X_filt = X.iloc[:, selection]
        X_tr, X_val, y_tr, y_val = train_test_split(X_filt, y, test_size=0.2, stratify=y, random_state=42)
        
        # Modelo final robusto
        final_model = None 
                
        if clf == 'KNN':
            final_model = KNeighborsClassifier()
        elif clf == 'RandomForest':
            final_model = RandomForestClassifier(random_state=42)
        elif clf == 'LightGBM':
            final_model = lgbm.LGBMClassifier(random_state=42, verbose=-1)
        elif clf == 'SVM':
            final_model = svm.SVC(kernel="linear",random_state=42)
        elif clf == 'LogisticRegression':
            final_model = LogisticRegression(max_iter=10000, random_state=42)
        elif clf == 'NaiveBayes':
            final_model = GaussianNB()
            
        cv_strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Usamos nuestro scorer personalizado
        scores = cross_val_score(
            final_model, 
            X_tr, 
            y_tr, 
            cv=cv_strat, 
            scoring=scorer_minoritaria # <--- Aquí está la clave
        )
        
        reporte_entrenamiento = []        
        reporte_entrenamiento.append({
            "observacion": 'k-folds',
            "valor": scores.tolist(),
        })
        reporte_entrenamiento.append({
            "observacion": 'mean',
            "valor": scores.mean(),
        })
        
        reporte_entrenamiento_df = pd.DataFrame(reporte_entrenamiento)
        reporte_entrenamiento_df.to_csv(f'./resultados/seleccion_caracteristicas/{mh}/{clf}/{ejecucion}/reporte_entrenamiento_{clf}_{mh}.csv', index=False)
        
        final_model.fit(X_tr, y_tr)
        y_pred = final_model.predict(X_val)
        
        generar_reporte(ejecucion, mh, clf, y_val, y_pred)
        print(f"--- WORKER FINALIZADO CON ÉXITO: {mh} + {clf} ---")
    else:
        print("Ninguna característica seleccionada.")