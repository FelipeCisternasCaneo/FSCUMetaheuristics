import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
# tecnicas de over sampling
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
# tecnicas de under sampling
from imblearn.under_sampling import NeighbourhoodCleaningRule, RandomUnderSampler, EditedNearestNeighbours, AllKNN, NearMiss
# tecnicas de hibridas
from imblearn.combine import SMOTEENN
# clasificadores
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

from mealpy import FloatVar, TransferBoolVar, SMA, Problem


# Lectura de datos
df = pd.read_csv('datosFSCUV4_post2012PROCESADO.csv', engine= 'c', index_col=False)
df = df.drop('id', axis=1)

datos = df.loc[:, df.columns != 'dejo_declarar']
clases = df['dejo_declarar']

clasificadores = ['XGBoost', 'RandomForest', 'KNN', 'GradientBoosting', 'LGBM', 'ExtraTrees', 'AdaBoost', 'SVM']

print(datos)
print(clases)

for clasificador in clasificadores:

    resultados = []

    for i in range(31):

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        if clasificador == 'XGBoost':
            model = XGBClassifier(objective='binary:hinge')
        elif clasificador == 'RandomForest':
            model = RandomForestClassifier(random_state=42)
        elif clasificador == 'KNN':
            model = KNeighborsClassifier()
        elif clasificador == 'GradientBoosting':
            model = GradientBoostingClassifier(random_state=42)
        elif clasificador == 'LGBM':
            model = LGBMClassifier(objective='binary', random_state=42)
        elif clasificador == 'ExtraTrees':
            model = ExtraTreesClassifier(random_state=42)
        elif clasificador == 'AdaBoost':
            model = AdaBoostClassifier(random_state=42)
        elif clasificador == 'SVM':
            model = SVC(random_state=42)
        

        scaler = MinMaxScaler()

        # Initialize the metrics
        precision_score_0           = []
        precision_score_1           = []
        precision_score_macro       = []
        precision_score_weighted    = []

        recall_score_0           = []
        recall_score_1           = []
        recall_score_macro       = []
        recall_score_weighted    = []

        fscore_score_0           = []
        fscore_score_1           = []
        fscore_score_macro       = []
        fscore_score_weighted    = []
        
        print(f"Corrida {i} para algoritmo {clasificador}")

        for train_index, test_index in kf.split(datos,clases):
            # obtenemos los datos de entrenamiento y datos de testing del fold
            X_train_fold, X_test_fold = datos.iloc[train_index], datos.iloc[test_index]
            y_train_fold, y_test_fold = clases.iloc[train_index], clases.iloc[test_index]
            
            
            X_train_fold_procesado  = scaler.fit_transform(X_train_fold)
            y_train_fold_procesado  = y_train_fold
            
            # entrenamos el modelo con los datos de entrenamiento normalizados
            model.fit(X_train_fold_procesado, y_train_fold_procesado)
            
            
            # normalizamos los datos de testing
            X_test_fold_norm = scaler.transform(X_test_fold)
            # hacemos la predicción de las clases con los datos de testing
            y_pred = model.predict(X_test_fold_norm)
            # comparamos los resultados obtenidos entre las clases predichas y las clases originales de testing
            reporte = classification_report(y_test_fold, y_pred, output_dict=True)
            # obtenemos las métricas de desempeño
            precision_score_0.append(reporte['0']['precision'])
            precision_score_1.append(reporte['1']['precision'])
            precision_score_macro.append(reporte['macro avg']['precision'])
            precision_score_weighted.append(reporte['weighted avg']['precision'])

            recall_score_0.append(reporte['0']['recall'])
            recall_score_1.append(reporte['1']['recall'])
            recall_score_macro.append(reporte['macro avg']['recall'])
            recall_score_weighted.append(reporte['weighted avg']['recall'])

            fscore_score_0.append(reporte['0']['f1-score'])
            fscore_score_1.append(reporte['1']['f1-score'])
            fscore_score_macro.append(reporte['macro avg']['f1-score'])
            fscore_score_weighted.append(reporte['weighted avg']['f1-score'])

        resultados.append({
            "corrida": i,
            "precision 0": np.mean(precision_score_0),
            "recall 0": np.mean(recall_score_0),
            "fscore 0": np.mean(fscore_score_0),
            "precision 1": np.mean(precision_score_1),
            "recall 1": np.mean(recall_score_1),
            "fscore 1": np.mean(fscore_score_1),
            "precision macro": np.mean(precision_score_macro),
            "recall macro": np.mean(recall_score_macro),
            "fscore macro": np.mean(fscore_score_macro),
            "precision weighted": np.mean(precision_score_weighted),
            "recall weighted": np.mean(recall_score_weighted),
            "fscore weighted": np.mean(fscore_score_weighted)
        })

    resultados_df = pd.DataFrame(resultados)
    print(resultados_df)
    resultados_df.to_csv(f'Resultados_Base/resultados_{clasificador}.csv', index=False)