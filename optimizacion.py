import numpy as np
import pandas as pd
# para division de datos de entrenamiento y testing
from sklearn.model_selection import train_test_split
# para división del conjunto de entramiento en k-folds
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

        # X son tus características, y tus etiquetas
        X_train_val, X_test, y_train_val, y_test = train_test_split(datos, clases, test_size=0.30, random_state=42)
        
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

        for train_index, test_index in kf.split(X_train_val,y_train_val):
            # obtenemos los datos de entrenamiento y datos de testing del fold
            X_train_fold, X_test_fold = X_train_val.iloc[train_index], X_train_val.iloc[test_index]
            y_train_fold, y_test_fold = y_train_val.iloc[train_index], y_train_val.iloc[test_index]
            
            
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
            print(reporte)
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
        
        # Usa el 'mejor_modelo' para predecir sobre el 30% (Test Set)
        y_pred = model.predict(X_test)

        # Calcula el rendimiento final UNA SOLA VEZ
        print("\nRendimiento en el Conjunto de Prueba (30%):")
        reporte_testing = classification_report(y_test, y_pred, output_dict=True)
        precision_score_0_testing = reporte_testing['0']['precision']
        precision_score_1_testing = reporte_testing['1']['precision']
        precision_score_macro_testing = reporte_testing['macro avg']['precision']
        precision_score_weighted_testing = reporte_testing['weighted avg']['precision']

        recall_score_0_testing = reporte_testing['0']['recall']
        recall_score_1_testing = reporte_testing['1']['recall']
        recall_score_macro_testing = reporte_testing['macro avg']['recall']
        recall_score_weighted_testing = reporte_testing['weighted avg']['recall']

        fscore_score_0_testing = reporte_testing['0']['f1-score']
        fscore_score_1_testing = reporte_testing['1']['f1-score']
        fscore_score_macro_testing = reporte_testing['macro avg']['f1-score']
        fscore_score_weighted_testing = reporte_testing['weighted avg']['f1-score']
        
        
        resultados.append({
            "corrida": i,
            "precision 0 train": np.mean(precision_score_0),
            "recall 0 train": np.mean(recall_score_0),
            "fscore 0 train": np.mean(fscore_score_0),
            "precision 1 train": np.mean(precision_score_1),
            "recall 1 train": np.mean(recall_score_1),
            "fscore 1 train": np.mean(fscore_score_1),
            "precision macro train": np.mean(precision_score_macro),
            "recall macro train": np.mean(recall_score_macro),
            "fscore macro train": np.mean(fscore_score_macro),
            "precision weighted train": np.mean(precision_score_weighted),
            "recall weighted train": np.mean(recall_score_weighted),
            "fscore weighted train": np.mean(fscore_score_weighted),
            "precision 0 test": precision_score_0_testing,
            "recall 0 test": recall_score_0_testing,
            "fscore 0 test": fscore_score_0_testing,
            "precision 1 test": precision_score_1_testing,
            "recall 1 test": recall_score_1_testing,
            "fscore 1 test": fscore_score_1_testing,
            "precision macro test": precision_score_macro_testing,
            "recall macro test": recall_score_macro_testing,
            "fscore macro test": fscore_score_macro_testing,
            "precision weighted test": precision_score_weighted_testing,
            "recall weighted test": recall_score_weighted_testing,
            "fscore weighted test": fscore_score_weighted_testing
        })

    resultados_df = pd.DataFrame(resultados)
    print(resultados_df)
    resultados_df.to_csv(f'Resultados_Base/resultados_{clasificador}.csv', index=False)