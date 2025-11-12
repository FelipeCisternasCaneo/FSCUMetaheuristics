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


def reporte_metricas(nombre, y_true, y_pred, y_prob=None, pos_label=1):

    resultados_testing_generales = []

    print(f"Accuracy:           {accuracy_score(y_true, y_pred):.4f}")
    resultados_testing_generales.append({
        "metric": 'Accuracy',
        "value": accuracy_score(y_true, y_pred)
    })
    print(f"Precision:          {precision_score(y_true, y_pred, pos_label=pos_label):.4f}")
    resultados_testing_generales.append({
        "metric": 'Precision',
        "value": precision_score(y_true, y_pred, pos_label=pos_label)
    })
    print(f"Recall:             {recall_score(y_true, y_pred, pos_label=pos_label):.4f}")
    resultados_testing_generales.append({
        "metric": 'Recall',
        "value": recall_score(y_true, y_pred, pos_label=pos_label)
    })
    print(f"F1 Score:           {f1_score(y_true, y_pred, pos_label=pos_label):.4f}")
    resultados_testing_generales.append({
        "metric": 'F1 Score',
        "value": f1_score(y_true, y_pred, pos_label=pos_label)
    })

    if y_prob is not None:
        print(f"AUC:                {roc_auc_score(y_true, y_prob):.4f}")
        resultados_testing_generales.append({
            "metric": 'AUC',
            "value": roc_auc_score(y_true, y_prob)
        })
    else:
        print("AUC:                (no calculado, falta y_prob)")
        resultados_testing_generales.append({
            "metric": 'AUC',
            "value": None
        })

    print(f"Cohen's Kappa:      {cohen_kappa_score(y_true, y_pred):.4f}")
    resultados_testing_generales.append({
        "metric": 'Cohen\'s Kappa',
        "value": cohen_kappa_score(y_true, y_pred)
    })
    print(f"Matthews CorrCoef:  {matthews_corrcoef(y_true, y_pred):.4f}")
    resultados_testing_generales.append({
        "metric": 'Matthews CorrCoef',
        "value": matthews_corrcoef(y_true, y_pred)
    })

    resultados_testing_generales_df = pd.DataFrame(resultados_testing_generales)
    resultados_testing_generales_df.to_csv(f'./resultados/base/testing/resultados_generales_{nombre}.csv', index=False)

    cm = confusion_matrix(y_true, y_pred)

    labels = ['Dejó de Declarar', 'Declaró']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')  # 'd' para enteros

    plt.title(f"{nombre} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f'./resultados/base/testing/confusion_matrix_{nombre}.png')
    plt.close()

    #print("Matriz de confusión: \n", confusion_matrix(y_true,y_pred))

    print("Valores por clase: \n")

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
    resultados_testing_clases_df.to_csv(f'./resultados/base/testing/resultados_clases_{nombre}.csv', index=False)
    
def train_cv_model(pipeline_aux, X, y, nombre, n_splits=10):
    """
    Entrena un pipeline con StratifiedKFold CV y devuelve el modelo entrenado.

    Args:
        pipeline: pipeline de sklearn/imbalanced-learn (ej: SMOTE + LGBM).
        X: features.
        y: target.
        nombre: nomnbre del modelo a utilizar
        n_splits: número de folds (default=10).
        scoring: métrica para evaluar (default='f1').

    Returns:
        pipeline entrenado con todos los datos.
    """
    start = time.time()

    registro = []

    pipeline = clone(pipeline_aux)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    mcc_scorer = make_scorer(matthews_corrcoef)

    mcc = cross_val_score(pipeline, X, y, cv=cv, scoring=mcc_scorer, n_jobs=-1)
    f1 = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)
    accuracy = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    precision = cross_val_score(pipeline, X, y, cv=cv, scoring="precision", n_jobs=-1)
    recall = cross_val_score(pipeline, X, y, cv=cv, scoring="recall", n_jobs=-1)

    mean_scores = {}
    mean_scores["mcc"] = np.mean(mcc)
    mean_scores["f1"] = np.mean(f1)
    mean_scores["acc"] = np.mean(accuracy)
    mean_scores["prc"] = np.mean(precision)
    mean_scores["rec"] = np.mean(recall)

    print(f"MCC en cada fold: {mcc}")
    print(f"Promedio accuracy: {np.mean(accuracy):.4f}")
    print(f"Promedio precision: {np.mean(precision):.4f}")
    print(f"Promedio recall: {np.mean(recall):.4f}")
    print(f"Promedio MCC: {np.mean(mcc):.4f}")
    print(f"Promedio fscore: {np.mean(f1):.4f}")
    
    registro.append({
        "metric": 'mcc',
        "fold": mcc.tolist(),
        "avg": np.mean(mcc)
    })
    
    registro.append({
        "metric": 'f1',
        "fold": f1.tolist(),
        "avg": np.mean(f1)
    })
    
    registro.append({
        "metric": 'accuracy',
        "fold": accuracy.tolist(),
        "avg": np.mean(accuracy)
    })
    
    registro.append({
        "metric": 'precision',
        "fold": precision.tolist(),
        "avg": np.mean(precision)
    })
    
    registro.append({
        "metric": 'recall',
        "fold": recall.tolist(),
        "avg": np.mean(recall)
    })

    registro_df = pd.DataFrame(registro)
    registro_df.to_csv(f'./resultados/base/training/resultados_{nombre}.csv', index=False)

    pipeline.fit(X, y)

    print("Tiempo transcurrido:", time.time()-start)

    return pipeline, mean_scores


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

df_final = pd.read_csv('datosFSCUV4_post2012PROCESADO.csv', engine= 'c', index_col=False)
df_final = df_final.drop('id', axis=1)
# df_final

print(df_final["dejo_declarar"].value_counts())

X = df_final.drop(columns="dejo_declarar")
y = df_final["dejo_declarar"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(X_train.shape, X_val.shape)

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