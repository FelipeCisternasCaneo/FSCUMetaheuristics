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

def reporte_metricas(modelo, y_true, y_pred, y_prob=None, pos_label=1):

    print(f"Accuracy:           {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision:          {precision_score(y_true, y_pred, pos_label=pos_label):.4f}")
    print(f"Recall:             {recall_score(y_true, y_pred, pos_label=pos_label):.4f}")
    print(f"F1 Score:           {f1_score(y_true, y_pred, pos_label=pos_label):.4f}")

    if y_prob is not None:
        print(f"AUC:                {roc_auc_score(y_true, y_prob):.4f}")
    else:
        print("AUC:                (no calculado, falta y_prob)")

    print(f"Cohen's Kappa:      {cohen_kappa_score(y_true, y_pred):.4f}")
    print(f"Matthews CorrCoef:  {matthews_corrcoef(y_true, y_pred):.4f}")


    cm = confusion_matrix(y_true, y_pred)

    labels = ['Never Declared', 'Declares']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')  # 'd' para enteros

    plt.title(f"{modelo} Confusion Matrix")
    plt.tight_layout()
    plt.show()

    #print("Matriz de confusión: \n", confusion_matrix(y_true,y_pred))

    print("Valores por clase: \n")
    
    resultados_metricas = classification_report(y_true, y_pred, output_dict=True)
    print(resultados_metricas)
    print(resultados_metricas['0'])
    print(resultados_metricas['1'])
    print(resultados_metricas['macro avg'])
    print(resultados_metricas['weighted avg'])

def train_cv_model(pipeline_aux, X, y, n_splits=10):
    """
    Entrena un pipeline con StratifiedKFold CV y devuelve el modelo entrenado.

    Args:
        pipeline: pipeline de sklearn/imbalanced-learn (ej: SMOTE + LGBM).
        X: features.
        y: target.
        n_splits: número de folds (default=10).
        scoring: métrica para evaluar (default='f1').

    Returns:
        pipeline entrenado con todos los datos.
    """
    start = time.time()

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

    pipeline.fit(X, y)
    
    print("Tiempo transcurrido:", time.time()-start)

    return pipeline, mean_scores



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

resultados = {}

for nombre, modelo in modelos_base.items():
    models[nombre] = modelo  # sin pipeline

for nombre, pipeline in models.items():
    print(f"\n=== Entrenando {nombre} ===")
    models[nombre], resultados[nombre] = train_cv_model(pipeline, X_train, y_train)
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