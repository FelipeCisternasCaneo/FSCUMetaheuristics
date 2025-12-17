import subprocess
import time
import sys
from pathlib import Path

# Lista de configuraciones
# clasificadores = ['KNN', 'LightGBM', 'SVM', 'LogisticRegression', 'NaiveBayes'] # Agrega los que quieras
clasificadores = ['RandomForest']
#clasificadores = ['LightGBM']
#clasificadores = ['SVM']
#clasificadores = ['LogisticRegression']
#clasificadores = ['NaiveBayes'] # Agrega los que quieras
mhs = ['PSO', 'GWO', 'WOA']

if __name__ == "__main__":
    for mh in mhs:
        for c in clasificadores:
            corridas = 1
            while corridas <= 31:
                carpeta_corrida = Path(f"resultados/seleccion_caracteristicas/{mh}/{c}/{corridas}")
                carpeta_corrida.mkdir(parents=True, exist_ok=True)
                print(f"\n===========================================================")
                print(f" LANZANDO PROCESO NUEVO: MH={mh} | CLF={c} | Corrida={corridas}")
                print(f"=============================================================")
                
                # Llamada al sistema operativo para crear un proceso Python independiente
                # Esto garantiza que al terminar, la RAM vuelve a 0 bytes usados.
                try:
                    subprocess.run(
                        [sys.executable, "seleccion_caracteristicas.py", "--mh", mh, "--clf", c, "--corrida", str(corridas)],
                        check=True
                    )
                except subprocess.CalledProcessError as e:
                    print(f"ERROR CRITICO en {mh}-{c}: {e}")
                
                # Pequena pausa para asegurar que Windows cierre los handles de archivos
                print("Limpieza de sistema operativo...")
                time.sleep(3)
                corridas+=1