import pandas as pd
import numpy as np
import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt

def generacion_data():

    directorio_base = './resultados/seleccion_caracteristicas'

    mhs = os.listdir(directorio_base)
    clasificadores = os.listdir(f'{directorio_base}/{mhs[0]}')
    ejecuciones = os.listdir(f'{directorio_base}/{mhs[0]}/{clasificadores[0]}')

    diccionario_0 = []
    diccionario_1 = []
    diccionario_macro_avg = []
    diccionario_weighted_avg = []

    for mh in mhs:
        
        for clasificador in clasificadores:
            
            for ejecucion in ejecuciones:
                
                data = pd.read_csv(f'{directorio_base}/{mh}/{clasificador}/{ejecucion}/resultados_clases_{clasificador}_{mh}.csv')
                
                print(f'datos de {mh} {clasificador}')
                for index,fila in data.iterrows():
                    # Accedes con punto, no con corchetes                
                    if fila['class'] == '0':
                        diccionario_0.append({
                            'mh': mh,
                            'clasificador': clasificador,
                            'metric': fila['metric'],
                            'value': fila['value']
                        })
                    elif fila['class'] == '1':
                        diccionario_1.append({
                            'mh': mh,
                            'clasificador': clasificador,
                            'metric': fila['metric'],
                            'value': fila['value']
                        })
                    elif fila['class'] == 'macro avg':
                        diccionario_macro_avg.append({
                            'mh': mh,
                            'clasificador': clasificador,
                            'metric': fila['metric'],
                            'value': fila['value']
                        })
                    elif fila['class'] == 'weighted avg':
                        diccionario_weighted_avg.append({
                            'mh': mh,
                            'clasificador': clasificador,
                            'metric': fila['metric'],
                            'value': fila['value']
                        })

    diccionario_0_df = pd.DataFrame(diccionario_0)
    diccionario_1_df = pd.DataFrame(diccionario_1)
    diccionario_macro_avg_df = pd.DataFrame(diccionario_macro_avg)
    diccionario_weighted_avg_df = pd.DataFrame(diccionario_weighted_avg)

    diccionario_0_df.to_csv(f'./resultados/data/rendimiento/diccionario_0.csv', index=False)
    diccionario_1_df.to_csv(f'./resultados/data/rendimiento/diccionario_1.csv', index=False)
    diccionario_macro_avg_df.to_csv(f'./resultados/data/rendimiento/diccionario_macro_avg.csv', index=False)
    diccionario_weighted_avg_df.to_csv(f'./resultados/data/rendimiento/diccionario_weighted_avg.csv', index=False)
    

def generacion_data_soluciones():
    
    directorio_base = './resultados/seleccion_caracteristicas'

    mhs = os.listdir(directorio_base)
    clasificadores = os.listdir(f'{directorio_base}/{mhs[0]}')
    ejecuciones = os.listdir(f'{directorio_base}/{mhs[0]}/{clasificadores[0]}')

    print(mhs)
    print(clasificadores)
    print(ejecuciones)
    
    soluciones_consolidado = []

    for mh in mhs:
        
        for clasificador in clasificadores:
            soluciones = np.zeros(76)
            for ejecucion in ejecuciones:
                
                data = pd.read_csv(f'{directorio_base}/{mh}/{clasificador}/{ejecucion}/solucion_{clasificador}_{mh}.csv')
                
                for index,fila in data.iterrows():
                    if fila['solucion'] == 'binaria':
                        
                        # 1. Convertimos el texto a lista (igual que antes)
                        lista_real = ast.literal_eval(fila['valor'])

                        # 2. Creamos el arreglo y convertimos a ENTEROS (.astype(int))
                        # Aquí ocurre la magia: True -> 1, False -> 0
                        arreglo_binario = np.array(lista_real).astype(int)                        
                        soluciones += arreglo_binario
                        
                        
            print(f'soluciones para {mh} {clasificador}: {soluciones}')
            soluciones_consolidado.append({
                'mh': mh,
                'clasificador': clasificador,
                'soluciones': soluciones.tolist()
            })

    soluciones_consolidado_df = pd.DataFrame(soluciones_consolidado)
    soluciones_consolidado_df.to_csv(f'./resultados/data/caracteristicas/soluciones_consolidado.csv', index=False)
   
   
   
def generacion_heatmap():
    directorio_base = './resultados/data/caracteristicas'
    soluciones_consolidado = pd.read_csv(f'{directorio_base}/soluciones_consolidado.csv')
    mhs = soluciones_consolidado['mh'].unique()
    clasificadores = soluciones_consolidado['clasificador'].unique()
    
    for mh in mhs:
    
        data = soluciones_consolidado[soluciones_consolidado['mh'] == mh]
        buffer_filas = []
        for index,fila in data.iterrows():
            lista_real = ast.literal_eval(fila['soluciones'])
            arreglo_binario = np.array(lista_real).astype(int)
            buffer_filas.append(arreglo_binario)
            
        df = pd.DataFrame(buffer_filas)
        df.index = data['clasificador']
        
        fig, ax = plt.subplots(figsize=(20, 4))
        sns.heatmap(df,
                cmap="YlGnBu",
                annot=True,
                cbar_kws={'label': 'Frequency'},
                ax=ax)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=90,
            ha='center',
            fontsize=10
        )
        ax.set_xlabel("Features")
        ax.set_ylabel("Classifiers")
        
        # Ajusta los márgenes: left, right, top, bottom en fracción del figure
        fig.subplots_adjust(left=0.025, right=1.13, top=0.95, bottom=0.15)
        plt.savefig(f'./resultados/data/caracteristicas/heatmap_{mh}.pdf')
        plt.close()
        
    for clasificador in clasificadores:
    
        data = soluciones_consolidado[soluciones_consolidado['clasificador'] == clasificador]
        buffer_filas = []
        for index,fila in data.iterrows():
            lista_real = ast.literal_eval(fila['soluciones'])
            arreglo_binario = np.array(lista_real).astype(int)
            buffer_filas.append(arreglo_binario)
            
        df = pd.DataFrame(buffer_filas)
        df.index = data['mh']
        
        fig, ax = plt.subplots(figsize=(20, 4))
        sns.heatmap(df,
                cmap="YlGnBu",
                annot=True,
                cbar_kws={'label': 'Frequency'},
                ax=ax)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=90,
            ha='center',
            fontsize=10
        )
        ax.set_xlabel("Features")
        ax.set_ylabel("Metaheuristics")
        
        # Ajusta los márgenes: left, right, top, bottom en fracción del figure
        fig.subplots_adjust(left=0.05, right=1.1, top=0.95, bottom=0.15)
        plt.savefig(f'./resultados/data/caracteristicas/heatmap_{clasificador}.pdf')
        plt.close()




def generacion_analitica_descriptiva():
    directorio_base = './resultados/data/rendimiento'
    archivos = os.listdir(directorio_base)
    
    analitica_descriptiva = []
    
    for archivo in archivos:
        if archivo.endswith('.csv'):
            df = pd.read_csv(f'{directorio_base}/{archivo}')
            tipo = archivo.split('.')[0].split('_')[1]
            
            mhs = df['mh'].unique()
            clasificadores = df['clasificador'].unique()
            metricas = df['metric'].unique()            
            for mh in mhs:
                for clasificador in clasificadores:
                    for metric in metricas:
                        data = df[(df['mh'] == mh) & (df['clasificador'] == clasificador) & (df['metric'] == metric)]
                        maximo = np.round(data['value'].max(),3)
                        media = np.round(data['value'].mean(),3)
                        desviacion_estandar = np.round(data['value'].std(),3)
                        analitica_descriptiva.append({
                            'tipo': tipo,
                            'mh': mh,
                            'clasificador': clasificador,
                            'metric': metric,
                            'best': maximo,
                            'avg': media,
                            'std': desviacion_estandar
                        })

    analitica_descriptiva_df = pd.DataFrame(analitica_descriptiva)
    analitica_descriptiva_df.to_csv(f'./resultados/data/analitica_descriptiva.csv', index=False)
    
    # creacion archivos por tipo de clase y metrica
    tipos = analitica_descriptiva_df['tipo'].unique()
    metricas = analitica_descriptiva_df['metric'].unique()
    for tipo in tipos:
        for metrica in metricas:
            data = analitica_descriptiva_df[(analitica_descriptiva_df['tipo'] == tipo) & (analitica_descriptiva_df['metric'] == metrica)]
            data.drop(['tipo', 'metric'], axis=1, inplace=True)
            data.to_csv(f'./resultados/data/tabla_{tipo}_{metrica}.csv', index=False)


def determinar_mejores_resultados():
    
    directorio_base = './resultados/seleccion_caracteristicas'

    mhs = os.listdir(directorio_base)
    clasificadores = os.listdir(f'{directorio_base}/{mhs[0]}')
    ejecuciones = os.listdir(f'{directorio_base}/{mhs[0]}/{clasificadores[0]}')
    
    archivo_data = open("./resultados/data/data_mejores_resultados.txt", "w")
    for mh in mhs:
        for clasificador in clasificadores:
            mejor = 0.0
            index = ""
            for ejecucion in ejecuciones:
                df = pd.read_csv(f'{directorio_base}/{mh}/{clasificador}/{ejecucion}/resultados_clases_{clasificador}_{mh}.csv')
                filtro = df[(df['class'] == '0') & (df['metric'] == 'recall')]
                recall = filtro['value'].unique()[0]
                if recall > mejor:
                    mejor = recall
                    index = ejecucion
            archivo_data.write(f'el mejor resultado para {mh} {clasificador} es {mejor} en la ejecucion {index}\n')
    archivo_data.close()


def data_test_estadistico():
    directorio_base = './resultados/data/rendimiento'
    archivos = os.listdir(directorio_base)
    
    analitica_descriptiva = []
    
    for archivo in archivos:
        if archivo.endswith('.csv') and archivo == "diccionario_1.csv":
            
            df_vacio = pd.DataFrame(columns=['MH', 'fitness'])
            
            df = pd.read_csv(f'{directorio_base}/{archivo}')
            tipo = archivo.split('.')[0].split('_')[1]
            
            mhs = df['mh'].unique()
            clasificadores = df['clasificador'].unique()
            metricas = df['metric'].unique()            
            for mh in mhs:
                for clasificador in clasificadores:
                    nombre_clasificador = None
                    if clasificador == 'LightGBM':
                        nombre_clasificador = 'LGBM'
                    elif clasificador == 'RandomForest':
                        nombre_clasificador = 'RF'
                    elif clasificador == 'KNN':
                        nombre_clasificador = 'KNN'
                    for metric in metricas:
                        data = df[(df['mh'] == mh) & (df['clasificador'] == clasificador) & (df['metric'] == metric)]
                        # print(data)
                        if metric == 'f1-score':
                            print(f'{mh}_{nombre_clasificador}_{metric}')
                            data['mh'] = data['mh'] + "_" + nombre_clasificador
                            data = data.drop('metric', axis=1)
                            data = data.drop('clasificador', axis=1)
                            data = data.rename(columns={'value': 'fitness'})
                            data = data.rename(columns={'mh': 'MH'})
                            # print(data)
                            df_vacio = pd.concat([df_vacio, data])
                            
            print(df_vacio)
            df_vacio.to_csv(f'./resultados/data/data_test_estadistico_1.csv', index=False)

if __name__ == "__main__":
    # generacion_data()
    # generacion_data_soluciones()
    # generacion_heatmap()
    # generacion_analitica_descriptiva()
    # determinar_mejores_resultados()
    data_test_estadistico()