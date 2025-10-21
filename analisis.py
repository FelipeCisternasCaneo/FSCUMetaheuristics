import pandas as pd
import os
import glob

archivos = os.listdir('Resultados_Base/data_bruta')

for archivo in archivos:
    if archivo.endswith('.csv'):
        df = pd.read_csv(f'Resultados_Base/data_bruta/{archivo}')
        nombre_algoritmo = archivo.split('_')[1].split('.')[0]
        print(f'Analizando los resultados de {nombre_algoritmo}')
        reporte = df.describe(include='all') 
        reporte.to_csv(f'./Resultados_Base/analisis_descriptivo/data/reporte_{nombre_algoritmo}.csv')
    
    
    # print(df.info())
    
ruta_carpeta = './Resultados_Base/analisis_descriptivo/data'
patron_archivos = os.path.join(ruta_carpeta, 'reporte_*.csv')

archivo_salida = './Resultados_Base/analisis_descriptivo/promedios_unificados.csv'

lista_archivos = glob.glob(patron_archivos)

if not lista_archivos:
    print(f"Error: No se encontraron archivos con el patrón '{patron_archivos}'")
else:
    print(f"Encontrados {len(lista_archivos)} archivos. Procesando...")
    
    # 3. Lista para guardar cada fila de "mean"
    lista_de_promedios = []

    # 4. Iterar sobre cada archivo encontrado
    for archivo in lista_archivos:
        try:
            # 5. Leer el archivo. 
            # ¡IMPORTANTE! index_col=0 le dice a pandas que la primera columna
            # (donde dice 'count', 'mean', 'std'...) es el índice.
            df_reporte = pd.read_csv(archivo, index_col=0)
            
            # 6. Seleccionar SOLO la fila 'mean'
            # Usamos .loc[] para buscar por el nombre del índice
            fila_mean = df_reporte.loc['mean']
            
            # 7. Convertir la fila (que es una Serie) a un DataFrame de una sola fila
            # El .T transpone el DataFrame para que vuelva a ser una fila
            df_mean = fila_mean.to_frame().T
            
            # 8. (MUY RECOMENDADO) Añadir una columna para saber de dónde vino
            # os.path.basename(archivo) extrae solo el nombre del archivo (ej. 'analisis_1.csv')
            df_mean['fuente_del_archivo'] = os.path.basename(archivo).split('_')[1].split('.')[0]
            
            # 9. Añadir este DataFrame de una fila a nuestra lista
            lista_de_promedios.append(df_mean)

        except Exception as e:
            print(f"No se pudo procesar el archivo '{archivo}'. Error: {e}")

    # 10. Unificar (concatenar) todos los DataFrames de la lista en uno solo
    if lista_de_promedios:
        df_final = pd.concat(lista_de_promedios, ignore_index=True)
        
        # 11. (Opcional) Mover la columna 'fuente_del_archivo' al principio
        columnas = ['fuente_del_archivo'] + [col for col in df_final.columns if col != 'fuente_del_archivo']
        df_final = df_final[columnas]
        
        # 12. Guardar el archivo final
        # index=False para no guardar el índice numérico (0, 1, 2, ...)
        df_final.to_csv(archivo_salida, index=False)
        
        print(f"\n¡Éxito! ✨")
        print(f"Archivo unificado con todos los promedios guardado en: '{archivo_salida}'")
        print("\nVista previa del resultado:")
        print(df_final)
    else:
        print("No se pudo procesar ningún archivo correctamente.")