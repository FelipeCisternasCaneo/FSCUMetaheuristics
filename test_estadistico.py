import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp

# === 1. Cargar y preparar datos ===
# Asegúrate de que el archivo esté en el mismo directorio
nombre = 'data_test_estadistico_1'
df = pd.read_csv(f"resultados/data/{nombre}.csv")  
df["repetition"] = df.groupby("MH").cumcount()

# Encontrar la cantidad mínima de repeticiones compartidas por todos
min_reps = df.groupby("MH")["repetition"].max().min() + 1
df_filtered = df[df["repetition"] < min_reps]

# Pivotear: columnas = metaheurísticas, filas = repeticiones
pivot_df = df_filtered.pivot(index="repetition", columns="MH", values="fitness")

# === 2. Test de Friedman ===
print("== Test de Friedman ==")
stat, p_friedman = friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])
print(f"Friedman chi² = {stat:.4f}, p-value = {p_friedman:.2E}")

# === 3. Test Post Hoc de Nemenyi (si Friedman es significativo) ===
if p_friedman < 0.05:
    print("\n== Test Post Hoc de Nemenyi ==")
    nemenyi = sp.posthoc_nemenyi_friedman(pivot_df.values)
    nemenyi.columns = pivot_df.columns
    nemenyi.index = pivot_df.columns

    # Mostrar p-valor < 0.05 como notación científica, el resto "-"
    nemenyi_formatted = nemenyi.applymap(lambda x: f"{x:.2e}" if x < 0.05 else "-")
    print(nemenyi_formatted)
    
    # === Obtener pares con diferencia significativa (p < 0.05) ===
    significant_pairs = []

    for i in range(len(nemenyi.index)):
        for j in range(i + 1, len(nemenyi.columns)):
            p_val = nemenyi.iloc[i, j]
            if p_val < 0.05:
                pair = (nemenyi.index[i], nemenyi.columns[j], round(p_val, 4))
                significant_pairs.append(pair)
    print(significant_pairs)
    print(len(significant_pairs))
else:
    print("No se encontraron diferencias significativas entre métodos según Friedman.")

# === 4. Test de Wilcoxon signed-rank para todos los pares ===
print("\n== Test de Wilcoxon signed-rank (pareado) ==")
wilcoxon_results = []
mh_names = pivot_df.columns
mhs = mh_names.tolist()
for mh1 in mhs:
    for mh2 in mhs:
        if mh1 != mh2:
            stat, p_wilcoxon = wilcoxon(pivot_df[mh1], pivot_df[mh2], alternative='greater')

            # Interpretación direccional para problema de minimización
            if p_wilcoxon < 0.05:
                    conclusion = f"{mh1} is better than {mh2}"
                    wilcoxon_results.append({
                        "MH1": mh1,
                        "MH2": mh2,
                        "p-value": round(p_wilcoxon, 4),
                        "conclusion": conclusion
                    })
            # else:
            #     conclusion = "No significativa"

            # wilcoxon_results.append({
            #     "MH1": mh1,
            #     "MH2": mh2,
            #     "p-value": round(p_wilcoxon, 4),
            #     "conclusion": conclusion
            # })

wilcoxon_df = pd.DataFrame(wilcoxon_results)
print(wilcoxon_df)

# === 5. (Opcional) Exportar resultados a CSV ===
nemenyi_formatted.to_csv(f"{nombre}_test_posthoc_nemenyi.csv")
wilcoxon_df.to_csv(f"{nombre}_test_wilcoxon.csv", index=False)