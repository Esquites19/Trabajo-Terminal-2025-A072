import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

# ————————————————————————————————————————————————
# Script: evaluate_yolov8_folds.py (versión extendida)
# Objetivo:
#   1) Evaluar métricas globales por fold (Precision, Recall, mAP@0.5, mAP@50-95)
#   2) Identificar el mejor fold según mAP@50-95
#   3) Obtener métricas por clase (AP@50-95 y soporte) para cada fold
#   4) Calcular el promedio de métricas por clase entre todos los folds
#   5) Visualizar resultados en tablas y gráficos
# ————————————————————————————————————————————————

KFOLD_DIR   = Path("2_Detector/output/yolov8_kfold")  # Carpeta raíz de los resultados por fold
NUM_FOLDS   = 5                                         # Número de folds esperados
EVAL_CLASSES = True                                     # Cambia a False si no deseas evaluar por‑clase

print("\n📊 Evaluando resultados por fold...\n")
print("🧠 Dispositivo activo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ————— Utilidades globales —————

def buscar_columna(columnas, clave):
    """Devuelve el nombre de la primera columna que contenga la clave (ignora espacios/mayúsculas)."""
    for col in columnas:
        if clave.lower() in col.replace(" ", "").lower():
            return col
    return None

# ————————————————————————————————
# 1. MÉTRICAS GLOBALES (ÁRBOL DE RESULTADOS.CSV)
# ————————————————————————————————

def extraer_metricas_csv(results_path):
    """Lee el CSV de resultados del entrenamiento y extrae las métricas finales."""
    try:
        df = pd.read_csv(results_path)
        final = df.iloc[-1]
        columnas = df.columns.tolist()
        col_map = {
            "precision":  buscar_columna(columnas, "metrics/precision"),
            "recall":     buscar_columna(columnas, "metrics/recall"),
            "map_50":     buscar_columna(columnas, "metrics/mAP50("),
            "map_50_95":  buscar_columna(columnas, "metrics/mAP50-95("),
        }
        if None in col_map.values():
            raise ValueError(f"Columnas no encontradas: {col_map}")
        return {
            "Precision":  final[col_map["precision"]],
            "Recall":     final[col_map["recall"]],
            "mAP@0.5":    final[col_map["map_50"]],
            "mAP@50-95":  final[col_map["map_50_95"]],
        }
    except Exception as e:
        print(f"⚠️  Error leyendo {results_path}: {e}")
        return None

# ————————————————————————————————
# 2. MÉTRICAS POR CLASE
# ————————————————————————————————

def extraer_metricas_clase(run_dir: Path):
    """Ejecuta model.val() para obtener AP@50‑95 y soporte por clase para un fold."""
    pesos = run_dir / "weights" / "best.pt"
    args_yaml = run_dir / "args.yaml"

    if not pesos.exists():
        print(f"❌ No se encontró {pesos.relative_to(run_dir.parent)} – se omite evaluación por clase.")
        return None

    # Recuperar la ruta del archivo data.yaml usado en el entrenamiento
    if not args_yaml.exists():
        print(f"❌ No se encontró {args_yaml.name}; no se puede determinar data.yaml")
        return None

    with open(args_yaml, "r", encoding="utf-8") as f:
        args_cfg = yaml.safe_load(f)
    data_yaml = args_cfg.get("data")
    if not data_yaml:
        print(f"❌ data.yaml no especificado en {args_yaml.name}")
        return None

    # Evaluar el modelo
    model = YOLO(str(pesos))
    metrics = model.val(data=data_yaml, split="val", verbose=False, plots=False, save_json=False)
    # metrics.box.maps  → list[float] AP50‑95 por clase
    # metrics.box.tcls  → list[int]   ground‑truths por clase (soporte)
    # model.names       → dict[int,str] o list[str]  nombres de clase
    class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
    df = pd.DataFrame({
        "Clase": class_names,
        "AP@50-95": metrics.box.maps,
        "Soporte": metrics.box.tcls,
    })
    return df

# ————————————————————————————————
# 3. RECORRIDO DE FOLDS
# ————————————————————————————————

metricas_globales = []
metricas_clases   = {}
fold_names        = []

for fold in range(NUM_FOLDS):
    run_dir = KFOLD_DIR / f"fold{fold}_run"
    results_csv = run_dir / "results.csv"

    print(f"\n—— Fold {fold} ———————————————————————————")

    # — Globales
    if results_csv.exists():
        glob = extraer_metricas_csv(results_csv)
        if glob:
            metricas_globales.append(glob)
            fold_names.append(run_dir.name)
    else:
        print(f"❌ {results_csv.relative_to(KFOLD_DIR.parent)} no encontrado")

    # — Por clase
    if EVAL_CLASSES:
        cls_df = extraer_metricas_clase(run_dir)
        if cls_df is not None:
            metricas_clases[run_dir.name] = cls_df

# ————————————————————————————————
# 4. RESULTADOS GLOBALES
# ————————————————————————————————

if metricas_globales:
    df_glob = pd.DataFrame(metricas_globales, index=fold_names)
    df_glob.loc["Promedio"] = df_glob.mean()

    # Mejor fold según mAP@50‑95
    mejor_fold_idx   = df_glob.drop(index="Promedio")["mAP@50-95"].idxmax()
    mejor_fold_stats = df_glob.loc[mejor_fold_idx]

    print("\n📋 Métricas globales por fold (+ promedio):")
    print(df_glob.round(4))

    print("\n🏆 Mejor fold (mAP@50‑95):", mejor_fold_idx)
    print(mejor_fold_stats.round(4))

    # — Gráfica de mAP y Recall —
    df_plot = df_glob.drop(index="Promedio")
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot.index, df_plot["mAP@0.5"], marker="o", label="mAP@0.5")
    plt.plot(df_plot.index, df_plot["Recall"],  marker="x", label="Recall")
    plt.scatter(mejor_fold_idx, mejor_fold_stats["mAP@0.5"], s=200, marker="*", label="Mejor mAP@50-95")
    plt.title("mAP@0.5 y Recall por Fold")
    plt.xlabel("Fold")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("⚠️  No se encontraron métricas globales válidas.")

# ————————————————————————————————
# 5. RESULTADOS POR CLASE
# ————————————————————————————————

if EVAL_CLASSES and metricas_clases:
    print("\n📊 Métricas AP@50‑95 por clase (cada fold):")
    for fold_name, df_cls in metricas_clases.items():
        print(f"\n— {fold_name} —")
        print(df_cls.round(4))

    # —— Promediar entre folds ——
    # Concatenamos y calculamos promedio por clase
    df_all = pd.concat(metricas_clases.values(), keys=metricas_clases.keys(), names=["Fold", "Idx"]).reset_index(0)
    df_prom = df_all.groupby("Clase").agg({"AP@50-95": "mean", "Soporte": "sum"}).sort_values("AP@50-95", ascending=False)

    print("\n📈 Promedio AP@50‑95 por clase (todos los folds):")
    print(df_prom.round(4))

    # — Gráfico barras —
    plt.figure(figsize=(12, 6))
    plt.bar(df_prom.index, df_prom["AP@50-95"])
    plt.title("AP@50‑95 promedio por clase (K‑Fold)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("AP@50-95")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️  No se encontraron métricas por clase válidas.")
