# demo_yolov8_tkinter.py

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
from pathlib import Path

# ======== CONFIGURACIÓN DEL MODELO ========
def detectar_mejor_modelo(kfold_dir="Detector/output/yolov8_kfold"):
    base_dir = Path(kfold_dir)
    mejor_fold = None
    mejor_map50 = -1.0
    for fold in range(5):
        result_file = base_dir / f"fold{fold}_run" / "results.csv"
        if result_file.exists():
            df = pd.read_csv(result_file)
            last_row = df.iloc[-1]
            map50_col = next((c for c in df.columns if "mAP50(" in c and "95" not in c), None)
            if map50_col:
                valor = last_row[map50_col]
                if valor > mejor_map50:
                    mejor_map50 = valor
                    mejor_fold = fold
    if mejor_fold is not None:
        return base_dir / f"fold{mejor_fold}_run" / "weights" / "best.pt"
    else:
        raise FileNotFoundError("No se encontró ningún modelo válido entrenado.")

modelo_path = detectar_mejor_modelo()
modelo = YOLO(str(modelo_path))

# ======== INTERFAZ GRÁFICA ========
class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Plantas de Fresa - YOLOv8")

        self.panel = tk.Label(root)
        self.panel.pack(padx=10, pady=10)

        btn_cargar = tk.Button(root, text="Cargar Imagen", command=self.cargar_imagen)
        btn_cargar.pack(pady=5)

        self.resultado = None

    def cargar_imagen(self):
        path = filedialog.askopenfilename(filetypes=[("Imagenes", "*.jpg *.png *.jpeg")])
        if not path:
            return

        try:
            imagen = cv2.imread(path)
            resultados = modelo(imagen)
            img_resultado = resultados[0].plot()

            # Convertir a formato para Tkinter
            img_rgb = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil.resize((640, 480)))

            self.panel.configure(image=img_tk)
            self.panel.image = img_tk
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Iniciar GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorApp(root)
    root.mainloop()
