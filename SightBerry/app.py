import threading
from ui import CombinedUI
from mprocess import ModelProcessor
from utils import cargar_modelo_clasificacion, cargar_detector, generar_reporte_pdf
from tkinter import filedialog, messagebox
import torch
import tkinter as tk
from PIL import Image, ImageTk

class CombinedApp(CombinedUI):
    def __init__(self):
        super().__init__()
        # Instanciar IA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls_path = r"SightBerry/models/resnet50.pth"
        det_path = r"SightBerry/models/yolov8.pt"
        clf, idx_to_class = cargar_modelo_clasificacion(cls_path, device)
        det = cargar_detector(det_path)
        self.processor = ModelProcessor(clf, det, device, idx_to_class)

        # Conectar botones de procesamiento
        self.btn_process_img .config(command=self._run_process_image)
        self.btn_process_vid .config(command=self._run_process_video)
        self.btn_pause       .config(command=self._stop_video_processing)
        self.btn_continue    .config(command=self._continue_video_processing)
        self.btn_export_pdf.config(command=self._export_pdf)

        self.canvas.bind('<MouseWheel>', self._on_mousewheel)

    # ——— Métodos de integración IA ———
    def _export_pdf(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF","*.pdf")]
        )
        if not path:
            return

        counts   = self.processor.class_counts
        history  = self.processor.new_classifications_ordered

        try:
            generar_reporte_pdf(counts, history, path)
            messagebox.showinfo("Éxito", f"Reporte guardado en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el reporte:\n{e}")

    def _run_process_image(self):
        if not self.image_path:
            return
        annotated, counts, colors, prediction = self.processor.process_image(self.image_path)
        self.current_img = annotated
        self._display_image()
        self._update_log(counts, colors, prediction)  # <-- agregar argumento


    def _run_process_video(self):
        if not self.video_path:
            return
        self.spin_fps.config(state="disabled")
        fps = self.fps_value.get()

        threading.Thread(
            target=lambda: self.processor.process_video(
                self.video_path,
                callback_overlay=self._update_frame,
                fps=fps
            ),
            daemon=True
        ).start()

    def _stop_video_processing(self):
        self.processor.pause_video()
        self.spin_fps.config(state="normal")

    def _continue_video_processing(self):
        new_fps = self.fps_value.get()
        self.processor.fps = new_fps
        self.processor.continue_video()
        self.spin_fps.config(state="disabled")

    def _update_frame(self, pil, frame_counts, classification):
        self.current_img = pil
        self._display_image()

        # Detecciones del frame actual
        self.log.config(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.insert(tk.END, "Detecciones en frame actual:\n")
        for etiqueta, conteo in frame_counts.items():
            rgb   = self.processor.class_colors.get(etiqueta,(255,255,0))
            color = '#%02x%02x%02x' % rgb
            self.log.insert(tk.END, "  ", (color,))
            self.log.insert(tk.END, f"{etiqueta}: {conteo}\n")
            self.log.tag_config(color, background=color, foreground='black')
        self.log.insert(tk.END, f"\nClasificación: {classification}\n")
        self.log.config(state="disabled")

    def _display_image(self):
        # Permitir zoom incluso si la imagen es más pequeña que el canvas
        if self.current_img is None:
            return
        # Obtener el tamaño del canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        # Si el usuario ha hecho zoom, usar el factor de zoom
        zoom = getattr(self, 'zoom_factor', 1.0)
        img = self.current_img.copy()
        # Redimensionar la imagen según el zoom (aunque sea más pequeña que el canvas)
        new_width = max(1, int(img.width * zoom))
        new_height = max(1, int(img.height * zoom))
        img = img.resize((new_width, new_height), resample=Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        # Centrar la imagen en el canvas
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        self.canvas.create_image(x, y, anchor='nw', image=self.tk_img)

    def _on_mousewheel(self, event):
        # Zoom con la rueda del ratón
        if not hasattr(self, 'zoom_factor'):
            self.zoom_factor = 1.0
        if event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        self._display_image()

