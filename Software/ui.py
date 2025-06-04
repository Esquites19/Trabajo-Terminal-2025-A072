# ui.py
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap import ttk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from PIL import Image, ImageTk
import tkinter as tk
import cv2

class CombinedUI(tb.Window):
    def __init__(self):
        # Arrancamos en modo oscuro (superhero)
        super().__init__(themename="darkly")
        self.title("SightBerry - v0.1.0")
        self.minsize(800, 600)

        # Estado
        self.image_path     = None
        self.video_path     = None
        self.current_img    = None
        self.current_img_tk = None
        self.zoom_factor    = 1.0
        self.fps_value      = tk.IntVar(value=5)

        self._drag_data = {"x": 0, "y": 0}

        # Define fuentes que usarás en el log
        self.font_title   = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        self.font_default = tkfont.Font(family="Segoe UI", size=10)
        self.font_count   = tkfont.Font(family="Segoe UI", size=10, slant="italic")
        self._build_ui()
        self._configure_log_tags()


    def _build_ui(self):
        # Grid principal
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)


         # Menú izquierdo
        menu = tb.Frame(self)
        menu.grid(row=0, column=0, sticky="ns", padx=8, pady=8)

         # Tema
        self.btn_theme = tb.Button(
            menu, text="🌓 Tema",
            bootstyle=(OUTLINE, LIGHT),
            command=self._toggle_theme,
            padding=(0,2)
        )
        self.btn_theme.pack(fill="x", pady=(0,12))

        # Bloque Carga
        load_frame = tb.Labelframe(menu, text="Opciones", bootstyle=PRIMARY, padding=(8,0))
        load_frame.pack(fill="x", pady=(0,8), ipadx=4, ipady=4)
        tb.Button(load_frame, text="Cargar Imagen",   bootstyle=PRIMARY,
                  command=self._ui_load_image, padding=(0,2)).pack(fill="x", pady=2)
        tb.Button(load_frame, text="Cargar Video",    bootstyle=PRIMARY,
                  command=self._ui_load_video, padding=(0,2)).pack(fill="x", pady=2)
        # Bloque Procesamiento (oculto hasta cargar)
        self.ctrl_frame = tb.Labelframe(menu, text="Procesamiento", bootstyle=INFO, padding=(8,0))
        # Widgets de procesamiento
        self.btn_process_img = tb.Button(self.ctrl_frame,  text="Procesar Imagen", bootstyle="success-outline", padding=(0,2))
        self.btn_process_vid = tb.Button(self.ctrl_frame,  text="Procesar Video",  bootstyle="success-outline", padding=(0,2))
        self.lbl_fps         = tb.Label(self.ctrl_frame,   text="FPS:",            bootstyle=LIGHT)
        self.spin_fps        = tb.Spinbox(self.ctrl_frame, from_=1, to=30, textvariable=self.fps_value,width=5, bootstyle=INFO)
        self.btn_export_pdf = tb.Button(self.ctrl_frame, text="Exportar PDF", bootstyle="info-outline",padding=(0,2))
        # Botones de control
        self.btn_pause       = tb.Button(self.ctrl_frame, text="Pausar",     bootstyle="warning-outline", padding=(0,2))
        self.btn_continue    = tb.Button(self.ctrl_frame, text="Continuar",  bootstyle="success-outline", padding=(0,2))
        # Panel central (imagen + log)
        central = tb.Frame(self)
        central.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        central.grid_rowconfigure(0, weight=1, minsize=400)
        central.grid_rowconfigure(1, weight=0, minsize=150)
        central.grid_columnconfigure(0, weight=1)

        # Canvas + scrollbars
        img_frame = tb.Frame(central, bootstyle=INFO)
        img_frame.grid(row=0, column=0, sticky="nsew")
        img_frame.grid_rowconfigure(0, weight=1)
        img_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(img_frame, bd=0, highlightthickness=0)
        vs = tb.Scrollbar(img_frame, orient=VERTICAL,   command=self.canvas.yview, bootstyle=ROUND)
        hs = tb.Scrollbar(img_frame, orient=HORIZONTAL, command=self.canvas.xview, bootstyle=ROUND)
        self.canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vs.grid (row=0, column=1, sticky="ns")
        hs.grid (row=1, column=0, sticky="ew")

        # Log de texto
        log_frame = tb.Labelframe(central, text="Reporte", bootstyle=INFO, padding=(4,4))
        log_frame.grid_propagate(False)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(8,0))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        vscroll = tb.Scrollbar(log_frame, orient=VERTICAL, command=lambda *a: self.log.yview(*a), bootstyle=ROUND)
        self.log = tk.Text(log_frame, wrap="none", yscrollcommand=vscroll.set, bd=0, highlightthickness=0)
        self.log.grid (row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        # Bind zoom ratón
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>",   self._on_mousewheel)
        self.canvas.bind("<Button-5>",   self._on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)

    # —— Métodos UI puros —— #
    def _export_pdf(self):
        pass
    
    def _on_drag_start(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag_move(self, event):
        dx = self._drag_data["x"] - event.x
        dy = self._drag_data["y"] - event.y
        self.canvas.xview_scroll(int(dx), "units")
        self.canvas.yview_scroll(int(dy), "units")
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y


    def _toggle_theme(self):
        current = self.style.theme_use()
        nuevo   = "minty" if current == "darkly" else "darkly"
        self.style.theme_use(nuevo)

        if nuevo == "darkly":
            self.btn_theme.configure(bootstyle=(OUTLINE, LIGHT))
        else:
            self.btn_theme.configure(bootstyle=(OUTLINE, DARK))

    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))
        self._display_image()


    def _display_image(self):
        if not self.current_img:
            return

        # Forzar actualización del canvas para obtener su tamaño real
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calcular tamaño objetivo con factor de zoom
        base_image = self.current_img.copy()
        max_width = int(canvas_width * self.zoom_factor)
        max_height = int(canvas_height * self.zoom_factor)
        base_image.thumbnail((max_width, max_height), resample=Image.LANCZOS)

        # Guardar imagen visual y mostrarla centrada
        self.current_img_tk = ImageTk.PhotoImage(base_image)
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor="center",
            image=self.current_img_tk
        )
        self.canvas.config(scrollregion=self.canvas.bbox("all"))


    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.config(state="disabled")

    def _update_log(self, counts, colors, prediction):
        self.log.config(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.insert(tk.END, "Detecciones acumuladas:\n")
        for etiqueta, conteo in counts.items():
            color = '#%02x%02x%02x' % tuple(colors.get(etiqueta, (255,255,0)))
            self.log.insert(tk.END, "  ", (color,))
            self.log.insert(tk.END, f"{etiqueta}: {conteo}\n")
            self.log.tag_config(color, background=color, foreground='black')
        self.log.insert(tk.END, f"\nClasificación: {prediction}\n")  # Solo la actual
        self.log.config(state="disabled")

    def _configure_log_tags(self):
        # Estilo para el título del log
            self.log.tag_configure(
                "title",
                font=self.font_title,
                foreground="#007ACC",   # color accent
                spacing1=5, spacing3=5
            )
            # Estilo para líneas de conteo
            self.log.tag_configure(
                "count",
                font=self.font_count,
                foreground=self.style.colors.success,  # si usas bootstrap
                lmargin1=20, lmargin2=20
            )
            # Estilo para cualquier mensaje de error
            self.log.tag_configure(
                "error",
                font=self.font_default,
                foreground="#FF5555",
                background="#2D2D30",
                underline=1
            )        


    def _ui_load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imagen","*.jpg;*.png")])
        if not path:
            return
        self.image_path = path
        pil = Image.open(path).convert("RGB")
        self.current_img, self.zoom_factor = pil, 1.0
        self._display_image()
        # Mostrar sólo el botón Procesar Imagen
        self._clear_log()
        self.ctrl_frame.pack_forget()
        for w in self.ctrl_frame.winfo_children():
            w.pack_forget()
        self.ctrl_frame.pack(fill="x", ipady=4)
        # Mostrar sólo el botón de procesar imagen
        self.btn_process_img.pack(fill="x", pady=2) 
        self.btn_export_pdf.pack(fill="x", pady=2)


    def _ui_load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video","*.mp4;*.avi")])
        if not path:
            return
        self.video_path = path
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "No se pudo leer el video")
            return
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_img, self.zoom_factor = Image.fromarray(img), 1.0
        self._display_image()

        # Mostrar controles de video
        self._clear_log()
        self.ctrl_frame.pack_forget()
        for w in self.ctrl_frame.winfo_children():
            w.pack_forget()
        self.ctrl_frame.pack(fill="x", ipady=4) 
        self.btn_process_vid.pack(fill="x", pady=2)
        self.lbl_fps     .pack(fill="x", pady=(8,2))
        self.spin_fps    .pack(fill="x", pady=2)
        # Mostrar botones de pausa y continuar  
        self.btn_pause.pack(fill="x", pady=2)
        self.btn_continue.pack(fill="x", pady=2)
        self.btn_pause   .pack(fill="x", pady=2)
        self.btn_continue.pack(fill="x", pady=2)
        
