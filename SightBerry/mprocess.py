import cv2
from PIL import Image, ImageDraw, ImageFont
import time
import torch
import numpy as np
from torchvision import transforms
from collections import defaultdict
import random
import colorsys

class ModelProcessor:
    def __init__(self, classifier, detector, device, idx_to_class):
        self.classifier = classifier
        self.detector = detector
        self.device = device
        self.idx_to_class = idx_to_class
        self.filter_class = None
        self.box_thickness = 2
        # Preparar generación de colores dinámica
        self._hue = 0.0

        self.class_colors  = {}

        # Inicializar el color inicial
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Cambiar tamaño a 224x224
            transforms.ToTensor()           # Convertir a tensor (sin normalización)
        ])

        self.class_colors = {}
        self.class_counts = defaultdict(int)
        self.seen_classes = set()
        self._stop_video = False
        self._pause_video = False

    def process_image(self, image_path):
        # Reiniciar conteos para no acumular detecciones anteriores
        self.class_counts.clear()

        pil_image = Image.open(image_path).convert("RGB")
        detections = self._run_detector(pil_image)
        prediction = self._run_classifier(pil_image)
        self._update_counts(detections)
        annotated = self._draw_boxes(pil_image, detections)

        print(f"[IMAGEN] Clasificación actual: {prediction}")
        print("[IMAGEN] Detecciones:")
        for label, conf, _ in detections:
            print(f" - {label} ({conf*100:.1f}%)")

        return annotated, self.class_counts, self.class_colors, prediction

    def stop_video(self):
        self._stop_video = True

    def continue_video(self):
        self._pause_video = False

    def pause_video(self):
        self._pause_video = True

    def toggle_pause(self):
        self._pause_video = not self._pause_video

    def process_video(self, video_path, callback_overlay, fps=5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # Ahora respetamos el intervalo definido por el usuario
        frame_interval = int(video_fps * 1)
        self._stop_video = False
        self._pause_video = False
        # guardamos la velocidad dinámica
        self.fps = fps

        i = 0

        while True:
            if self._stop_video:
                break
            if self._pause_video:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            if i % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                # 1) detectar
                detections = self._run_detector(pil_frame)
                # asignar color sin acumular conteos
                self._update_seen_classes(detections)

                # 2) clasificar y registrar solo nuevas
                prediction = self._run_classifier(pil_frame)
                print(f"[VIDEO] Frame {i}: Clasificación actual: {prediction}")

                print(f"[VIDEO] Frame {i}: Detecciones:")
                for label, conf, _ in detections:
                    print(f" - {label} ({conf*100:.1f}%)")

                # 3) conteo por frame
                frame_counts = defaultdict(int)
                for label, _, _ in detections:
                    frame_counts[label] += 1

                # 4) dibujar cajas
                annotated = self._draw_boxes(pil_frame, detections)

                # 5) callback con imagen, conteos y clasificación actual
                callback_overlay(annotated, frame_counts, prediction)

                # respetamos self.fps dinámico
                time.sleep(1.0 / self.fps)

            i += 1

        cap.release()

    def _run_classifier(self, pil_image):
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.classifier(tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        return self.idx_to_class[predicted_class]

    def _run_detector(self, pil_image):
        results = self.detector(pil_image)
        boxes = results[0].boxes
        detections = []
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if conf < 0.1:
                continue  # Solo aceptar detecciones con confianza >= 0.5
            xyxy = box.xyxy.squeeze().tolist()
            label = self.detector.names[cls_id]
            detections.append((label, conf, xyxy))
        return detections

    def _update_counts(self, detections):
        for label, conf, _ in detections:
            self.class_counts[label] += 1
            if label not in self.class_colors:
                self._assign_color(label)

    def _update_seen_classes(self, detections):
        for label, conf, _ in detections:
            self.seen_classes.add(label)
            if label not in self.class_colors:
                self._assign_color(label)

    def _assign_color(self, label):
        """Asigna un color nuevo usando fracción áurea en HSV."""
        self._hue = (self._hue + 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(self._hue, 0.6, 0.9)
        self.class_colors[label] = (
            int(r * 255),
            int(g * 255),
            int(b * 255)
        )

    def _draw_boxes(self, pil_image, detections):
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("arial.ttf", size=11)
        except:
            font = ImageFont.load_default()

        for label, conf, xyxy in detections:
            # Si hay filtro y no coincide, saltamos
            if self.filter_class and label != self.filter_class:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            color = self.class_colors.get(label, (255, 0, 0))
            draw.rectangle([x1, y1, x2, y2],
                           outline=color,
                           width=self.box_thickness)
            text = f" {conf*100:.1f}%"
            bbox = font.getbbox(text)
            text_w = bbox[2]-bbox[0]
            text_h = bbox[3]-bbox[1]
            draw.rectangle([x1, y1 - text_h, x1+text_w, y1],
                           fill=color)
            draw.text((x1, y1 - text_h),
                      text,
                      fill=(0,0,0),
                      font=font)

        return pil_image
    
    def set_filter_class(self, label: str|None):
        """Define qué etiqueta se debe mostrar. None = todas."""
        self.filter_class = label
