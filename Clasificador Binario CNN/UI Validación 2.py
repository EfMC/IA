import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ============================================================
# CONFIGURACIÓN TÉCNICA (RESOLUCIÓN BASE 128x128)
# ============================================================
IMG_SIZE = (128, 128)  # Resolución original de entrenamiento
CLASES = ['barco', 'no_barco']

class SistemaPercepcionUAV:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Percepción UAV — Inspección y Monitoreo Marítimo")
        self.root.geometry("1200x820")
        
        self.modelo = None
        self.ruta_modelo = ""
        self.ruta_carpeta = ""
        self.archivos_img = []
        self.etiquetas_reales = []
        self.predicciones_prob = []
        self.predicciones_clase = []
        self.indice_actual = 0

        self._crear_interfaz()

    def _crear_interfaz(self):
        # ---------- Panel Superior: Carga de Modelo y Carpeta ----------
        frame_top = ttk.LabelFrame(self.root, text=" Configuración de Componentes ")
        frame_top.pack(fill="x", padx=10, pady=5)

        btn_cargar_modelo = ttk.Button(frame_top, text="📁 Seleccionar Modelo (.keras)", command=self.seleccionar_modelo)
        btn_cargar_modelo.pack(side="left", padx=5, pady=8)

        self.lbl_nombre_modelo = ttk.Label(frame_top, text="Modelo: (Ninguno seleccionado)", font=("Arial", 9, "italic"))
        self.lbl_nombre_modelo.pack(side="left", padx=5)

        ttk.Separator(frame_top, orient="vertical").pack(side="left", fill="y", padx=10, pady=5)

        btn_cargar_carpeta = ttk.Button(frame_top, text="📂 Seleccionar Carpeta Test", command=self.cargar_carpeta)
        btn_cargar_carpeta.pack(side="left", padx=5, pady=8)

        self.lbl_estado = ttk.Label(frame_top, text="Estado: Esperando modelo...", font=("Arial", 10, "bold"), foreground="orange")
        self.lbl_estado.pack(side="right", padx=10)

        # ---------- Panel Central: Inferencia y Visualización ----------
        frame_center = ttk.Frame(self.root)
        frame_center.pack(fill="both", expand=True, padx=10, pady=5)

        # Visor de Imagen Satelital
        frame_visor = ttk.LabelFrame(frame_center, text=" Inspección Satelital UAV ")
        frame_visor.pack(side="left", fill="both", expand=True, padx=5)

        self.lbl_canvas_img = ttk.Label(frame_visor)
        self.lbl_canvas_img.pack(pady=10)

        self.lbl_info_img = ttk.Label(frame_visor, text="Archivo: -", font=("Arial", 10))
        self.lbl_info_img.pack()

        self.lbl_prediccion = ttk.Label(frame_visor, text="Predicción: -", font=("Arial", 12, "bold"))
        self.lbl_prediccion.pack(pady=5)

        # Controles de Etiquetado Manual
        frame_etiquetado = ttk.Frame(frame_visor)
        frame_etiquetado.pack(pady=10)

        ttk.Label(frame_etiquetado, text="Confirmar Etiqueta Real:").pack(side="left", padx=5)
        btn_barco = ttk.Button(frame_etiquetado, text="BARCO (1)", command=lambda: self.registrar_etiqueta(0))
        btn_barco.pack(side="left", padx=3)
        btn_no_barco = ttk.Button(frame_etiquetado, text="NO_BARCO (0)", command=lambda: self.registrar_etiqueta(1))
        btn_no_barco.pack(side="left", padx=3)

        # Panel Derecho: Métricas Detalladas en Vivo
        frame_metricas = ttk.LabelFrame(frame_center, text=" Evaluación y Analítica en Tiempo Real ")
        frame_metricas.pack(side="right", fill="both", expand=True, padx=5)

        # Sub-frame para Métricas Numéricas
        frame_text_metricas = ttk.Frame(frame_metricas)
        frame_text_metricas.pack(fill="x", padx=10, pady=5)

        self.lbl_accuracy = ttk.Label(frame_text_metricas, text="Accuracy: -- %", font=("Arial", 11, "bold"))
        self.lbl_accuracy.grid(row=0, column=0, sticky="w", pady=2, padx=5)

        self.lbl_precision = ttk.Label(frame_text_metricas, text="Precisión (Barco): -- %", font=("Arial", 10))
        self.lbl_precision.grid(row=1, column=0, sticky="w", pady=2, padx=5)

        self.lbl_recall = ttk.Label(frame_text_metricas, text="Recall (Barco): -- %", font=("Arial", 10))
        self.lbl_recall.grid(row=2, column=0, sticky="w", pady=2, padx=5)

        self.lbl_f1 = ttk.Label(frame_text_metricas, text="F1-Score (Barco): -- %", font=("Arial", 10))
        self.lbl_f1.grid(row=3, column=0, sticky="w", pady=2, padx=5)

        self.lbl_meta = ttk.Label(frame_text_metricas, text="Meta Oficial: > 98.00 %", font=("Arial", 10), foreground="blue")
        self.lbl_meta.grid(row=0, column=1, sticky="e", pady=2, padx=15)

        self.lbl_penalizacion = ttk.Label(frame_text_metricas, text="Penalización: 0.0 pts", font=("Arial", 11, "bold"), foreground="green")
        self.lbl_penalizacion.grid(row=1, column=1, sticky="e", pady=2, padx=15)

        # Matriz de Confusión Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas_matriz = FigureCanvasTkAgg(self.fig, master=frame_metricas)
        self.canvas_matriz.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def seleccionar_modelo(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar Modelo Entrenado Keras",
            filetypes=[("Archivos Keras", "*.keras"), ("Todos los archivos", "*.*")]
        )
        if ruta:
            try:
                self.modelo = tf.keras.models.load_model(ruta)
                self.ruta_modelo = ruta
                nombre_archivo = os.path.basename(ruta)
                self.lbl_nombre_modelo.config(text=f"Modelo: {nombre_archivo}", font=("Arial", 9, "bold"))
                self.lbl_estado.config(text="✅ Modelo Cargado", foreground="green")
                messagebox.showinfo("Modelo Cargado", f"Se cargó correctamente el modelo:\n{nombre_archivo}")
            except Exception as e:
                messagebox.showerror("Error de Carga", f"No se pudo cargar el archivo .keras:\n{e}")

    def cargar_carpeta(self):
        if not self.modelo:
            messagebox.showwarning("Advertencia", "Por favor, seleccione primero un modelo .keras")
            return

        self.ruta_carpeta = filedialog.askdirectory(title="Seleccionar Carpeta de Imágenes de Test")
        if self.ruta_carpeta:
            exts = ('.png', '.jpg', '.jpeg', '.bmp')
            self.archivos_img = [f for f in os.listdir(self.ruta_carpeta) if f.lower().endswith(exts)]
            
            if not self.archivos_img:
                messagebox.showerror("Error", "No se encontraron imágenes válidas en la carpeta seleccionada.")
                return

            self.etiquetas_reales = []
            self.predicciones_prob = []
            self.predicciones_clase = []
            self.indice_actual = 0
            
            self.procesar_imagen_actual()

    def procesar_imagen_actual(self):
        if self.indice_actual >= len(self.archivos_img):
            messagebox.showinfo("Evaluación Finalizada", "Se ha completado el procesamiento de todas las imágenes.")
            return

        nombre_archivo = self.archivos_img[self.indice_actual]
        ruta_completa = os.path.join(self.ruta_carpeta, nombre_archivo)

        # Preprocesamiento a 128x128 px
        img_pil = Image.open(ruta_completa).convert('RGB').resize(IMG_SIZE, Image.Resampling.BILINEAR)
        img_array = np.array(img_pil, dtype=np.float32)
        img_tensor = np.expand_dims(img_array, axis=0)

        # Inferencia del Modelo Seleccionado
        prob = float(self.modelo.predict(img_tensor, verbose=0)[0][0])
        clase_pred = 0 if prob < 0.50 else 1  # 0: barco, 1: no_barco (orden Keras)

        self.predicciones_prob.append(prob)
        self.predicciones_clase.append(clase_pred)

        # Renderizado de Imagen
        img_display = img_pil.resize((240, 240), Image.Resampling.NEAREST)
        self.tk_img = ImageTk.PhotoImage(img_display)
        self.lbl_canvas_img.config(image=self.tk_img)

        txt_clase = "BARCO" if clase_pred == 0 else "NO_BARCO"
        self.lbl_info_img.config(text=f"Imagen {self.indice_actual + 1}/{len(self.archivos_img)}: {nombre_archivo}")
        self.lbl_prediccion.config(text=f"Predicción: {txt_clase} (Sigmoide: {prob:.4f})")

        # Asignación automática si el nombre del archivo contiene la etiqueta
        if "barco" in nombre_archivo.lower() and "no_barco" not in nombre_archivo.lower():
            self.registrar_etiqueta(0)
        elif "no_barco" in nombre_archivo.lower():
            self.registrar_etiqueta(1)

    def registrar_etiqueta(self, etiqueta_real):
        if len(self.etiquetas_reales) <= self.indice_actual:
            self.etiquetas_reales.append(etiqueta_real)
        else:
            self.etiquetas_reales[self.indice_actual] = etiqueta_real

        self.actualizar_metricas()
        self.indice_actual += 1
        self.procesar_imagen_actual()

    def actualizar_metricas(self):
        if not self.etiquetas_reales:
            return

        y_true = np.array(self.etiquetas_reales)
        y_pred = np.array(self.predicciones_clase[:len(y_true)])

        acc = np.mean(y_true == y_pred) * 100.0
        prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0
        rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0
        f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0

        self.lbl_accuracy.config(text=f"Accuracy en Vivo: {acc:.2f} %")
        self.lbl_precision.config(text=f"Precisión (Barco): {prec:.2f} %")
        self.lbl_recall.config(text=f"Recall (Barco): {rec:.2f} %")
        self.lbl_f1.config(text=f"F1-Score (Barco): {f1:.2f} %")

        # Penalización oficial: 0.5 puntos por cada 2% debajo del 98%
        if acc < 98.0:
            diferencia = 98.0 - acc
            penalizacion = (diferencia / 2.0) * 0.5
            self.lbl_penalizacion.config(text=f"Penalización: -{penalizacion:.2f} pts", foreground="red")
        else:
            self.lbl_penalizacion.config(text="Penalización: 0.0 pts (Meta Cumplida)", foreground="green")

        # Gráfica Matriz de Confusión
        self.ax.clear()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.ax,
                    xticklabels=CLASES, yticklabels=CLASES, cbar=False)
        self.ax.set_xlabel('Predicción')
        self.ax.set_ylabel('Real')
        self.ax.set_title('Matriz de Confusión en Vivo')
        self.fig.tight_layout()
        self.canvas_matriz.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaPercepcionUAV(root)
    root.mainloop()