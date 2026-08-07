import os
from tkinter import messagebox, ttk
import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 0. CONFIGURACIÓN DE RUTA DE TRABAJO (UMNG)
# ==========================================
script_dir = r'C:\Users\Esteb\OneDrive - unimilitar.edu.co\Universidad\9o Semestre\Inteligencia Artificial\Laboratorio\Guía #1\Códigos'

if os.path.exists(script_dir):
    os.chdir(script_dir)

# ==========================================
# 1. CARGA DE DATOS Y PREPROCESAMIENTO
# ==========================================
csv_path = 'desempeno_estudiantes.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"No se encontró el archivo '{csv_path}'. Ejecuta la generación del CSV."
    )

df = pd.read_csv(csv_path)

X = df.drop(columns=['Desempeno_Academico'])
if 'Codigo_Carrera' in X.columns:
    X = X.drop(columns=['Codigo_Carrera'])

X_encoded = pd.get_dummies(X, columns=['Carrera'], drop_first=False, dtype=int)
y = df['Desempeno_Academico']

feature_columns = X_encoded.columns

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# 2. ENTRENAMIENTO DEL MODELO OPTIMIZADO
# ==========================================
model = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    max_depth=10,
    min_samples_split=7,
    min_samples_leaf=3,
    random_state=42,
)
model.fit(X_train, y_train)


# ==========================================
# 3. INTERFAZ GRÁFICA CON COMPARACIÓN TEÓRICA
# ==========================================
class PrediccionApp:

    def __init__(self, root):
        self.root = root
        self.root.title('Comparativa Teórica vs Modelo - Random Forest')
        self.root.geometry('580x700')
        self.root.resizable(False, False)

        # Encabezado
        title_label = ttk.Label(
            root,
            text='Evaluación de Desempeño: Teórico vs. Random Forest',
            font=('Helvetica', 13, 'bold'),
        )
        title_label.pack(pady=10)

        # Contenedor de datos de entrada
        frame_input = ttk.LabelFrame(root, text=' Datos del Estudiante ')
        frame_input.pack(padx=20, pady=5, fill='both', expand=True)

        ttk.Label(frame_input, text='Carrera de Ingeniería:').grid(
            row=0, column=0, sticky='w', padx=10, pady=8
        )
        self.combo_carrera = ttk.Combobox(
            frame_input,
            values=['Mecatrónica', 'Electrónica', 'Sistemas', 'Industrial'],
            state='readonly',
        )
        self.combo_carrera.current(0)
        self.combo_carrera.grid(row=0, column=1, padx=10, pady=8)

        ttk.Label(
            frame_input, text='Horas Estudio Semanales (4.0 - 35.0):'
        ).grid(row=1, column=0, sticky='w', padx=10, pady=8)
        self.entry_horas = ttk.Entry(frame_input)
        self.entry_horas.insert(0, '18.0')
        self.entry_horas.grid(row=1, column=1, padx=10, pady=8)

        ttk.Label(frame_input, text='Promedio Acumulado (2.0 - 5.0):').grid(
            row=2, column=0, sticky='w', padx=10, pady=8
        )
        self.entry_promedio = ttk.Entry(frame_input)
        self.entry_promedio.insert(0, '3.8')
        self.entry_promedio.grid(row=2, column=1, padx=10, pady=8)

        ttk.Label(frame_input, text='Porcentaje Asistencia (50 - 100):').grid(
            row=3, column=0, sticky='w', padx=10, pady=8
        )
        self.entry_asistencia = ttk.Entry(frame_input)
        self.entry_asistencia.insert(0, '85.0')
        self.entry_asistencia.grid(row=3, column=1, padx=10, pady=8)

        # Botones
        frame_btn = ttk.Frame(root)
        frame_btn.pack(pady=10)

        btn_predict = ttk.Button(
            frame_btn, text=' Predecir y Comparar ', command=self.ejecutar_comparacion
        )
        btn_predict.grid(row=0, column=0, padx=5)

        btn_help = ttk.Button(frame_btn, text=' Ayuda ', command=self.mostrar_ayuda)
        btn_help.grid(row=0, column=1, padx=5)

        btn_about = ttk.Button(
            frame_btn, text=' Créditos ', command=self.mostrar_creditos
        )
        btn_about.grid(row=0, column=2, padx=5)

        # Panel de Resultados Comparativos
        frame_res = ttk.LabelFrame(root, text=' Resultados Comparativos ')
        frame_res.pack(padx=20, pady=5, fill='both', expand=True)

        self.lbl_teorico = ttk.Label(
            frame_res,
            text='Cálculo Teórico: ---',
            font=('Helvetica', 10),
            anchor='w',
        )
        self.lbl_teorico.pack(padx=10, pady=5, fill='x')

        self.lbl_modelo = ttk.Label(
            frame_res,
            text='Predicción Modelo: ---',
            font=('Helvetica', 10),
            anchor='w',
        )
        self.lbl_modelo.pack(padx=10, pady=5, fill='x')

        self.lbl_estado = ttk.Label(
            frame_res,
            text='Concordancia: ---',
            font=('Helvetica', 11, 'bold'),
            anchor='center',
        )
        self.lbl_estado.pack(padx=10, pady=10, fill='x')

        self.lbl_probs = ttk.Label(
            frame_res,
            text='',
            font=('Helvetica', 9),
            justify='left',
            foreground='gray',
        )
        self.lbl_probs.pack(padx=10, pady=5)

    def ejecutar_comparacion(self):
        try:
            carrera = self.combo_carrera.get()
            horas = float(self.entry_horas.get())
            promedio = float(self.entry_promedio.get())
            asistencia = float(self.entry_asistencia.get())

            if not (4.0 <= horas <= 35.0):
                raise ValueError('Horas fuera de rango (4.0 a 35.0).')
            if not (2.0 <= promedio <= 5.0):
                raise ValueError('Promedio fuera de rango (2.0 a 5.0).')
            if not (50.0 <= asistencia <= 100.0):
                raise ValueError('Asistencia fuera de rango (50% a 100%).')

            # -------------------------------------------------------------
            # 1. CÁLCULO DEL VALOR TEÓRICO ANALÍTICO (Ecuación pura sin ruido)
            # -------------------------------------------------------------
            score_teorico = (
                (promedio / 5.0) * 0.70
                + (horas / 35.0) * 0.20
                + (asistencia / 100.0) * 0.10
            )

            if score_teorico < 0.58:
                clase_teorica = 'Bajo'
            elif score_teorico < 0.74:
                clase_teorica = 'Medio'
            else:
                clase_teorica = 'Alto'

            # -------------------------------------------------------------
            # 2. CÁLCULO DADO POR EL MODELO RANDOM FOREST
            # -------------------------------------------------------------
            input_dict = {col: [0] for col in feature_columns}
            input_dict['Horas_Estudio_Semana'] = [horas]
            input_dict['Promedio_Acumulado'] = [promedio]
            input_dict['Asistencia_Pct'] = [asistencia]

            col_carrera = f'Carrera_{carrera}'
            if col_carrera in input_dict:
                input_dict[col_carrera] = [1]

            input_df = pd.DataFrame(input_dict)

            prediccion_modelo = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            prob_dict = dict(zip(model.classes_, probs))

            # -------------------------------------------------------------
            # 3. ACTUALIZACIÓN DE LA INTERFAZ CON LA COMPARACIÓN
            # -------------------------------------------------------------
            txt_teorico = (
                f'• Valor Teórico Analítico:  Score = {score_teorico:.4f}  '
                f'---> Clase: [{clase_teorica.upper()}]'
            )
            txt_modelo = (
                f'• Predicción Random Forest: Clase = [{prediccion_modelo.upper()}]'
            )

            self.lbl_teorico.config(text=txt_teorico, foreground='black')
            self.lbl_modelo.config(text=txt_modelo, foreground='black')

            # Verificar concordancia entre la teoría y la predicción
            if clase_teorica == prediccion_modelo:
                self.lbl_estado.config(
                    text='✓ COINCIDENCIA EXACTA (Teórico == Modelo)',
                    foreground='green',
                )
            else:
                self.lbl_estado.config(
                    text='⚠ DISCREPANCIA (Efecto del ruido en fronteras)',
                    foreground='darkorange',
                )

            txt_probs = (
                f'Probabilidades asignadas por el modelo:\n'
                f'  - Alto:  {prob_dict.get("Alto", 0)*100:.1f}%\n'
                f'  - Medio: {prob_dict.get("Medio", 0)*100:.1f}%\n'
                f'  - Bajo:  {prob_dict.get("Bajo", 0)*100:.1f}%'
            )
            self.lbl_probs.config(text=txt_probs)

        except ValueError as e:
            messagebox.showerror('Error de Entrada', str(e))

    def mostrar_ayuda(self):
        msg = (
            'INSTRUCCIONES DE USO:\n\n'
            '1. Ingrese los datos académicos del estudiante.\n'
            '2. Presione "Predecir y Comparar".\n'
            '3. El sistema calculará el valor teórico puro (sin ruido)\n'
            '   y lo comparará en vivo contra la predicción del Random Forest.'
        )
        messagebox.showinfo('Ayuda', msg)

    def mostrar_creditos(self):
        msg = (
            'SISTEMA DE PREDICCIÓN Y VALIDACIÓN TEÓRICA v1.1\n'
            'Algoritmo: Random Forest Classifier\n'
            'Universidad Militar Nueva Granada\n'
            'Año: 2026'
        )
        messagebox.showinfo('Créditos', msg)


if __name__ == '__main__':
    root = tk.Tk()
    app = PrediccionApp(root)
    root.mainloop()