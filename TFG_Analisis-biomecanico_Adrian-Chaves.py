"""
TFG - Análisis biomecánico de la sentadilla
=============================================
Aplicación de escritorio que, a partir del vídeo de una cámara web o de un
archivo de vídeo, detecta en tiempo real los puntos anatómicos del perfil
izquierdo del cuerpo (hombro, cadera, rodilla y tobillo) mediante MediaPipe
Pose.
(Versión B con el Modelo Físico y Dinámica Inversa descendente de Newton-Euler
importados del Código A).
"""

import tkinter as tk                                   # Biblioteca estándar de Python para construir la interfaz gráfica
from tkinter import ttk, filedialog, messagebox        # Widgets mejorados y cuadros de diálogo
import cv2                                             # OpenCV: captura de vídeo y operaciones de dibujo
from PIL import Image, ImageTk                         # Pillow: conversión de fotogramas para Tkinter
import mediapipe as mp                                 # MediaPipe: modelo de estimación de pose
import math                                            # Funciones trigonométricas para ángulos
import numpy as np                                     # Operaciones con arrays y vectores matemáticos
import time                                            # Marca de tiempo de cada fotograma
import pandas as pd                                    # DataFrame para exportar a Excel
from datetime import datetime                          # Fecha/hora para el archivo exportado
from openpyxl.styles import PatternFill                # Relleno de color en Excel

###=============================== PARÁMETROS DEL MODELO FÍSICO ===============================###
# Parámetros importados del Código A para la Dinámica Inversa de Newton-Euler

G = 9.81                          # Aceleración de la gravedad (m/s²)

# Proporciones de longitud respecto a la estatura H
FRACCION_CADENA = 0.779           # Distancia hombro-tobillo / H, usada en calibración
FRACCION_HAT = 0.489              # Longitud del HAT (tronco) / H
FRACCION_MUSLO = 0.237            # Longitud del muslo / H
FRACCION_PIERNA = 0.231           # Longitud de la pierna / H

# Fracciones de masa respecto a la masa corporal total
MASA_HAT = 0.678                  # Cabeza, brazos y tronco
MASA_MUSLOS = 0.100 * 2           # Ambos muslos
MASA_PIERNAS = 0.046 * 2          # Ambas piernas

# Posición del centro de masa (fracción de la longitud del segmento)
CDG_HAT = 0.374                   # Desde la cadera
CDG_MUSLO = 0.433                 # Desde la cadera
CDG_PIERNA = 0.433                # Desde la rodilla

# Radios de giro respecto al centro de masa
RADIO_GIRO_HAT = 0.307
RADIO_GIRO_MUSLO = 0.323
RADIO_GIRO_PIERNA = 0.302

###=============================== FUNCIONES MATEMÁTICAS DINÁMICAS ===============================###

def producto_vectorial(r, f):
    """Componente z del producto vectorial plano r × f."""
    return r[0] * f[1] - r[1] * f[0]

def resolver_dinamica_inversa_simplificada(pos, acel, alfa, masa, inercia):
    """
    Dinámica inversa descendente (Newton-Euler) adaptada para 4 puntos (Tronco, Muslo, Pierna).
    El sistema requiere vectores 2D reales y gravedad. Ejes: X hacia delante (sentido de avance
    del sujeto), Y hacia arriba. Momentos de cadera y rodilla positivos extensores; tobillo
    positivo flexor plantar.
    """
    g = np.array([0.0, -G])

    # Tronco (HAT): extremo superior libre. Obtenemos reacción en cadera.
    F_B = masa['t'] * acel['Gt'] - masa['t'] * g
    M_B = inercia['t'] * alfa['t'] - producto_vectorial(pos['B'] - pos['Gt'], F_B)

    # Muslo: recibe reacción del tronco y ejerce sobre rodilla.
    F_C = masa['m'] * acel['Gm'] - masa['m'] * g + F_B
    M_C = (inercia['m'] * alfa['m'] + M_B
           + producto_vectorial(pos['B'] - pos['Gm'], F_B)
           - producto_vectorial(pos['C'] - pos['Gm'], F_C))

    # Pierna: recibe reacción del muslo y ejerce sobre tobillo.
    F_D = masa['c'] * acel['Gc'] - masa['c'] * g + F_C
    M_D = (inercia['c'] * alfa['c'] + M_C
           + producto_vectorial(pos['C'] - pos['Gc'], F_C)
           - producto_vectorial(pos['D'] - pos['Gc'], F_D))

    return {
        'F_cadera': F_B, 'F_rodilla': F_C, 'F_tobillo': F_D,
        'M_cadera': M_B, 'M_rodilla': -M_C, 'M_tobillo': M_D
    }


class Punto:
    """Punto de referencia auxiliar (no proviene de MediaPipe)."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

class AppBiomecanica:
    def __init__(self, root):
        self.root = root
        self.root.title("TFG - Análisis biomecánico de la sentadilla")
        self.root.geometry("1550x750")
        self.root.resizable(False, False)

        self.pose = None
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.mp_drawing = mp.solutions.drawing_utils
            print("Mediapipe OK")
        except Exception as e:
            print(f"Error... Fallo al cargar mediapipe: {e}")

        self.cap = None
        self.run = False
        self.es_video = False      # True al analizar un archivo: se usa la marca de tiempo del vídeo en lugar del reloj
        self.t_video_ant = -1.0    # Última marca de tiempo leída del archivo de vídeo

        self.checkbox_triangulos = tk.BooleanVar(value=True)
        self.checkbox_grados = tk.BooleanVar(value=True)
        self.checkbox_dist = tk.BooleanVar(value=True)
        self.checkbox_veloc_lineal = tk.BooleanVar(value=True)
        self.checkbox_acel_lineal = tk.BooleanVar(value=True)
        self.checkbox_veloc_ang = tk.BooleanVar(value=True)
        self.checkbox_acel_ang = tk.BooleanVar(value=True)
        self.checkbox_fuerza = tk.BooleanVar(value=True)
        self.checkbox_momento = tk.BooleanVar(value=True)

        self.entrada_altura = tk.StringVar(value="175")
        self.entrada_peso = tk.StringVar(value="75")

        self.contador = 0
        self.estado_posicion = "arriba"
        self.rep_en_progreso = False
        self.reps_guardadas = []
        self.datos_rep_actual = self.iniciar_datos_rep()

        # --- VARIABLES PREVIAS (Adaptadas para vectores 2D y cinemática rigurosa) ---
        self.tiempo_ant = None
        self.escala_ant = None

        # Vectores 2D para la dinámica de Newton-Euler
        self.pGt_ant = None
        self.pGm_ant = None
        self.pGc_ant = None
        self.vel_Gt = np.zeros(2); self.vel_Gm = np.zeros(2); self.vel_Gc = np.zeros(2)
        self.acel_Gt = np.zeros(2); self.acel_Gm = np.zeros(2); self.acel_Gc = np.zeros(2)

        # Cinemática angular absoluta de segmentos (para dinámica)
        self.ang_seg_t_ant = None; self.ang_seg_m_ant = None; self.ang_seg_c_ant = None
        self.w_seg_t = 0.0; self.w_seg_m = 0.0; self.w_seg_c = 0.0
        self.alfa_seg_t = 0.0; self.alfa_seg_m = 0.0; self.alfa_seg_c = 0.0

        # Variables originales de la UI para cinemática lineal escalar (magnitudes)
        self.velocidad_lineal_AB = 0.0; self.velocidad_lineal_BC = 0.0; self.velocidad_lineal_CD = 0.0
        self.aceleracion_AB = 0.0; self.aceleracion_BC = 0.0; self.aceleracion_CD = 0.0

        # Variables originales de cinemática angular de articulaciones (solo UI)
        self.angulo_tronco_ant = None
        self.angulo_cadera_ant = None
        self.angulo_rodilla_ant = None
        self.angulo_pie_ant = None
        self.velocidad_angular_tronco = 0.0; self.velocidad_angular_cadera = 0.0
        self.velocidad_angular_rodilla = 0.0; self.velocidad_angular_pie = 0.0
        self.velocidad_angular_tronco_ant = 0.0; self.velocidad_angular_cadera_ant = 0.0
        self.velocidad_angular_rodilla_ant = 0.0; self.velocidad_angular_pie_ant = 0.0
        self.aceleracion_angular_tronco = 0.0; self.aceleracion_angular_cadera = 0.0
        self.aceleracion_angular_rodilla = 0.0; self.aceleracion_angular_pie = 0.0

        # Interfaz de la Tabla
        self.frame_tabla = ttk.LabelFrame(self.root, text="Vectores Biomecánicos (Perfil Izq.)")
        self.frame_tabla.place(x=20, y=20, width=820, height=450)
        self.contenedor = ttk.Frame(self.frame_tabla)
        self.contenedor.pack(fill="both", expand=True, padx=5, pady=5)

        cols = ("Puntos", "Longitud", "Angulo", "VelLin", "AcelLin", "VelAng", "AcelAng", "Fuerza", "Momento", "Evaluacion")
        self.tabla = ttk.Treeview(self.contenedor, columns=cols, show="headings")

        self.tabla.heading("Puntos", text="Segmento / θ")
        self.tabla.heading("Longitud", text="Long(cm)")
        self.tabla.heading("Angulo", text="Áng(°)")
        self.tabla.heading("VelLin", text="V(m/s)")
        self.tabla.heading("AcelLin", text="a(m/s²)")
        self.tabla.heading("VelAng", text="w(rad/s)")
        self.tabla.heading("AcelAng", text="α(rad/s²)")
        self.tabla.heading("Fuerza", text="F(N)")
        self.tabla.heading("Momento", text="M(Nm)")
        self.tabla.heading("Evaluacion", text="Evaluación")

        self.tabla.column("Puntos", width=110, anchor="center")
        self.tabla.column("Longitud", width=65, anchor="center")
        self.tabla.column("Angulo", width=55, anchor="center")
        self.tabla.column("VelLin", width=60, anchor="center")
        self.tabla.column("AcelLin", width=65, anchor="center")
        self.tabla.column("VelAng", width=75, anchor="center")
        self.tabla.column("AcelAng", width=80, anchor="center")
        self.tabla.column("Fuerza", width=60, anchor="center")
        self.tabla.column("Momento", width=65, anchor="center")
        self.tabla.column("Evaluacion", width=90, anchor="center")

        self.tabla.tag_configure("tronco", background="#ADD8E6")
        self.tabla.tag_configure("cadera", background="#FFFACD")
        self.tabla.tag_configure("rodilla", background="#90EE90")
        self.tabla.tag_configure("pie", background="#DDA0DD")
        self.tabla.tag_configure("separador", background="#F0F0F0")
        scroll = ttk.Scrollbar(self.contenedor, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=scroll.set)
        self.tabla.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.lbl_titulo = ttk.Label(self.root, text="VISOR DE ANÁLISIS BIOMECÁNICO EN TIEMPO REAL", font=("Arial", 10, "bold"))
        self.lbl_titulo.place(x=860, y=20)
        self.canvas_vid = tk.Label(self.root, bg="black")
        self.canvas_vid.place(x=860, y=50, width=640, height=360)

        self.frame_ctrl = ttk.LabelFrame(self.root, text="Controles y Configuración")
        self.frame_ctrl.place(x=20, y=490, width=600, height=140)

        ttk.Button(self.frame_ctrl, text="Cargar Video", command=self.cargar_video).grid(row=0, column=0, padx=10, pady=15)
        ttk.Button(self.frame_ctrl, text="Detener Cámara", command=self.parar).grid(row=0, column=1, padx=10, pady=15)
        ttk.Button(self.frame_ctrl, text="Iniciar Cámara", command=self.encender_camara).grid(row=0, column=2, padx=10, pady=15)
        ttk.Button(self.frame_ctrl, text="Exportar Reporte", command=self.guardar_excel).grid(row=0, column=3, padx=10, pady=15)

        self.box_params = ttk.Frame(self.frame_ctrl)
        self.box_params.grid(row=1, column=0, columnspan=4, pady=5, sticky="w")
        ttk.Label(self.box_params, text="Tu Altura (cm):").pack(side="left", padx=10)
        ttk.Entry(self.box_params, textvariable=self.entrada_altura, width=6).pack(side="left", padx=5)
        ttk.Label(self.box_params, text="Tu Peso (kg):").pack(side="left", padx=25)
        ttk.Entry(self.box_params, textvariable=self.entrada_peso, width=6).pack(side="left", padx=5)

        self.frame_cont = ttk.LabelFrame(self.root, text="Sentadillas Correctas")
        self.frame_cont.place(x=860, y=430, width=450, height=90)
        self.lbl_cont = ttk.Label(self.frame_cont, text="0", font=("Arial", 28, "bold"), foreground="green")
        self.lbl_cont.pack(side="left", padx=30, pady=5)
        ttk.Button(self.frame_cont, text="Resetear", command=self.reset).pack(side="right", padx=30, pady=5)

        self.frame_opc = ttk.LabelFrame(self.root, text="Visualización en Video")
        self.frame_opc.place(x=860, y=540, width=640, height=100)
        fila1 = ttk.Frame(self.frame_opc)
        fila1.pack(fill="x", pady=6)
        ttk.Checkbutton(fila1, text="Triángulos", variable=self.checkbox_triangulos).pack(side="left", padx=10)
        ttk.Checkbutton(fila1, text="Grados", variable=self.checkbox_grados).pack(side="left", padx=10)
        ttk.Checkbutton(fila1, text="Dist(cm)", variable=self.checkbox_dist).pack(side="left", padx=10)
        ttk.Checkbutton(fila1, text="Veloc.Lin", variable=self.checkbox_veloc_lineal).pack(side="left", padx=10)
        ttk.Checkbutton(fila1, text="Acel.Lin", variable=self.checkbox_acel_lineal).pack(side="left", padx=10)
        fila2 = ttk.Frame(self.frame_opc)
        fila2.pack(fill="x", pady=4)
        ttk.Checkbutton(fila2, text="Veloc.Ang", variable=self.checkbox_veloc_ang).pack(side="left", padx=10)
        ttk.Checkbutton(fila2, text="Acel.Ang", variable=self.checkbox_acel_ang).pack(side="left", padx=10)
        ttk.Checkbutton(fila2, text="Fuerzas (N)", variable=self.checkbox_fuerza).pack(side="left", padx=10)
        ttk.Checkbutton(fila2, text="Momentos (Nm)", variable=self.checkbox_momento).pack(side="left", padx=10)

    def calcular_angulo(self, punto1, punto2, punto3):
        rad = math.atan2(punto3.y - punto2.y, punto3.x - punto2.x) - math.atan2(punto1.y - punto2.y, punto1.x - punto2.x)
        angulo = abs(rad * 180.0 / math.pi)
        if angulo > 180.0: angulo = 360.0 - angulo
        return angulo

    def calcular_distancia(self, punto1, punto2, w, h):
        return math.hypot(punto2[0] - punto1[0], punto2[1] - punto1[1])

    def iniciar_datos_rep(self):
        # "incl_max" registra el cierre máximo de la cadera, es decir, el ángulo de cadera mínimo (θc,mín)
        return {"prof_max": 180.0, "incl_max": 180.0, "vmax": 0.0, "fmax": 0.0, "mmax": 0.0}

    def limpiar_memoria(self):
        self.reps_guardadas = []
        self.datos_rep_actual = self.iniciar_datos_rep()
        self.contador = 0
        self.estado_posicion = "arriba"
        self.rep_en_progreso = False
        self.lbl_cont.config(text="0")

    def reset(self):
        self.limpiar_memoria()

    def reiniciar_derivadas(self):
        """Reinicia las posiciones previas y los filtros al empezar una captura."""
        self.tiempo_ant = None
        self.escala_ant = None
        self.pGt_ant = None
        self.vel_Gt = np.zeros(2); self.vel_Gm = np.zeros(2); self.vel_Gc = np.zeros(2)
        self.acel_Gt = np.zeros(2); self.acel_Gm = np.zeros(2); self.acel_Gc = np.zeros(2)
        self.w_seg_t = 0.0; self.w_seg_m = 0.0; self.w_seg_c = 0.0
        self.alfa_seg_t = 0.0; self.alfa_seg_m = 0.0; self.alfa_seg_c = 0.0

    def encender_camara(self):
        if not self.run:
            self.limpiar_memoria()
            self.reiniciar_derivadas()
            self.cap = cv2.VideoCapture(0)
            self.es_video = False
            self.run = True
            self.procesar_frame()

    def parar(self):
        self.run = False
        self.tiempo_ant = None
        self.pGt_ant = None
        if self.cap: self.cap.release()
        self.canvas_vid.config(image="")
        for item in self.tabla.get_children():
            self.tabla.delete(item)

    def cargar_video(self):
        ruta = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov")])
        if ruta:
            self.parar()
            self.limpiar_memoria()
            self.reiniciar_derivadas()
            self.cap = cv2.VideoCapture(ruta)
            self.es_video = True
            self.t_video_ant = -1.0
            self.run = True
            self.procesar_frame()

    def marca_de_tiempo(self):
        """Tiempo del fotograma: reloj del sistema con la cámara; marca de tiempo del archivo con un vídeo,
        para que las velocidades no dependan de la velocidad de procesado del ordenador."""
        if not self.es_video:
            return time.time()
        t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not t > self.t_video_ant:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            t = self.t_video_ant + 1.0 / (fps if fps and fps > 1 else 30.0)
        self.t_video_ant = t
        return t

    def guardar_excel(self):
        if not self.reps_guardadas:
            messagebox.showwarning("Error", "Haz alguna sentadilla primero antes de exportar.")
            return
        name = f"Reporte_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")], initialfile=name)
        if path:
            try:
                df = pd.DataFrame(self.reps_guardadas)
                with pd.ExcelWriter(path, engine='openpyxl') as w:
                    df.to_excel(w, sheet_name='Resultados', index=False)
                    hoja = w.sheets['Resultados']
                    fondo_verde = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid")
                    fondo_rojo = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")
                    for r in range(2, len(df) + 2):
                        txt_res = str(hoja.cell(row=r, column=2).value).lower()

                        if "válida" in txt_res:
                            for c in range(1, 8):
                                hoja.cell(row=r, column=c).fill = fondo_verde
                        else:
                            if "profundidad" in txt_res or "múltiples" in txt_res:
                                hoja.cell(row=r, column=3).fill = fondo_rojo
                            if "pecho" in txt_res or "múltiples" in txt_res:
                                hoja.cell(row=r, column=4).fill = fondo_rojo
                messagebox.showinfo("Ok!", f"Se guardó todo en:\n{path}")
            except Exception as e:
                messagebox.showerror("Fallo", f"Ocurrió un error al guardar:\n{e}")

    def procesar_frame(self):
        if self.run and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_draw = rgb.copy()
                h, w, _ = rgb.shape
                t_ahora = self.marca_de_tiempo()

                if self.pose is not None:
                    res = self.pose.process(rgb)
                    if res.pose_landmarks:
                        puntos_mediapipe = res.pose_landmarks.landmark
                        puntoHombro = puntos_mediapipe[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        puntoCadera = puntos_mediapipe[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                        puntoRodilla = puntos_mediapipe[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
                        puntoPie = puntos_mediapipe[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                        puntoTalon = puntos_mediapipe[self.mp_pose.PoseLandmark.LEFT_HEEL.value]
                        puntoPunta = puntos_mediapipe[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

                        pixelA = (int(puntoHombro.x * w), int(puntoHombro.y * h))
                        pixelB = (int(puntoCadera.x * w), int(puntoCadera.y * h))
                        pixelC = (int(puntoRodilla.x * w), int(puntoRodilla.y * h))
                        pixelD = (int(puntoPie.x * w), int(puntoPie.y * h))

                        punto_aux_vert = (pixelB[0], pixelA[1])
                        punto_aux_horiz = (pixelC[0], pixelD[1])
                        punto_medio_AB = ((pixelA[0] + pixelB[0]) / 2.0, (pixelA[1] + pixelB[1]) / 2.0)
                        punto_medio_BC = ((pixelB[0] + pixelC[0]) / 2.0, (pixelB[1] + pixelC[1]) / 2.0)
                        punto_medio_CD = ((pixelC[0] + pixelD[0]) / 2.0, (pixelC[1] + pixelD[1]) / 2.0)

                        angulo_cadera = self.calcular_angulo(puntoHombro, puntoCadera, puntoRodilla)
                        angulo_rodilla = self.calcular_angulo(puntoCadera, puntoRodilla, puntoPie)
                        angulo_tronco = self.calcular_angulo(puntoHombro, puntoCadera, Punto(puntoCadera.x, puntoHombro.y))
                        angulo_pie = self.calcular_angulo(puntoRodilla, puntoPie, Punto(puntoRodilla.x, puntoPie.y))
                        distancia_ab = self.calcular_distancia(pixelA, pixelB, w, h)
                        distancia_bc = self.calcular_distancia(pixelB, pixelC, w, h)
                        distancia_cd = self.calcular_distancia(pixelC, pixelD, w, h)

                        try: altura = float(self.entrada_altura.get())
                        except ValueError: altura = 175.0
                        try: peso = float(self.entrada_peso.get())
                        except ValueError: peso = 75.0

                        suma_pixeles = distancia_ab + distancia_bc + distancia_cd
                        if suma_pixeles > 0: relacion_cm_px = (altura * FRACCION_CADENA) / suma_pixeles
                        else: relacion_cm_px = 0

                        longitud_ab = distancia_ab * relacion_cm_px
                        longitud_bc = distancia_bc * relacion_cm_px
                        longitud_cd = distancia_cd * relacion_cm_px

                        # === INTEGRACIÓN FÍSICA A (Masas, inercias y vectores 2D) ===
                        estatura = altura / 100.0
                        escala = relacion_cm_px / 100.0
                        # Ejes físicos: X hacia delante (sentido de avance, según la orientación talón-punta del pie)
                        # e Y hacia arriba, para que el signo de los momentos no dependa de hacia dónde mire el sujeto
                        sentido = 1.0 if puntoPunta.x >= puntoTalon.x else -1.0
                        signo = np.array([sentido, -1.0])

                        pA = np.array(pixelA) * signo * escala
                        pB = np.array(pixelB) * signo * escala
                        pC = np.array(pixelC) * signo * escala
                        pD = np.array(pixelD) * signo * escala

                        # Posición real de los Centros de Gravedad
                        u_tronco = (pA - pB) / max(np.linalg.norm(pA - pB), 1e-9)
                        pGt = pB + (CDG_HAT * FRACCION_HAT * estatura) * u_tronco
                        pGm = pB + CDG_MUSLO * (pC - pB)
                        pGc = pC + CDG_PIERNA * (pD - pC)
                        pos_fisicas = {'A': pA, 'B': pB, 'C': pC, 'D': pD, 'Gt': pGt, 'Gm': pGm, 'Gc': pGc}

                        # Ángulos absolutos de segmentos para cinemática angular
                        def orientacion(p_ini, p_fin):
                            d = p_fin - p_ini
                            return math.atan2(d[1], d[0])
                        ang_seg_t = orientacion(pB, pA)
                        ang_seg_m = orientacion(pB, pC)
                        ang_seg_c = orientacion(pC, pD)

                        masa = {'t': MASA_HAT * peso, 'm': MASA_MUSLOS * peso, 'c': MASA_PIERNAS * peso}
                        inercia = {
                            't': masa['t'] * (RADIO_GIRO_HAT * FRACCION_HAT * estatura) ** 2,
                            'm': masa['m'] * (RADIO_GIRO_MUSLO * FRACCION_MUSLO * estatura) ** 2,
                            'c': masa['c'] * (RADIO_GIRO_PIERNA * FRACCION_PIERNA * estatura) ** 2,
                        }

                        Fuerza_cadera = 0.0; Fuerza_rodilla = 0.0; Fuerza_tobillo = 0.0
                        Momento_cadera = 0.0; Momento_rodilla = 0.0; Momento_tobillo = 0.0

                        if self.tiempo_ant is not None and self.pGt_ant is not None and escala > 0:
                            dt = t_ahora - self.tiempo_ant
                            if dt > 0:
                                alf_v = 0.2; alf_a = 0.15
                                # Cinemática Vectorial 2D (Lineal): el desplazamiento se mide en píxeles y se convierte
                                # con la escala actual, para que la variación del factor de escala entre fotogramas no
                                # se interprete como movimiento
                                vel_Gt_bruta = (pGt / escala - self.pGt_ant / self.escala_ant) * escala / dt
                                vel_Gm_bruta = (pGm / escala - self.pGm_ant / self.escala_ant) * escala / dt
                                vel_Gc_bruta = (pGc / escala - self.pGc_ant / self.escala_ant) * escala / dt

                                # La aceleración se obtiene derivando la velocidad ya suavizada (filtro EMA)
                                vel_Gt_nueva = (1 - alf_v) * self.vel_Gt + alf_v * vel_Gt_bruta
                                vel_Gm_nueva = (1 - alf_v) * self.vel_Gm + alf_v * vel_Gm_bruta
                                vel_Gc_nueva = (1 - alf_v) * self.vel_Gc + alf_v * vel_Gc_bruta
                                self.acel_Gt = (1 - alf_a) * self.acel_Gt + alf_a * (vel_Gt_nueva - self.vel_Gt) / dt
                                self.acel_Gm = (1 - alf_a) * self.acel_Gm + alf_a * (vel_Gm_nueva - self.vel_Gm) / dt
                                self.acel_Gc = (1 - alf_a) * self.acel_Gc + alf_a * (vel_Gc_nueva - self.vel_Gc) / dt
                                self.vel_Gt, self.vel_Gm, self.vel_Gc = vel_Gt_nueva, vel_Gm_nueva, vel_Gc_nueva

                                # Extraemos escalares (módulos) para rellenar la UI como lo hacía el Código B
                                self.velocidad_lineal_AB = float(np.linalg.norm(self.vel_Gt))
                                self.velocidad_lineal_BC = float(np.linalg.norm(self.vel_Gm))
                                self.velocidad_lineal_CD = float(np.linalg.norm(self.vel_Gc))
                                self.aceleracion_AB = float(np.linalg.norm(self.acel_Gt))
                                self.aceleracion_BC = float(np.linalg.norm(self.acel_Gm))
                                self.aceleracion_CD = float(np.linalg.norm(self.acel_Gc))

                                # Cinemática Angular Absoluta (Para Dinámica), con el mismo esquema de filtrado
                                inc_t = (ang_seg_t - self.ang_seg_t_ant + math.pi) % (2 * math.pi) - math.pi
                                inc_m = (ang_seg_m - self.ang_seg_m_ant + math.pi) % (2 * math.pi) - math.pi
                                inc_c = (ang_seg_c - self.ang_seg_c_ant + math.pi) % (2 * math.pi) - math.pi
                                w_t_nueva = (1 - alf_v) * self.w_seg_t + alf_v * inc_t / dt
                                w_m_nueva = (1 - alf_v) * self.w_seg_m + alf_v * inc_m / dt
                                w_c_nueva = (1 - alf_v) * self.w_seg_c + alf_v * inc_c / dt
                                self.alfa_seg_t = (1 - alf_a) * self.alfa_seg_t + alf_a * (w_t_nueva - self.w_seg_t) / dt
                                self.alfa_seg_m = (1 - alf_a) * self.alfa_seg_m + alf_a * (w_m_nueva - self.w_seg_m) / dt
                                self.alfa_seg_c = (1 - alf_a) * self.alfa_seg_c + alf_a * (w_c_nueva - self.w_seg_c) / dt
                                self.w_seg_t, self.w_seg_m, self.w_seg_c = w_t_nueva, w_m_nueva, w_c_nueva

                                # Cinemática Angular de Articulaciones (Solo para dibujar en UI, idéntico al Código B original)
                                velocidad_angular_tronco = (math.radians(angulo_tronco) - math.radians(self.angulo_tronco_ant)) / dt
                                velocidad_angular_cadera = (math.radians(angulo_cadera) - math.radians(self.angulo_cadera_ant)) / dt
                                velocidad_angular_rodilla = (math.radians(angulo_rodilla) - math.radians(self.angulo_rodilla_ant)) / dt
                                velocidad_angular_pie = (math.radians(angulo_pie) - math.radians(self.angulo_pie_ant)) / dt
                                aceleracion_angular_tronco = (velocidad_angular_tronco - self.velocidad_angular_tronco_ant) / dt
                                aceleracion_angular_cadera = (velocidad_angular_cadera - self.velocidad_angular_cadera_ant) / dt
                                aceleracion_angular_rodilla = (velocidad_angular_rodilla - self.velocidad_angular_rodilla_ant) / dt
                                aceleracion_angular_pie = (velocidad_angular_pie - self.velocidad_angular_pie_ant) / dt
                                self.velocidad_angular_tronco = (1 - alf_v) * self.velocidad_angular_tronco + alf_v * velocidad_angular_tronco
                                self.velocidad_angular_cadera = (1 - alf_v) * self.velocidad_angular_cadera + alf_v * velocidad_angular_cadera
                                self.velocidad_angular_rodilla = (1 - alf_v) * self.velocidad_angular_rodilla + alf_v * velocidad_angular_rodilla
                                self.velocidad_angular_pie = (1 - alf_v) * self.velocidad_angular_pie + alf_v * velocidad_angular_pie
                                self.aceleracion_angular_tronco = (1 - alf_a) * self.aceleracion_angular_tronco + alf_a * aceleracion_angular_tronco
                                self.aceleracion_angular_cadera = (1 - alf_a) * self.aceleracion_angular_cadera + alf_a * aceleracion_angular_cadera
                                self.aceleracion_angular_rodilla = (1 - alf_a) * self.aceleracion_angular_rodilla + alf_a * aceleracion_angular_rodilla
                                self.aceleracion_angular_pie = (1 - alf_a) * self.aceleracion_angular_pie + alf_a * aceleracion_angular_pie
                                self.velocidad_angular_tronco_ant = velocidad_angular_tronco
                                self.velocidad_angular_cadera_ant = velocidad_angular_cadera
                                self.velocidad_angular_rodilla_ant = velocidad_angular_rodilla
                                self.velocidad_angular_pie_ant = velocidad_angular_pie

                                # Aplicación de Newton-Euler
                                acel_lin = {'Gt': self.acel_Gt, 'Gm': self.acel_Gm, 'Gc': self.acel_Gc}
                                acel_ang = {'t': self.alfa_seg_t, 'm': self.alfa_seg_m, 'c': self.alfa_seg_c}
                                din = resolver_dinamica_inversa_simplificada(pos_fisicas, acel_lin, acel_ang, masa, inercia)

                                # Fuerzas de reacción (módulo) y momentos articulares netos en cada articulación
                                Fuerza_cadera = float(np.linalg.norm(din['F_cadera']))
                                Fuerza_rodilla = float(np.linalg.norm(din['F_rodilla']))
                                Fuerza_tobillo = float(np.linalg.norm(din['F_tobillo']))
                                Momento_cadera = float(din['M_cadera'])
                                Momento_rodilla = float(din['M_rodilla'])
                                Momento_tobillo = float(din['M_tobillo'])

                        # Guardar estados previos
                        self.tiempo_ant = t_ahora
                        self.escala_ant = escala
                        self.pGt_ant = pGt
                        self.pGm_ant = pGm
                        self.pGc_ant = pGc
                        self.ang_seg_t_ant = ang_seg_t
                        self.ang_seg_m_ant = ang_seg_m
                        self.ang_seg_c_ant = ang_seg_c
                        self.angulo_tronco_ant = angulo_tronco
                        self.angulo_cadera_ant = angulo_cadera
                        self.angulo_rodilla_ant = angulo_rodilla
                        self.angulo_pie_ant = angulo_pie

                        #================== DIBUJOS EN IMAGEN ==================#
                        if self.checkbox_triangulos.get():
                            capa = frame_draw.copy()
                            cv2.fillPoly(capa, [np.array([pixelA, pixelB, pixelC], dtype=np.int32)], (255, 250, 205))
                            cv2.fillPoly(capa, [np.array([pixelB, pixelC, pixelD], dtype=np.int32)], (144, 238, 144))
                            cv2.fillPoly(capa, [np.array([pixelA, pixelB, punto_aux_vert], dtype=np.int32)], (173, 216, 230))
                            cv2.fillPoly(capa, [np.array([pixelC, pixelD, punto_aux_horiz], dtype=np.int32)], (221, 160, 221))
                            cv2.addWeighted(capa, 0.4, frame_draw, 0.6, 0, frame_draw)
                        self.mp_drawing.draw_landmarks(frame_draw, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                        fnt = cv2.FONT_HERSHEY_SIMPLEX

                        if self.checkbox_grados.get():
                            cv2.putText(frame_draw, f"{int(angulo_tronco)}o", (pixelA[0]-40, pixelA[1]), fnt, 0.6, (0,0,255), 2)
                            cv2.putText(frame_draw, f"{int(angulo_cadera)}o", (pixelB[0]-40, pixelB[1]+20), fnt, 0.6, (0,0,255), 2)
                            cv2.putText(frame_draw, f"{int(angulo_rodilla)}o", (pixelC[0]+15, pixelC[1]), fnt, 0.6, (0,0,255), 2)
                            cv2.putText(frame_draw, f"{int(angulo_pie)}o", (pixelD[0]+15, pixelD[1]), fnt, 0.6, (0,0,255), 2)

                        if self.checkbox_dist.get():
                            cv2.putText(frame_draw, f"{int(longitud_ab)}cm", (int(punto_medio_AB[0])-15, int(punto_medio_AB[1])), fnt, 0.5, (75, 0, 130), 2)
                            cv2.putText(frame_draw, f"{int(longitud_bc)}cm", (int(punto_medio_BC[0])-15, int(punto_medio_BC[1])), fnt, 0.5, (75, 0, 130), 2)
                            cv2.putText(frame_draw, f"{int(longitud_cd)}cm", (int(punto_medio_CD[0])-15, int(punto_medio_CD[1])), fnt, 0.5, (75, 0, 130), 2)

                        if self.checkbox_veloc_lineal.get():
                            cv2.putText(frame_draw, f"{self.velocidad_lineal_AB:.1f}m/s", (int(punto_medio_AB[0])+25, int(punto_medio_AB[1])+15), fnt, 0.5, (0, 100, 255), 2)
                            cv2.putText(frame_draw, f"{self.velocidad_lineal_BC:.1f}m/s", (int(punto_medio_BC[0])+25, int(punto_medio_BC[1])+15), fnt, 0.5, (0, 100, 255), 2)
                            cv2.putText(frame_draw, f"{self.velocidad_lineal_CD:.1f}m/s", (int(punto_medio_CD[0])+25, int(punto_medio_CD[1])+15), fnt, 0.5, (0, 100, 255), 2)

                        if self.checkbox_acel_lineal.get():
                            cv2.putText(frame_draw, f"{self.aceleracion_AB:.1f}m/s2", (int(punto_medio_AB[0])+25, int(punto_medio_AB[1])-15), fnt, 0.5, (255, 255, 0), 2)
                            cv2.putText(frame_draw, f"{self.aceleracion_BC:.1f}m/s2", (int(punto_medio_BC[0])+25, int(punto_medio_BC[1])-15), fnt, 0.5, (255, 255, 0), 2)
                            cv2.putText(frame_draw, f"{self.aceleracion_CD:.1f}m/s2", (int(punto_medio_CD[0])+25, int(punto_medio_CD[1])-15), fnt, 0.5, (255, 255, 0), 2)

                        if self.checkbox_veloc_ang.get():
                            cv2.putText(frame_draw, f"w:{self.velocidad_angular_tronco:.1f}", (pixelA[0]-95, pixelA[1]-15), fnt, 0.5, (42, 42, 165), 2)
                            cv2.putText(frame_draw, f"w:{self.velocidad_angular_cadera:.1f}", (pixelB[0]-95, pixelB[1]-15), fnt, 0.5, (42, 42, 165), 2)
                            cv2.putText(frame_draw, f"w:{self.velocidad_angular_rodilla:.1f}", (pixelC[0]+15, pixelC[1]-30), fnt, 0.5, (42, 42, 165), 2)
                            cv2.putText(frame_draw, f"w:{self.velocidad_angular_pie:.1f}", (pixelD[0]+15, pixelD[1]-30), fnt, 0.5, (42, 42, 165), 2)

                        if self.checkbox_acel_ang.get():
                            cv2.putText(frame_draw, f"a:{self.aceleracion_angular_tronco:.1f}", (pixelA[0]-95, pixelA[1]+15), fnt, 0.5, (147, 20, 255), 2)
                            cv2.putText(frame_draw, f"a:{self.aceleracion_angular_cadera:.1f}", (pixelB[0]-95, pixelB[1]+15), fnt, 0.5, (147, 20, 255), 2)
                            cv2.putText(frame_draw, f"a:{self.aceleracion_angular_rodilla:.1f}", (pixelC[0]+15, pixelC[1]-15), fnt, 0.5, (147, 20, 255), 2)
                            cv2.putText(frame_draw, f"a:{self.aceleracion_angular_pie:.1f}", (pixelD[0]+15, pixelD[1]-15), fnt, 0.5, (147, 20, 255), 2)

                        if self.checkbox_fuerza.get():
                            # Fuerza de reacción en cada articulación (cadera, rodilla y tobillo)
                            cv2.putText(frame_draw, f"{Fuerza_cadera:.0f}N", (pixelB[0]+15, pixelB[1]+40), fnt, 0.5, (0, 0, 255), 2)
                            cv2.putText(frame_draw, f"{Fuerza_rodilla:.0f}N", (pixelC[0]+40, pixelC[1]+40), fnt, 0.5, (0, 0, 255), 2)
                            cv2.putText(frame_draw, f"{Fuerza_tobillo:.0f}N", (pixelD[0]+15, pixelD[1]+40), fnt, 0.5, (0, 0, 255), 2)

                        if self.checkbox_momento.get():
                            # Momento articular neto en cada articulación (positivo extensor; flexor plantar en el tobillo)
                            cv2.putText(frame_draw, f"{Momento_cadera:.1f}Nm", (pixelB[0]+15, pixelB[1]-20), fnt, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame_draw, f"{Momento_rodilla:.1f}Nm", (pixelC[0]+40, pixelC[1]+20), fnt, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame_draw, f"{Momento_tobillo:.1f}Nm", (pixelD[0]+15, pixelD[1]+20), fnt, 0.5, (0, 255, 0), 2)

                        ####================= CONDICIONES DE EVALUACION DE SENTADILLA ==================####
                        if angulo_rodilla <= 155 and self.estado_posicion == "arriba":
                            self.rep_en_progreso = True
                            self.estado_posicion = "bajando"
                            self.datos_rep_actual = self.iniciar_datos_rep()

                        if self.rep_en_progreso:
                            self.datos_rep_actual["prof_max"] = min(self.datos_rep_actual["prof_max"], angulo_rodilla)
                            self.datos_rep_actual["incl_max"] = min(self.datos_rep_actual["incl_max"], angulo_cadera)   # Cierre de cadera: θc mínimo
                            self.datos_rep_actual["vmax"] = max(self.datos_rep_actual["vmax"], self.velocidad_lineal_AB, self.velocidad_lineal_BC, self.velocidad_lineal_CD)
                            self.datos_rep_actual["fmax"] = max(self.datos_rep_actual["fmax"], Fuerza_cadera, Fuerza_rodilla, Fuerza_tobillo)
                            self.datos_rep_actual["mmax"] = max(self.datos_rep_actual["mmax"], Momento_cadera, Momento_rodilla, Momento_tobillo)

                        estado = "-"
                        if angulo_rodilla > 155 and angulo_cadera > 145:
                            estado = "De pie"
                            if self.rep_en_progreso:
                                p_min = self.datos_rep_actual["prof_max"]
                                i_max = self.datos_rep_actual["incl_max"]

                                pech_ok = i_max >= 45      # Criterio de cadera: θc,mín ≥ 45°; por debajo, "pecho colapsado"
                                prof_ok = p_min <= 100     # Criterio de profundidad: θr,mín ≤ 100°
                                if pech_ok and prof_ok:
                                    res_final = "Válida"
                                    self.contador += 1
                                    self.lbl_cont.config(text=str(self.contador))
                                elif not prof_ok and not pech_ok:
                                    res_final = "Múltiples fallos"
                                elif not prof_ok:
                                    res_final = "Poca profundidad"
                                elif not pech_ok:
                                    res_final = "Pecho colapsado"

                                self.reps_guardadas.append({
                                    "Repetición": len(self.reps_guardadas) + 1,
                                    "Resultado": res_final,
                                    "Profundidad Mín (°)": round(p_min, 1),
                                    "Cadera Mín (°)": round(i_max, 1),
                                    "Velocidad Máx (m/s)": round(self.datos_rep_actual["vmax"], 2),
                                    "Fuerza Pico (N)": round(self.datos_rep_actual["fmax"], 1),
                                    "Momento Pico (Nm)": round(self.datos_rep_actual["mmax"], 1)
                                })
                            self.estado_posicion = "arriba"
                            self.rep_en_progreso = False

                        elif 100 < angulo_rodilla <= 155:
                            if self.datos_rep_actual["incl_max"] < 45:
                                estado = "Pecho colapsado"
                            else:
                                estado = "En movimiento"
                        elif angulo_rodilla <= 100:
                            if self.datos_rep_actual["incl_max"] < 45:
                                estado = "Pecho colapsado"
                            elif angulo_rodilla <= 75:
                                estado = "Profunda"
                            else:
                                estado = "Paralela"

                        # Actualización de tabla UI: las fuerzas de reacción y los momentos se muestran en la fila
                        # de la articulación correspondiente (cadera, rodilla y tobillo)
                        for val in self.tabla.get_children(): self.tabla.delete(val)

                        self.tabla.insert("", "end", values=("Tronco (AB)", int(longitud_ab), "-", f"{self.velocidad_lineal_AB:.1f}", f"{self.aceleracion_AB:.1f}", "-", "-", "-", "-", "-"))
                        self.tabla.insert("", "end", values=("Muslo (BC)", int(longitud_bc), "-", f"{self.velocidad_lineal_BC:.1f}", f"{self.aceleracion_BC:.1f}", "-", "-", "-", "-", "-"))
                        self.tabla.insert("", "end", values=("Pierna (CD)", int(longitud_cd), "-", f"{self.velocidad_lineal_CD:.1f}", f"{self.aceleracion_CD:.1f}", "-", "-", "-", "-", "-"))
                        self.tabla.insert("", "end", values=("", "", "", "", "", "", "", "", "", ""), tags=("separador",))

                        self.tabla.insert("", "end", values=("Angulo Tronco (θt)", "-", f"{int(angulo_tronco)}°", "-", "-", f"{self.velocidad_angular_tronco:.1f}", f"{self.aceleracion_angular_tronco:.1f}", "-", "-", "-"), tags=("tronco",))
                        self.tabla.insert("", "end", values=("Angulo Cadera (θc)", "-", f"{int(angulo_cadera)}°", "-", "-", f"{self.velocidad_angular_cadera:.1f}", f"{self.aceleracion_angular_cadera:.1f}", f"{Fuerza_cadera:.0f}", f"{Momento_cadera:.1f}", estado if "Pecho" in estado else "-"), tags=("cadera",))
                        self.tabla.insert("", "end", values=("Angulo Rodilla (θr)", "-", f"{int(angulo_rodilla)}°", "-", "-", f"{self.velocidad_angular_rodilla:.1f}", f"{self.aceleracion_angular_rodilla:.1f}", f"{Fuerza_rodilla:.0f}", f"{Momento_rodilla:.1f}", estado if "Pecho" not in estado else "-"), tags=("rodilla",))
                        self.tabla.insert("", "end", values=("Angulo Pie (θp)", "-", f"{int(angulo_pie)}°", "-", "-", f"{self.velocidad_angular_pie:.1f}", f"{self.aceleracion_angular_pie:.1f}", f"{Fuerza_tobillo:.0f}", f"{Momento_tobillo:.1f}", "-"), tags=("pie",))

                v_final = cv2.resize(frame_draw, (640, 360))
                img_ok = ImageTk.PhotoImage(image=Image.fromarray(v_final))
                self.canvas_vid.config(image=img_ok); self.canvas_vid.img_tk = img_ok

                self.root.after(15, self.procesar_frame)
            else:
                self.parar()

if __name__ == "__main__":
    app_raiz = tk.Tk()
    mi_app = AppBiomecanica(app_raiz)
    app_raiz.mainloop()
