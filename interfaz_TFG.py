"""
TFG - Análisis biomecánico de la sentadilla
=============================================
Aplicación de escritorio que, a partir del vídeo de una cámara web o de un
archivo de vídeo, detecta en tiempo real los puntos anatómicos del perfil
izquierdo del cuerpo (hombro, cadera, rodilla y tobillo) mediante MediaPipe
Pose, calcula la cinemática (ángulos, velocidades y aceleraciones lineales
y angulares) y la dinámica (fuerzas y momentos articulares mediante una
dinámica inversa simplificada) de los segmentos tronco-muslo-pierna, evalúa
automáticamente la técnica de cada repetición de sentadilla y permite
exportar un informe de la sesión en formato Excel.

La interfaz gráfica está construida con Tkinter y se actualiza fotograma a
fotograma mediante el método procesar_frame(), que se reprograma a sí mismo
con root.after() para crear un bucle de vídeo no bloqueante (sin congelar
la interfaz, a diferencia de un bucle "while" tradicional).
"""

import tkinter as tk                                  # Biblioteca estándar de Python para construir la interfaz gráfica (ventanas, botones, etiquetas, etc.)
from tkinter import ttk, filedialog, messagebox        # ttk: widgets con estilo mejorado (botones, tablas...); filedialog: cuadros de diálogo para abrir/guardar archivos; messagebox: ventanas emergentes de aviso/error
import cv2                                             # OpenCV: captura de vídeo desde cámara/archivo y operaciones de dibujo sobre los fotogramas
from PIL import Image, ImageTk                         # Pillow: convierte los fotogramas (arrays de NumPy) en imágenes que Tkinter puede mostrar en un widget
import mediapipe as mp                                 # MediaPipe: biblioteca de Google que ejecuta el modelo de estimación de pose (detección de los 33 keypoints corporales)
import math                                            # Funciones trigonométricas (atan2, radians...) para el cálculo de ángulos articulares
import numpy as np                                     # Operaciones con arrays; aquí se usa para definir los polígonos (triángulos) que se dibujan sobre la imagen
import time                                            # Marca de tiempo de cada fotograma, necesaria para calcular velocidades y aceleraciones por diferenciación numérica
import pandas as pd                                    # Estructura los datos de cada repetición en un DataFrame para exportarlos a Excel
from datetime import datetime                          # Genera la marca de fecha/hora que se incluye en el nombre del archivo de informe exportado
from openpyxl.styles import PatternFill                # Permite aplicar relleno de color a las celdas del Excel exportado (verde/rojo según el resultado de la repetición)

# clase para meter puntos de referencia falsos
class Punto:
    """
    Punto de referencia auxiliar (no proviene de MediaPipe).

    Algunos ángulos del modelo (p. ej. el del tronco o el del pie respecto
    a la vertical) no se forman entre tres landmarks reales, sino entre dos
    landmarks reales y un punto vertical "imaginario". Esta clase imita la
    interfaz de un landmark de MediaPipe (atributos .x y .y) para poder
    reutilizar la misma función calcular_angulo() también en esos casos.
    """
    def __init__(self, x, y):
        self.x = x   # Coordenada horizontal (mismo sistema de referencia que los landmarks de MediaPipe)
        self.y = y   # Coordenada vertical (mismo sistema de referencia que los landmarks de MediaPipe)

class AppBiomecanica:
    """
    Clase principal de la aplicación.

    Encapsula toda la lógica del TFG: construcción de la interfaz gráfica
    (Tkinter), carga del modelo de estimación de pose (MediaPipe), captura
    y procesado de vídeo (OpenCV), cálculo cinemático y dinámico de los
    segmentos corporales, detección de fases y evaluación automática de la
    técnica de la sentadilla, y exportación de resultados a Excel.
    """
    ###=============================== DECLARACION INTERFAZ GRAFICA (TKINTER) ===============================###
    # En esta sección se construyen todos los widgets (paneles, tabla, botones, casillas...) que forman la ventana de la aplicación.

    def __init__(self, root):
        """
        Constructor de la aplicación.

        Inicializa la ventana principal de Tkinter, carga el modelo de
        MediaPipe Pose, declara todas las variables de estado necesarias
        para el cálculo de cinemática/dinámica y construye los distintos
        paneles de la interfaz gráfica: la tabla de vectores biomecánicos,
        el visor de vídeo, los controles, el contador de repeticiones y las
        opciones de visualización.

        Parámetros:
            root (tk.Tk): ventana raíz de Tkinter sobre la que se construye
                toda la interfaz.
        """
        self.root = root
        self.root.title("TFG - Análisis biomecánico de la sentadilla")
        self.root.geometry("1550x750") 
        self.root.resizable(False, False)   # Impide redimensionar la ventana, ya que todos los widgets están posicionados con coordenadas fijas (place)

        # cargamos mediapipe
        self.pose = None 
        try:
            self.mp_pose = mp.solutions.pose                          # Submódulo de MediaPipe con el modelo de estimación de pose
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Instancia del modelo: confianza mínima del 50% tanto para detectar como para mantener el seguimiento de la persona
            self.mp_drawing = mp.solutions.drawing_utils               # Utilidades de MediaPipe para dibujar el esqueleto (landmarks y conexiones) sobre la imagen
            print("Mediapipe OK")
        except Exception as e:
            print(f"Error... Fallo al cargar mediapipe: {e}")

        self.cap = None        # Objeto cv2.VideoCapture (cámara o archivo de vídeo); se crea al pulsar "Iniciar Cámara" o "Cargar Video"
        self.run = False       # Bandera que controla si el bucle de procesado de fotogramas (procesar_frame) sigue activo

        # booleanos de la interfaz (los ticks)
        # Cada variable controla, de forma independiente, si una capa de información se dibuja sobre el vídeo
        self.checkbox_triangulos = tk.BooleanVar(value=True)    # Mostrar/ocultar los triángulos de color que delimitan los segmentos corporales
        self.checkbox_grados = tk.BooleanVar(value=True)        # Mostrar/ocultar el valor en grados de cada ángulo articular
        self.checkbox_dist = tk.BooleanVar(value=True)          # Mostrar/ocultar la longitud en cm de cada segmento
        self.checkbox_veloc_lineal = tk.BooleanVar(value=True)  # Mostrar/ocultar la velocidad lineal del centro de masa de cada segmento
        self.checkbox_acel_lineal = tk.BooleanVar(value=True)   # Mostrar/ocultar la aceleración lineal del centro de masa de cada segmento
        self.checkbox_veloc_ang = tk.BooleanVar(value=True)     # Mostrar/ocultar la velocidad angular de cada articulación
        self.checkbox_acel_ang = tk.BooleanVar(value=True)      # Mostrar/ocultar la aceleración angular de cada articulación
        self.checkbox_fuerza = tk.BooleanVar(value=True)        # Mostrar/ocultar la fuerza neta estimada sobre cada segmento
        self.checkbox_momento = tk.BooleanVar(value=True)       # Mostrar/ocultar el momento articular estimado
        
        # Estatura y peso
        self.entrada_altura = tk.StringVar(value="175")   # Valor por defecto de la estatura del sujeto (cm), editable desde la interfaz; se usa para la calibración píxel-cm
        self.entrada_peso = tk.StringVar(value="75")      # Valor por defecto del peso del sujeto (kg), editable desde la interfaz; se usa para estimar la masa de cada segmento
        
        # Contador sentadillas
        self.contador = 0                  # Número de repeticiones evaluadas como "Válida" en la sesión actual
        self.estado_posicion = "arriba"    # Estado de la máquina de estados de la repetición: "arriba" (de pie) o "bajando" (en ejecución)
        self.rep_en_progreso = False       # Indica si en este instante se está registrando una repetición (entre el inicio del descenso y el retorno a la postura erguida)

        self.reps_guardadas = []                           # Lista de diccionarios con el resumen de cada repetición completada (la que luego se exporta a Excel)
        self.datos_rep_actual = self.iniciar_datos_rep()    # Diccionario que acumula los valores extremos de la repetición en curso

        # variables previas para calcular derivadas (velocidad y acel)
        # Todas las variables terminadas en "_ant" guardan el valor del fotograma anterior, imprescindible para aproximar
        # las derivadas (velocidad = Δposición/Δt, aceleración = Δvelocidad/Δt) mediante diferencias finitas.
        self.tiempo_ant = None #Tiempo del frame anterior
        self.punto_medio_AB_ant = None     # Punto medio del segmento tronco (A-B, hombro-cadera) en el fotograma anterior
        self.punto_medio_BC_ant = None     # Punto medio del segmento muslo (B-C, cadera-rodilla) en el fotograma anterior
        self.punto_medio_CD_ant = None     # Punto medio del segmento pierna (C-D, rodilla-tobillo) en el fotograma anterior
        self.velocidad_lineal_AB = 0.0     # Velocidad lineal (ya suavizada con el filtro EMA) del centro de masa del tronco
        self.velocidad_lineal_BC = 0.0     # Velocidad lineal (ya suavizada con el filtro EMA) del centro de masa del muslo
        self.velocidad_lineal_CD = 0.0     # Velocidad lineal (ya suavizada con el filtro EMA) del centro de masa de la pierna
        self.velocidad_lineal_AB_ant = 0.0 # Velocidad lineal bruta (sin filtrar) del tronco en el fotograma anterior, usada para derivar la aceleración
        self.velocidad_lineal_BC_ant = 0.0 # Velocidad lineal bruta (sin filtrar) del muslo en el fotograma anterior
        self.velocidad_lineal_CD_ant = 0.0 # Velocidad lineal bruta (sin filtrar) de la pierna en el fotograma anterior
        self.aceleracion_AB = 0.0          # Aceleración lineal (suavizada con el filtro EMA) del centro de masa del tronco
        self.aceleracion_BC = 0.0          # Aceleración lineal (suavizada con el filtro EMA) del centro de masa del muslo
        self.aceleracion_CD = 0.0          # Aceleración lineal (suavizada con el filtro EMA) del centro de masa de la pierna
        self.angulo_tronco_ant = None      # Ángulo del tronco (θt) en el fotograma anterior
        self.angulo_cadera_ant = None      # Ángulo de la cadera (θc) en el fotograma anterior
        self.angulo_rodilla_ant = None     # Ángulo de la rodilla (θr) en el fotograma anterior
        self.angulo_pie_ant = None         # Ángulo del pie/tobillo (θp) en el fotograma anterior
        self.velocidad_angular_tronco = 0.0    # Velocidad angular (suavizada) del segmento tronco
        self.velocidad_angular_cadera = 0.0    # Velocidad angular (suavizada) de la cadera
        self.velocidad_angular_rodilla = 0.0   # Velocidad angular (suavizada) de la rodilla
        self.velocidad_angular_pie = 0.0       # Velocidad angular (suavizada) del pie/tobillo
        self.velocidad_angular_tronco_ant = 0.0  # Velocidad angular bruta del tronco en el fotograma anterior, usada para derivar la aceleración angular
        self.velocidad_angular_cadera_ant = 0.0  # Velocidad angular bruta de la cadera en el fotograma anterior
        self.velocidad_angular_rodilla_ant = 0.0 # Velocidad angular bruta de la rodilla en el fotograma anterior
        self.velocidad_angular_pie_ant = 0.0     # Velocidad angular bruta del pie/tobillo en el fotograma anterior
        self.aceleracion_angular_tronco = 0.0    # Aceleración angular (suavizada) del tronco
        self.aceleracion_angular_cadera = 0.0    # Aceleración angular (suavizada) de la cadera
        self.aceleracion_angular_rodilla = 0.0   # Aceleración angular (suavizada) de la rodilla
        self.aceleracion_angular_pie = 0.0       # Aceleración angular (suavizada) del pie/tobillo

        # Tabla de datos biomecanicos y dinamicos
        # ttk.LabelFrame: contenedor con borde y título, agrupa visualmente la tabla de resultados
        self.frame_tabla = ttk.LabelFrame(self.root, text="Vectores Biomecánicos (Perfil Izq.)")
        self.frame_tabla.place(x=20, y=20, width=820, height=450) 
        self.contenedor = ttk.Frame(self.frame_tabla)
        self.contenedor.pack(fill="both", expand=True, padx=5, pady=5)

        # ttk.Treeview: widget de tabla de Tkinter; se usa en modo "headings" (sin columna de árbol) para mostrar filas de datos tabulares
        cols = ("Puntos", "Longitud", "Angulo", "VelLin", "AcelLin", "VelAng", "AcelAng", "Fuerza", "Momento", "Evaluacion")
        self.tabla = ttk.Treeview(self.contenedor, columns=cols, show="headings")
        
        # heading(): texto que se muestra en la cabecera de cada columna
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
        # column(): ancho en píxeles y alineación del contenido de cada columna
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
        
        # colores de filas
        # tag_configure(): define un estilo de fila (color de fondo) que después se asigna a cada insert() mediante el parámetro tags=
        self.tabla.tag_configure("tronco", background="#ADD8E6")
        self.tabla.tag_configure("cadera", background="#FFFACD")
        self.tabla.tag_configure("rodilla", background="#90EE90")
        self.tabla.tag_configure("pie", background="#DDA0DD")
        self.tabla.tag_configure("separador", background="#F0F0F0")
        scroll = ttk.Scrollbar(self.contenedor, orient="vertical", command=self.tabla.yview)  # Barra de desplazamiento vertical vinculada a la tabla
        self.tabla.configure(yscrollcommand=scroll.set)
        self.tabla.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # vista principal de la camara
        self.lbl_titulo = ttk.Label(self.root, text="VISOR DE ANÁLISIS BIOMECÁNICO EN TIEMPO REAL", font=("Arial", 10, "bold"))
        self.lbl_titulo.place(x=860, y=20)
        # tk.Label reutilizado como "lienzo" de vídeo: Tkinter no tiene un widget nativo de vídeo, así que cada
        # fotograma se convierte en una imagen (PhotoImage) y se asigna a este Label mediante config(image=...)
        self.canvas_vid = tk.Label(self.root, bg="black")
        self.canvas_vid.place(x=860, y=50, width=640, height=360) 

        # Botones Control y configuracion
        self.frame_ctrl = ttk.LabelFrame(self.root, text="Controles y Configuración")
        self.frame_ctrl.place(x=20, y=490, width=600, height=140)

        # Cada botón asocia, mediante el parámetro command, una función que se ejecuta al hacer clic (callback)
        ttk.Button(self.frame_ctrl, text="Cargar Video", command=self.cargar_video).grid(row=0, column=0, padx=10, pady=15)
        ttk.Button(self.frame_ctrl, text="Detener Cámara", command=self.parar).grid(row=0, column=1, padx=10, pady=15)
        ttk.Button(self.frame_ctrl, text="Iniciar Cámara", command=self.encender_camara).grid(row=0, column=2, padx=10, pady=15)
        ttk.Button(self.frame_ctrl, text="Exportar Reporte", command=self.guardar_excel).grid(row=0, column=3, padx=10, pady=15)
        # Peso y altura del usuario
        self.box_params = ttk.Frame(self.frame_ctrl)
        self.box_params.grid(row=1, column=0, columnspan=4, pady=5, sticky="w")
        ttk.Label(self.box_params, text="Tu Altura (cm):").pack(side="left", padx=10)
        ttk.Entry(self.box_params, textvariable=self.entrada_altura, width=6).pack(side="left", padx=5)   # Campo editable vinculado a self.entrada_altura
        ttk.Label(self.box_params, text="Tu Peso (kg):").pack(side="left", padx=25)
        ttk.Entry(self.box_params, textvariable=self.entrada_peso, width=6).pack(side="left", padx=5)     # Campo editable vinculado a self.entrada_peso

        # panel de repeticiones
        self.frame_cont = ttk.LabelFrame(self.root, text="Sentadillas Correctas")
        self.frame_cont.place(x=860, y=430, width=450, height=90)
        self.lbl_cont = ttk.Label(self.frame_cont, text="0", font=("Arial", 28, "bold"), foreground="green")  # Etiqueta que muestra el contador de repeticiones válidas
        self.lbl_cont.pack(side="left", padx=30, pady=5)
        ttk.Button(self.frame_cont, text="Resetear", command=self.reset).pack(side="right", padx=30, pady=5)

        # ticks de visualizar en pantalla
        # Checkbutton: casilla de verificación vinculada a una BooleanVar; al marcarla/desmarcarla cambia el valor de la
        # variable, que luego se consulta en procesar_frame() (self.checkbox_x.get()) para decidir si se dibuja esa capa
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

    ###=============================== DECLARACION FUNCIONES MATEMATICAS ===============================###
    # Funciones puramente matemáticas/geométricas, sin efectos sobre la interfaz, reutilizadas desde procesar_frame().

    def calcular_angulo(self, punto1, punto2, punto3):
        """
        Calcula el ángulo articular (en grados) formado en el vértice
        punto2 por los segmentos punto2-punto1 y punto2-punto3.

        Se utiliza math.atan2() sobre las diferencias de coordenadas en
        lugar del producto escalar, ya que atan2 gestiona correctamente los
        cuatro cuadrantes del plano sin singularidades, evitando ambigüedades
        de signo cuando los segmentos cambian de orientación durante el
        movimiento.

        Parámetros:
            punto1, punto2, punto3: objetos con atributos .x y .y (landmarks
                de MediaPipe o instancias de la clase Punto), en coordenadas
                normalizadas o en píxeles (el resultado en grados es el
                mismo en ambos casos, al ser un ángulo relativo).

        Devuelve:
            float: ángulo en grados, en el rango [0, 180].
        """
        rad = math.atan2(punto3.y - punto2.y, punto3.x - punto2.x) - math.atan2(punto1.y - punto2.y, punto1.x - punto2.x)
        angulo = abs(rad * 180.0 / math.pi)
        if angulo > 180.0: angulo = 360.0 - angulo   # Normaliza al rango [0, 180], ya que un ángulo articular no puede superar los 180°
        return angulo

    def calcular_distancia(self, punto1, punto2, w, h):
        """
        Distancia euclídea en píxeles entre dos puntos.

        Parámetros:
            punto1, punto2 (tuple): coordenadas (x, y) en píxeles.
            w, h: no se utilizan en el cálculo en sí (la conversión de
                coordenadas normalizadas a píxeles ya se realiza antes de
                llamar a esta función); se mantienen en la firma por si se
                necesitan en futuras ampliaciones.

        Devuelve:
            float: distancia en píxeles entre punto1 y punto2.
        """
        # Al recibir tuplas en píxeles (pixelA, pixelB), calculamos la distancia euclídea directa entre sus componentes
        return math.hypot(punto2[0] - punto1[0], punto2[1] - punto1[1])

    def iniciar_datos_rep(self):
        """
        Crea (o reinicia) el diccionario acumulador de una repetición.

        Estos valores se actualizan fotograma a fotograma mientras
        rep_en_progreso es True, y se consultan al final de la repetición
        tanto para decidir si esta es válida como para componer la fila
        que se exporta al informe Excel.

        Devuelve:
            dict: con las claves:
                "prof_max": ángulo mínimo de rodilla alcanzado (profundidad),
                    inicializado a 180° (posición de pie, sin flexión).
                "incl_max": ángulo máximo de tronco alcanzado (inclinación).
                "vmax", "fmax", "mmax": valores pico de velocidad, fuerza y
                    momento registrados durante la repetición.
        """
        return {"prof_max": 180.0, "incl_max": 0.0, "vmax": 0.0, "fmax": 0.0, "mmax": 0.0}
    
###=============================== DECLARACION FUNCIONALIDADES HERRAMIENTA ===============================###
# Callbacks de los botones de la interfaz: gestionan el ciclo de vida de la captura de vídeo y la exportación de resultados.
    def limpiar_memoria(self):
        """
        Reinicia todo el estado de seguimiento de repeticiones.

        Se invoca tanto al pulsar "Resetear" como al iniciar una nueva
        captura (cámara o vídeo), de forma que cada sesión comience con el
        contador a cero y sin repeticiones almacenadas de sesiones previas.
        """
        self.reps_guardadas = []
        self.datos_rep_actual = self.iniciar_datos_rep()
        self.contador = 0
        self.estado_posicion = "arriba"
        self.rep_en_progreso = False
        self.tecnica_ok = True
        self.lbl_cont.config(text="0")

    def reset(self):
        """Callback del botón "Resetear": delega directamente en limpiar_memoria()."""
        self.limpiar_memoria()

    def encender_camara(self):
        """
        Callback del botón "Iniciar Cámara".

        Abre la cámara web por defecto del sistema (índice 0) mediante
        OpenCV, reinicia el estado de la sesión y lanza el bucle de
        procesado de fotogramas en tiempo real (procesar_frame).
        """
        if not self.run:
            self.limpiar_memoria()
            self.cap = cv2.VideoCapture(0)
            self.run = True
            self.tiempo_ant = None 
            self.procesar_frame()

    def parar(self):
        """
        Callback del botón "Detener Cámara".

        Detiene el bucle de procesado, libera el recurso de vídeo
        (cv2.VideoCapture.release) y limpia tanto el visor de vídeo como la
        tabla de resultados.
        """
        self.run = False
        self.tiempo_ant = None
        if self.cap: self.cap.release()
        self.canvas_vid.config(image="") 
        for item in self.tabla.get_children():
            self.tabla.delete(item)

    def cargar_video(self):
        """
        Callback del botón "Cargar Video".

        Abre un cuadro de diálogo para que el usuario seleccione un archivo
        de vídeo local (.mp4, .avi, .mov) y, si se selecciona uno, detiene
        cualquier captura en curso, reinicia el estado y comienza el
        procesado del archivo exactamente igual que si fuera la cámara en
        directo.
        """
        ruta = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov")])
        if ruta:
            self.parar()
            self.limpiar_memoria()
            self.cap = cv2.VideoCapture(ruta)
            self.run = True
            self.tiempo_ant = None
            self.procesar_frame()

    def guardar_excel(self):
        """
        Callback del botón "Exportar Reporte".

        Construye un DataFrame de pandas a partir de self.reps_guardadas
        (una fila por repetición completada) y lo exporta a un archivo
        .xlsx mediante openpyxl. A continuación recorre las celdas
        generadas para aplicar un código de colores: las filas de
        repeticiones "Válida" se resaltan en verde, y en las filas no
        válidas se resalta en rojo la celda concreta del criterio
        incumplido (profundidad, inclinación de tronco, o ambas).
        """
        if not self.reps_guardadas:
            messagebox.showwarning("Error", "Haz alguna sentadilla primero antes de exportar.")
            return
        name = f"Reporte_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"   # Nombre de archivo sugerido, con marca de fecha/hora
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")], initialfile=name)
        if path:
            try:
                df = pd.DataFrame(self.reps_guardadas)
                with pd.ExcelWriter(path, engine='openpyxl') as w:
                    df.to_excel(w, sheet_name='Resultados', index=False)
                    hoja = w.sheets['Resultados']
                    fondo_verde = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid")  # Color de fondo para repeticiones válidas
                    fondo_rojo = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")   # Color de fondo para resaltar el criterio incumplido
                    for r in range(2, len(df) + 2):   # Empieza en la fila 2 porque la fila 1 contiene las cabeceras
                        txt_res = str(hoja.cell(row=r, column=2).value).lower()   # Columna 2 = "Resultado" de la repetición
                        
                        if "válida" in txt_res:
                            for c in range(1, 8):
                                hoja.cell(row=r, column=c).fill = fondo_verde   # Pinta toda la fila en verde
                        else:
                            if "profundidad" in txt_res or "múltiples" in txt_res:
                                hoja.cell(row=r, column=3).fill = fondo_rojo   # Columna 3 = "Profundidad Mín (°)"
                            if "pecho" in txt_res or "múltiples" in txt_res:
                                hoja.cell(row=r, column=4).fill = fondo_rojo   # Columna 4 = "Inclinación Máx (°)"
                messagebox.showinfo("Ok!", f"Se guardó todo en:\n{path}")
            except Exception as e:
                messagebox.showerror("Fallo", f"Ocurrió un error al guardar:\n{e}")

###=============================== DECLARACION PROCESADO IMAGEN (CAMARA) ===============================###
# Núcleo de la aplicación: el bucle de análisis fotograma a fotograma.

    def procesar_frame(self):
        """
        Bucle principal de procesado, ejecutado una vez por cada fotograma.

        Encadena todas las etapas del sistema descritas en la memoria:
            1. Captura del fotograma y conversión de color BGR->RGB.
            2. Estimación de pose con MediaPipe y extracción de los 4
               landmarks del perfil izquierdo (hombro, cadera, rodilla,
               tobillo).
            3. Cálculo de ángulos articulares y longitudes de segmento.
            4. Calibración píxel-cm a partir de la estatura del usuario.
            5. Cálculo cinemático (velocidades/aceleraciones lineales y
               angulares) mediante diferencias finitas, suavizado con un
               filtro de Media Móvil Exponencial (EMA).
            6. Cálculo dinámico (fuerzas y momentos) mediante F=m·a y
               M=I·α.
            7. Dibujo de las distintas capas visuales sobre el fotograma,
               según las casillas de visualización activas.
            8. Máquina de estados para detectar el inicio/fin de cada
               repetición y evaluar su técnica.
            9. Actualización de la tabla de resultados y del visor de
               vídeo.
            10. Reprogramación de sí misma mediante root.after(), creando
                un bucle no bloqueante.

        No recibe parámetros ni devuelve nada: actúa modificando
        directamente los widgets de la interfaz y el estado interno de la
        instancia (self).
        """
        if self.run and self.cap is not None:
            ret, frame = self.cap.read()   # Lee el siguiente fotograma; ret=False si no quedan más fotogramas (fin de vídeo) o la cámara se desconecta
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV captura en BGR; MediaPipe espera RGB, de ahí la conversión
                frame_draw = rgb.copy()                         # Copia sobre la que se dibujan todas las capas (triángulos, esqueleto, texto), dejando 'rgb' intacto
                h, w, _ = rgb.shape                              # Alto y ancho del fotograma en píxeles, usados para desnormalizar las coordenadas de MediaPipe
                t_ahora = time.time()                            # Marca de tiempo de este fotograma, usada para calcular dt en las derivadas cinemáticas
                if self.pose is not None:   # Solo se procesa si MediaPipe se cargó correctamente en el constructor
                    res = self.pose.process(rgb)   # Ejecuta la inferencia del modelo de pose sobre el fotograma actual
                    if res.pose_landmarks:   # Continúa solo si se ha detectado una persona en el fotograma
                        puntos_mediapipe = res.pose_landmarks.landmark
                        # Coger los 4 puntos basicos del lado izq (perfil izquierdo; suficiente al asumir simetría bilateral del movimiento)
                        puntoHombro = puntos_mediapipe[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        puntoCadera = puntos_mediapipe[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                        puntoRodilla = puntos_mediapipe[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
                        puntoPie = puntos_mediapipe[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

                        # pasar a pixeles (las coordenadas de MediaPipe vienen normalizadas en [0,1]; se desnormalizan multiplicando por el ancho/alto del frame)
                        pixelA = (int(puntoHombro.x * w), int(puntoHombro.y * h))
                        pixelB = (int(puntoCadera.x * w), int(puntoCadera.y * h))
                        pixelC = (int(puntoRodilla.x * w), int(puntoRodilla.y * h))
                        pixelD = (int(puntoPie.x * w), int(puntoPie.y * h))
                        #Puntos auxiliares para dibujar los triangulos (catetos vertical/horizontal de los triángulos auxiliares de tronco y pie)
                        punto_aux_vert = (pixelB[0], pixelA[1]) 
                        punto_aux_horiz = (pixelC[0], pixelD[1]) 

                        # punto medio de las articulaciones para poner datos dinamicos (también se usa para el seguimiento de velocidad de cada segmento)
                        punto_medio_AB = ((pixelA[0] + pixelB[0]) / 2.0, (pixelA[1] + pixelB[1]) / 2.0)
                        punto_medio_BC = ((pixelB[0] + pixelC[0]) / 2.0, (pixelB[1] + pixelC[1]) / 2.0)
                        punto_medio_CD = ((pixelC[0] + pixelD[0]) / 2.0, (pixelC[1] + pixelD[1]) / 2.0)
                        
                        # Calcular ángulos y distancias
                        angulo_cadera = self.calcular_angulo(puntoHombro, puntoCadera, puntoRodilla)      # Ángulo θc en el vértice cadera (hombro-cadera-rodilla)
                        angulo_rodilla = self.calcular_angulo(puntoCadera, puntoRodilla, puntoPie)        # Ángulo θr en el vértice rodilla (cadera-rodilla-tobillo); variable discriminadora de la máquina de estados
                        angulo_tronco = self.calcular_angulo(puntoHombro, puntoCadera, Punto(puntoCadera.x, puntoHombro.y))   # Ángulo del tronco respecto a la vertical, con un punto auxiliar situado sobre la cadera a la altura del hombro
                        angulo_pie = self.calcular_angulo(puntoRodilla, puntoPie, Punto(puntoRodilla.x, puntoPie.y))          # Ángulo de la pierna respecto a la vertical, con el mismo recurso del punto auxiliar
                        distancia_ab = self.calcular_distancia(pixelA, pixelB, w, h)   # Longitud en píxeles del segmento tronco (hombro-cadera)
                        distancia_bc = self.calcular_distancia(pixelB, pixelC, w, h)   # Longitud en píxeles del segmento muslo (cadera-rodilla)
                        distancia_cd = self.calcular_distancia(pixelC, pixelD, w, h)   # Longitud en píxeles del segmento pierna (rodilla-tobillo)

                        # Comprobar datos peso y altura del usuario (con valor de respaldo si el campo no contiene un número válido)
                        try: 
                            altura = float(self.entrada_altura.get())
                        except ValueError: altura = 175.0
                        
                        try: 
                            peso = float(self.entrada_peso.get())
                        except ValueError: peso = 75.0
                        # Conversion pixeles a cm (segun la altura introducida): se asume que tronco+muslo+pierna equivale al 77,9% de la estatura total
                        suma_pixeles = distancia_ab + distancia_bc + distancia_cd
                        if suma_pixeles > 0:
                            relacion_cm_px = (altura * 0.779) / suma_pixeles
                        else:
                            relacion_cm_px = 0
                        # longitudes aprox de los segmentos en cm
                        longitud_ab = distancia_ab * relacion_cm_px
                        longitud_bc = distancia_bc * relacion_cm_px
                        longitud_cd = distancia_cd * relacion_cm_px
                        # peso aprox de los segmentos en kg (segun tablas antropometricas: 50% tronco, 10% muslo, 4,6% pierna)
                        masa_AB = peso * 0.50  
                        masa_BC = peso * 0.10  
                        masa_CD = peso * 0.046 
                        # Calculo de momentos de inercia aproximados (suponiendo segmentos como cilindros) para luego calcular momentos de fuerza (I = m * r^2, con r aprox el 30% de la longitud del segmento)
                        I_AB = masa_AB * (0.3 * (longitud_ab/100.0))**2
                        I_BC = masa_BC * (0.3 * (longitud_bc/100.0))**2
                        I_CD = masa_CD * (0.3 * (longitud_cd/100.0))**2
                        #Inicializar variables de fuerza y momento a 0; solo se recalculan más abajo si ya existe un fotograma anterior con el que derivar
                        Fuerza_ab = 0.0
                        Fuerza_bc = 0.0
                        Fuerza_cd = 0.0
                        Momento_ab = 0.0
                        Momento_bc = 0.0
                        Momento_cd = 0.0

                        # calculos cinematica (solo es posible a partir del segundo fotograma, cuando ya existen valores "_ant" del fotograma previo)
                        if self.tiempo_ant is not None and self.punto_medio_AB_ant is not None:
                            #dt = delta de tiempo
                            dt = t_ahora - self.tiempo_ant
                            if dt > 0:   # Evita división por cero si dos fotogramas llegasen con la misma marca de tiempo
                                #distancia en cm recorrida por el punto medio de cada segmento entre el fotograma anterior y el actual
                                # (estas variables reutilizan el nombre de las distancias en píxeles calculadas más arriba, pero ahora representan un desplazamiento en metros)
                                distancia_ab = (math.hypot(punto_medio_AB[0] - self.punto_medio_AB_ant[0], punto_medio_AB[1] - self.punto_medio_AB_ant[1]) * relacion_cm_px) / 100.0
                                distancia_bc = (math.hypot(punto_medio_BC[0] - self.punto_medio_BC_ant[0], punto_medio_BC[1] - self.punto_medio_BC_ant[1]) * relacion_cm_px) / 100.0
                                distancia_cd = (math.hypot(punto_medio_CD[0] - self.punto_medio_CD_ant[0], punto_medio_CD[1] - self.punto_medio_CD_ant[1]) * relacion_cm_px) / 100.0
                                #velocidad = distancia / dt
                                velocidad_tronco = distancia_ab / dt
                                velocidad_cadera = distancia_bc / dt
                                velocidad_rodilla = distancia_cd / dt
                                #aceleracion = velocidad / dt (aproximada a partir de la velocidad bruta, sin filtrar, del fotograma anterior)
                                aceleracion_tronco = (velocidad_tronco - self.velocidad_lineal_AB_ant) / dt
                                aceleracion_cadera = (velocidad_cadera - self.velocidad_lineal_BC_ant) / dt
                                aceleracion_rodilla = (velocidad_rodilla - self.velocidad_lineal_CD_ant) / dt
                                #Setear valores en interfaz aplicando el filtro EMA (Media Móvil Exponencial), que atenúa el ruido de la estimación de pose
                                alf_v = 0.2; alf_a = 0.15   # Coeficientes de suavizado: 20% de peso a la medida nueva en velocidades, 15% en aceleraciones (más agresivo, al ser una señal más ruidosa)
                                self.velocidad_lineal_AB = (1 - alf_v) * self.velocidad_lineal_AB + alf_v * velocidad_tronco
                                self.velocidad_lineal_BC = (1 - alf_v) * self.velocidad_lineal_BC + alf_v * velocidad_cadera
                                self.velocidad_lineal_CD = (1 - alf_v) * self.velocidad_lineal_CD + alf_v * velocidad_rodilla
                                self.aceleracion_AB = (1 - alf_a) * self.aceleracion_AB + alf_a * aceleracion_tronco
                                self.aceleracion_BC = (1 - alf_a) * self.aceleracion_BC + alf_a * aceleracion_cadera
                                self.aceleracion_CD = (1 - alf_a) * self.aceleracion_CD + alf_a * aceleracion_rodilla
                                self.velocidad_lineal_AB_ant = velocidad_tronco   # Guarda la velocidad bruta de este fotograma, necesaria para la aceleración bruta del siguiente
                                self.velocidad_lineal_BC_ant = velocidad_cadera
                                self.velocidad_lineal_CD_ant = velocidad_rodilla
                                
                                #Velocidad angular = delta angulo / dt
                                velocidad_angular_tronco = (math.radians(angulo_tronco) - math.radians(self.angulo_tronco_ant)) / dt
                                velocidad_angular_cadera = (math.radians(angulo_cadera) - math.radians(self.angulo_cadera_ant)) / dt
                                velocidad_angular_rodilla = (math.radians(angulo_rodilla) - math.radians(self.angulo_rodilla_ant)) / dt
                                velocidad_angular_pie = (math.radians(angulo_pie) - math.radians(self.angulo_pie_ant)) / dt
                                #Aceleracion angular = delta velocidad angular / dt
                                aceleracion_angular_tronco = (velocidad_angular_tronco - self.velocidad_angular_tronco_ant) / dt
                                aceleracion_angular_cadera = (velocidad_angular_cadera - self.velocidad_angular_cadera_ant) / dt
                                aceleracion_angular_rodilla = (velocidad_angular_rodilla - self.velocidad_angular_rodilla_ant) / dt
                                aceleracion_angular_pie = (velocidad_angular_pie - self.velocidad_angular_pie_ant) / dt
                                # calculos dinamica
                                #Fuerza = masa * aceleracion (2ª ley de Newton), con la aceleración lineal ya filtrada del centro de masa de cada segmento
                                Fuerza_ab = masa_AB * abs(self.aceleracion_AB)
                                Fuerza_bc = masa_BC * abs(self.aceleracion_BC)
                                Fuerza_cd = masa_CD * abs(self.aceleracion_CD)
                                #Momento = I * aceleracion angular (2ª ley de Euler para la rotación), con I el momento de inercia aproximado calculado más arriba
                                Momento_ab = I_AB * abs(self.aceleracion_angular_tronco)
                                Momento_bc = I_BC * abs(self.aceleracion_angular_cadera)
                                Momento_cd = I_CD * abs(self.aceleracion_angular_rodilla)

                                #Setear valores en interfaz con suavizado - Para que valores se actualicen de forma mas fluida y no tan brusca
                                self.velocidad_angular_tronco = (1 - alf_v) * self.velocidad_angular_tronco + alf_v * velocidad_angular_tronco
                                self.velocidad_angular_cadera = (1 - alf_v) * self.velocidad_angular_cadera + alf_v * velocidad_angular_cadera
                                self.velocidad_angular_rodilla = (1 - alf_v) * self.velocidad_angular_rodilla + alf_v * velocidad_angular_rodilla
                                self.velocidad_angular_pie = (1 - alf_v) * self.velocidad_angular_pie + alf_v * velocidad_angular_pie
                                self.aceleracion_angular_tronco = (1 - alf_a) * self.aceleracion_angular_tronco + alf_a * aceleracion_angular_tronco
                                self.aceleracion_angular_cadera = (1 - alf_a) * self.aceleracion_angular_cadera + alf_a * aceleracion_angular_cadera
                                self.aceleracion_angular_rodilla = (1 - alf_a) * self.aceleracion_angular_rodilla + alf_a * aceleracion_angular_rodilla
                                self.aceleracion_angular_pie = (1 - alf_a) * self.aceleracion_angular_pie + alf_a * aceleracion_angular_pie

                                self.velocidad_angular_tronco_ant = velocidad_angular_tronco   # Guarda el valor bruto de este fotograma para calcular la aceleración angular en el siguiente
                                self.velocidad_angular_cadera_ant = velocidad_angular_cadera
                                self.velocidad_angular_rodilla_ant = velocidad_angular_rodilla
                                self.velocidad_angular_pie_ant = velocidad_angular_pie

                        # Guardar valores previos para el siguiente frame (se usarán como referencia "_ant" en la siguiente llamada a procesar_frame)
                        self.tiempo_ant = t_ahora
                        self.punto_medio_AB_ant = punto_medio_AB
                        self.punto_medio_BC_ant = punto_medio_BC
                        self.punto_medio_CD_ant = punto_medio_CD
                        self.angulo_tronco_ant = angulo_tronco
                        self.angulo_cadera_ant = angulo_cadera
                        self.angulo_rodilla_ant = angulo_rodilla
                        self.angulo_pie_ant = angulo_pie
                        #================== DIBUJOS EN IMAGEN (MEDIAPIPE) ==================#
                        # A partir de aquí se dibuja sobre frame_draw (no sobre 'rgb') toda la información visual, capa a capa, según las casillas marcadas por el usuario
                        if self.checkbox_triangulos.get():
                            # Dibuja los triángulos en una capa aparte y la mezcla con transparencia (addWeighted) sobre la imagen, sin ocultar el vídeo original
                            capa = frame_draw.copy()
                            cv2.fillPoly(capa, [np.array([pixelA, pixelB, pixelC])], (255, 250, 205))    # Triángulo hombro-cadera-rodilla (amarillo pálido)
                            cv2.fillPoly(capa, [np.array([pixelB, pixelC, pixelD])], (144, 238, 144))   # Triángulo cadera-rodilla-tobillo (verde claro)
                            cv2.fillPoly(capa, [np.array([pixelA, pixelB, punto_aux_vert])], (173, 216, 230))  # Triángulo auxiliar: inclinación del tronco respecto a la vertical (azul claro)
                            cv2.fillPoly(capa, [np.array([pixelC, pixelD, punto_aux_horiz])], (221, 160, 221)) # Triángulo auxiliar: inclinación de la pierna respecto a la vertical (violeta)
                            cv2.addWeighted(capa, 0.4, frame_draw, 0.6, 0, frame_draw)   # Mezcla la capa de triángulos (40%) con la imagen original (60%) para lograr el efecto translúcido
                        self.mp_drawing.draw_landmarks(frame_draw, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)   # Dibuja el esqueleto completo detectado por MediaPipe (33 puntos y sus conexiones)
                        fnt = cv2.FONT_HERSHEY_SIMPLEX   # Tipografía usada para todas las etiquetas de texto superpuestas
                        
                        if self.checkbox_grados.get():
                            # Etiquetas con el valor en grados de cada ángulo articular, junto a su vértice
                            cv2.putText(frame_draw, f"{int(angulo_tronco)}o", (pixelA[0]-40, pixelA[1]), fnt, 0.6, (0,0,255), 2)
                            cv2.putText(frame_draw, f"{int(angulo_cadera)}o", (pixelB[0]-40, pixelB[1]+20), fnt, 0.6, (0,0,255), 2)
                            cv2.putText(frame_draw, f"{int(angulo_rodilla)}o", (pixelC[0]+15, pixelC[1]), fnt, 0.6, (0,0,255), 2)
                            cv2.putText(frame_draw, f"{int(angulo_pie)}o", (pixelD[0]+15, pixelD[1]), fnt, 0.6, (0,0,255), 2)

                        if self.checkbox_dist.get():
                            # Etiquetas con la longitud en cm de cada segmento, centradas en su punto medio
                            cv2.putText(frame_draw, f"{int(longitud_ab)}cm", (int(punto_medio_AB[0])-15, int(punto_medio_AB[1])), fnt, 0.5, (75, 0, 130), 2)
                            cv2.putText(frame_draw, f"{int(longitud_bc)}cm", (int(punto_medio_BC[0])-15, int(punto_medio_BC[1])), fnt, 0.5, (75, 0, 130), 2)
                            cv2.putText(frame_draw, f"{int(longitud_cd)}cm", (int(punto_medio_CD[0])-15, int(punto_medio_CD[1])), fnt, 0.5, (75, 0, 130), 2)

                        if self.checkbox_veloc_lineal.get():
                            # Etiquetas con la velocidad lineal (m/s) del centro de masa de cada segmento
                            cv2.putText(frame_draw, f"{self.velocidad_lineal_AB:.1f}m/s", (int(punto_medio_AB[0])+25, int(punto_medio_AB[1])+15), fnt, 0.5, (0, 100, 255), 2)
                            cv2.putText(frame_draw, f"{self.velocidad_lineal_BC:.1f}m/s", (int(punto_medio_BC[0])+25, int(punto_medio_BC[1])+15), fnt, 0.5, (0, 100, 255), 2)
                            cv2.putText(frame_draw, f"{self.velocidad_lineal_CD:.1f}m/s", (int(punto_medio_CD[0])+25, int(punto_medio_CD[1])+15), fnt, 0.5, (0, 100, 255), 2)

                        if self.checkbox_acel_lineal.get():
                            # Etiquetas con la aceleración lineal (m/s²) del centro de masa de cada segmento
                            cv2.putText(frame_draw, f"{self.aceleracion_AB:.1f}m/s2", (int(punto_medio_AB[0])+25, int(punto_medio_AB[1])-15), fnt, 0.5, (255, 255, 0), 2)
                            cv2.putText(frame_draw, f"{self.aceleracion_BC:.1f}m/s2", (int(punto_medio_BC[0])+25, int(punto_medio_BC[1])-15), fnt, 0.5, (255, 255, 0), 2)
                            cv2.putText(frame_draw, f"{self.aceleracion_CD:.1f}m/s2", (int(punto_medio_CD[0])+25, int(punto_medio_CD[1])-15), fnt, 0.5, (255, 255, 0), 2)

                        if self.checkbox_veloc_ang.get():
                            # Etiquetas con la velocidad angular (rad/s) de cada articulación
                            cv2.putText(frame_draw, f"w:{self.velocidad_angular_tronco:.1f}", (pixelA[0]-95, pixelA[1]-15), fnt, 0.5, (42, 42, 165), 2)
                            cv2.putText(frame_draw, f"w:{self.velocidad_angular_cadera:.1f}", (pixelB[0]-95, pixelB[1]-15), fnt, 0.5, (42, 42, 165), 2)
                            cv2.putText(frame_draw, f"w:{self.velocidad_angular_rodilla:.1f}", (pixelC[0]+15, pixelC[1]-30), fnt, 0.5, (42, 42, 165), 2)
                            cv2.putText(frame_draw, f"w:{self.velocidad_angular_pie:.1f}", (pixelD[0]+15, pixelD[1]-30), fnt, 0.5, (42, 42, 165), 2)

                        if self.checkbox_acel_ang.get():
                            # Etiquetas con la aceleración angular (rad/s²) de cada articulación
                            cv2.putText(frame_draw, f"a:{self.aceleracion_angular_tronco:.1f}", (pixelA[0]-95, pixelA[1]+15), fnt, 0.5, (147, 20, 255), 2)
                            cv2.putText(frame_draw, f"a:{self.aceleracion_angular_cadera:.1f}", (pixelB[0]-95, pixelB[1]+15), fnt, 0.5, (147, 20, 255), 2)
                            cv2.putText(frame_draw, f"a:{self.aceleracion_angular_rodilla:.1f}", (pixelC[0]+15, pixelC[1]-15), fnt, 0.5, (147, 20, 255), 2)
                            cv2.putText(frame_draw, f"a:{self.aceleracion_angular_pie:.1f}", (pixelD[0]+15, pixelD[1]-15), fnt, 0.5, (147, 20, 255), 2)

                        if self.checkbox_fuerza.get():
                            # Etiquetas con la fuerza neta estimada (N) sobre cada segmento
                            cv2.putText(frame_draw, f"{Fuerza_ab:.1f}N", (int(punto_medio_AB[0])-40, int(punto_medio_AB[1])-25), fnt, 0.5, (0, 0, 255), 2)
                            cv2.putText(frame_draw, f"{Fuerza_bc:.1f}N", (int(punto_medio_BC[0])-40, int(punto_medio_BC[1])-25), fnt, 0.5, (0, 0, 255), 2)
                            cv2.putText(frame_draw, f"{Fuerza_cd:.1f}N", (int(punto_medio_CD[0])-40, int(punto_medio_CD[1])-25), fnt, 0.5, (0, 0, 255), 2)

                        if self.checkbox_momento.get():
                            # Etiquetas con el momento articular estimado (Nm) en cada articulación
                            cv2.putText(frame_draw, f"{Momento_ab:.1f}Nm", (pixelA[0]+10, pixelA[1]-20), fnt, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame_draw, f"{Momento_bc:.1f}Nm", (pixelB[0]+15, pixelB[1]-20), fnt, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame_draw, f"{Momento_cd:.1f}Nm", (pixelC[0]+40, pixelC[1]+20), fnt, 0.5, (0, 255, 0), 2)
                        
                        ####================= CONDICIONES DE EVALUACION DE SENTADILLA TODO==================####
                        # Máquina de estados de la repetición: se basa principalmente en el ángulo de rodilla (θr) para detectar
                        # el inicio del descenso y el retorno a la postura erguida, y en el ángulo de cadera (θc) como criterio
                        # adicional para confirmar que se ha vuelto realmente a la posición de pie.
                        
                        if angulo_rodilla <= 155 and self.estado_posicion == "arriba":
                            # Umbral de inicio del descenso: 155° ya supone una flexión perceptible frente a los ~180° de la bipedestación,
                            # y descarta el ruido de pequeñas oscilaciones posturales estando de pie
                            self.rep_en_progreso = True
                            self.estado_posicion = "bajando"
                            self.datos_rep_actual = self.iniciar_datos_rep()

                        if self.rep_en_progreso:
                            # Mientras la repetición está en curso, se van actualizando los valores extremos que definirán su evaluación final
                            self.datos_rep_actual["prof_max"] = min(self.datos_rep_actual["prof_max"], angulo_rodilla)
                            self.datos_rep_actual["incl_max"] = max(self.datos_rep_actual["incl_max"], angulo_tronco)
                            self.datos_rep_actual["vmax"] = max(self.datos_rep_actual["vmax"], self.velocidad_lineal_AB, self.velocidad_lineal_BC, self.velocidad_lineal_CD)
                            self.datos_rep_actual["fmax"] = max(self.datos_rep_actual["fmax"], Fuerza_ab, Fuerza_bc, Fuerza_cd)
                            self.datos_rep_actual["mmax"] = max(self.datos_rep_actual["mmax"], Momento_ab, Momento_bc, Momento_cd)

                        estado = "-"
                        if angulo_rodilla > 155 and angulo_cadera > 145:
                            # Doble criterio de retorno a la posición erguida (rodilla Y cadera), para evitar conteos falsos en fases intermedias
                            estado = "De pie"
                            if self.rep_en_progreso:   # Solo se evalúa y registra la repetición si realmente se había iniciado un descenso
                                p_min = self.datos_rep_actual["prof_max"]
                                i_max = self.datos_rep_actual["incl_max"]

                                pech_ok = i_max <= 45    # Criterio de inclinación de tronco: por debajo de 45° de cadera se considera "pecho colapsado"
                                prof_ok = p_min <= 100   # Criterio de profundidad: la rodilla debe haber bajado, como mínimo, a la sentadilla paralela (≤100°)
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
                                
                                # Añade la fila de la repetición a la lista que después se exporta a Excel
                                self.reps_guardadas.append({
                                    "Repetición": len(self.reps_guardadas) + 1,
                                    "Resultado": res_final,
                                    "Profundidad Mín (°)": round(p_min, 1),
                                    "Inclinación Máx (°)": round(i_max, 1),
                                    "Velocidad Máx (m/s)": round(self.datos_rep_actual["vmax"], 2),
                                    "Fuerza Pico (N)": round(self.datos_rep_actual["fmax"], 1),
                                    "Momento Pico (Nm)": round(self.datos_rep_actual["mmax"], 1)
                                })

                            self.estado_posicion = "arriba"
                            self.rep_en_progreso = False

                        elif 100 < angulo_rodilla <= 155:
                            # Fase de transición (bajando o subiendo), sin haber alcanzado aún la profundidad de sentadilla paralela
                            if self.datos_rep_actual["incl_max"] > 45:
                                estado = "Pecho colapsado"
                            else:
                                estado = "En movimiento"
                        elif angulo_rodilla <= 100:
                            # Se ha alcanzado o superado la profundidad de sentadilla paralela
                            if self.datos_rep_actual["incl_max"] > 45:
                                estado = "Pecho colapsado"
                            elif angulo_rodilla <= 75:
                                estado = "Profunda"
                            else:
                                estado = "Paralela"

                        # rellenar tabla izq (se vacía y se reconstruye entera cada fotograma; más sencillo que actualizar fila a fila)
                        for val in self.tabla.get_children(): self.tabla.delete(val)
                        
                        # Filas de los tres segmentos: longitud, velocidad y aceleración lineal, y fuerza
                        self.tabla.insert("", "end", values=("Tronco (AB)", int(longitud_ab), "-", f"{self.velocidad_lineal_AB:.1f}", f"{self.aceleracion_AB:.1f}", "-", "-", f"{Fuerza_ab:.1f}", "-", "-"))
                        self.tabla.insert("", "end", values=("Muslo (BC)", int(longitud_bc), "-", f"{self.velocidad_lineal_BC:.1f}", f"{self.aceleracion_BC:.1f}", "-", "-", f"{Fuerza_bc:.1f}", "-", "-"))
                        self.tabla.insert("", "end", values=("Pierna (CD)", int(longitud_cd), "-", f"{self.velocidad_lineal_CD:.1f}", f"{self.aceleracion_CD:.1f}", "-", "-", f"{Fuerza_cd:.1f}", "-", "-"))
                        self.tabla.insert("", "end", values=("", "", "", "", "", "", "", "", "", ""), tags=("separador",))   # Fila vacía a modo de separador visual entre segmentos y ángulos
                        
                        # Filas de los cuatro ángulos articulares: valor en grados, velocidad/aceleración angular y momento;
                        # la columna "Evaluación" solo se rellena en cadera/rodilla, mostrando "Pecho colapsado" cuando corresponde
                        self.tabla.insert("", "end", values=("Angulo Tronco (θt)", "-", f"{int(angulo_tronco)}°", "-", "-", f"{self.velocidad_angular_tronco:.1f}", f"{self.aceleracion_angular_tronco:.1f}", "-", f"{Momento_ab:.1f}", "-"), tags=("tronco",))
                        self.tabla.insert("", "end", values=("Angulo Cadera (θc)", "-", f"{int(angulo_cadera)}°", "-", "-", f"{self.velocidad_angular_cadera:.1f}", f"{self.aceleracion_angular_cadera:.1f}", "-", f"{Momento_bc:.1f}", estado if "Pecho" in estado else "-"), tags=("cadera",))
                        self.tabla.insert("", "end", values=("Angulo Rodilla (θr)", "-", f"{int(angulo_rodilla)}°", "-", "-", f"{self.velocidad_angular_rodilla:.1f}", f"{self.aceleracion_angular_rodilla:.1f}", "-", f"{Momento_cd:.1f}", estado if "Pecho" not in estado else "-"), tags=("rodilla",))
                        self.tabla.insert("", "end", values=("Angulo Pie (θp)", "-", f"{int(angulo_pie)}°", "-", "-", f"{self.velocidad_angular_pie:.1f}", f"{self.aceleracion_angular_pie:.1f}", "-", "-", "-"), tags=("pie",))

                v_final = cv2.resize(frame_draw, (640, 360))                          # Redimensiona el fotograma dibujado al tamaño del visor de la interfaz
                img_ok = ImageTk.PhotoImage(image=Image.fromarray(v_final))           # Convierte el array de NumPy/OpenCV en una imagen compatible con Tkinter
                self.canvas_vid.config(image=img_ok); self.canvas_vid.img_tk = img_ok  # Actualiza el Label-visor; se guarda también una referencia (img_tk) para que el recolector de basura de Python no elimine la imagen antes de mostrarla

                self.root.after(15, self.procesar_frame)   # Reprograma esta misma función dentro de 15 ms: bucle de vídeo no bloqueante (no se usa un "while" para no congelar la interfaz de Tkinter)
            else:
                self.parar()   # No se ha podido leer un fotograma (fin del vídeo o cámara desconectada): se detiene la captura automáticamente

if __name__ == "__main__":
    #Inicializa
    # Punto de entrada del script: crea la ventana raíz de Tkinter, instancia la aplicación
    # y arranca el bucle principal de eventos (mainloop), que mantiene la interfaz gráfica 
    # activa y receptiva a los eventos del usuario (clics, teclas...) hasta que se cierra la ventana.
    app_raiz = tk.Tk()
    mi_app = AppBiomecanica(app_raiz)
    app_raiz.mainloop()
