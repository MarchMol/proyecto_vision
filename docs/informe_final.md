# Proyecto de Visión por Computadora: Detección y Colocación Automática de Logos en Productos

**Universidad del Valle de Guatemala**  
**Curso:** Visión por Computadora  
**Integrantes:** Francis Aguilar · César López · Jose Marchena  
**Fecha:** Mayo 2026

---

## Resumen

Este proyecto desarrolla un sistema de visión por computadora capaz de predecir automáticamente la zona óptima de colocación de un logo en imágenes de productos (camisetas y botellones/pachones). Se entrena un modelo YOLOv8n bajo el paradigma de Transfer Learning utilizando datos extraídos de las imágenes de catálogo del proveedor mayorista Makito.es, que incluyen marcas verdes indicando la zona de impresión. El sistema incluye tres módulos: (1) un pipeline de extracción de datos mediante morfología clásica, (2) un módulo de anotación interactivo tipo CAPTCHA para construir el conjunto de prueba con acuerdo inter-anotador medido por IoU, y (3) una aplicación de demostración que superpone un logo personalizado sobre la zona detectada.

---

## 1. Introducción

La personalización masiva de productos constituye un mercado en crecimiento dentro del comercio electrónico. Los proveedores de impresión necesitan colocar logotipos de clientes en posiciones estéticamente correctas sobre diferentes tipos de productos. Actualmente, este proceso requiere intervención manual por parte de diseñadores. Automatizarlo con visión por computadora reduciría costos y tiempos de producción.

Makito.es, empresa distribuidora de productos publicitarios, publica en su catálogo imágenes de sus artículos con una región en verde que delimita la zona habilitada para impresión. Esta información visual puede ser aprovechada como señal de supervisión para entrenar un detector de bounding boxes.

El objetivo central de este proyecto es entrenar un modelo que, dada una imagen de producto sin marcas, prediga las coordenadas de la zona de colocación del logo.

---

## 2. Objetivos

**Objetivo general:**  
Desarrollar un sistema de visión por computadora que prediga automáticamente la zona de colocación de un logo en camisetas y botellones, y que permita superponer un logo personalizado sobre la predicción.

**Objetivos específicos:**
- Construir un pipeline de extracción automática de bounding boxes a partir de imágenes etiquetadas por color.
- Entrenar un modelo YOLOv8n con Transfer Learning sobre el dataset extraído.
- Implementar un módulo de anotación interactivo para construir un conjunto de prueba con imágenes descargadas de internet.
- Calcular el acuerdo inter-anotador mediante IoU y exportar el ground truth del conjunto de prueba.
- Implementar un módulo de composición que superponga un logo PNG sobre la zona predicha.

---

## 3. Marco Teórico

### 3.1 Detección de objetos y bounding boxes

La detección de objetos es una tarea de visión por computadora que consiste en localizar y clasificar instancias de categorías predefinidas dentro de una imagen. La salida estándar es un conjunto de *bounding boxes* (cajas delimitadoras), cada una con coordenadas `(x_centro, y_centro, ancho, alto)` normalizadas al tamaño de la imagen, junto con una clase y una puntuación de confianza.

### 3.2 YOLO (You Only Look Once)

YOLO es una familia de arquitecturas de detección de objetos en tiempo real que formulan la detección como un problema de regresión directa sobre una cuadrícula. YOLOv8, desarrollado por Ultralytics, introduce mejoras en la cabeza de detección (anchor-free), el cuello FPN/PAN mejorado y un conjunto de hiperparámetros de aumentación más robusto. La variante *nano* (YOLOv8n) reduce el número de parámetros para funcionar eficientemente en CPU, lo cual es relevante dado el hardware disponible.

### 3.3 Transfer Learning

El Transfer Learning consiste en reutilizar los pesos de un modelo preentrenado en una tarea de origen (en este caso, detección general de objetos en COCO con 80 clases) como punto de partida para una tarea destino más específica. Esto es especialmente valioso con datasets pequeños (~150 imágenes por clase), ya que las capas iniciales ya aprendieron a extraer características visuales genéricas (bordes, texturas, formas), y solo las capas más profundas y la cabeza de detección necesitan ajustarse.

### 3.4 Intersección sobre Unión (IoU)

El IoU es la métrica estándar para evaluar la precisión de una predicción de bounding box:

```
IoU = Área(Predicción ∩ Ground Truth) / Área(Predicción ∪ Ground Truth)
```

Un IoU de 0.5 es el umbral mínimo común para considerar una detección como correcta (TP). En este proyecto, el IoU se utiliza además como métrica de **acuerdo inter-anotador** entre los miembros del equipo al construir el conjunto de prueba.

### 3.5 Composición de imágenes con canal alpha

La superposición de un logo PNG (con canal de transparencia) sobre una imagen de fondo se realiza mediante composición alfa:

```
C_out = alpha * C_logo + (1 - alpha) * C_fondo
```

donde `alpha` es el valor del canal de transparencia del logo normalizado a `[0, 1]`.

---

## 4. Datos

### 4.1 Fuente

Las imágenes de entrenamiento y validación se obtuvieron del catálogo en línea de **Makito.es**. El proveedor publica imágenes limpias de sus productos junto con versiones que muestran una región rectangular en verde indicando la zona habilitada para impresión de logos.

### 4.2 Estadísticas del dataset

| Categoría  | Imágenes únicas | Pares (original + etiqueta) |
|------------|----------------:|----------------------------:|
| Botellones | 161             | 161                         |
| Camisetas  | 86              | 86                          |
| **Total**  | **247**         | **247**                     |

División train/val: aproximadamente 80/20 por clase.

### 4.3 Extracción de bounding boxes

Dado que las imágenes etiquetadas contienen un rectángulo verde (no un bounding box en formato texto), se implementó un pipeline de visión clásica para extraer automáticamente las coordenadas:

1. Convertir a espacio de color HSV.
2. Aplicar máscara sobre rango de tonos verdes (`H: 40–80, S: 60–255, V: 40–255`).
3. Aplicar operaciones morfológicas (erosión + dilatación) para eliminar ruido.
4. Detectar contornos y encontrar el bounding box del contorno de mayor área.
5. Convertir a formato YOLO normalizado y exportar como `.txt`.

### 4.4 Conjunto de prueba (imágenes reales)

Para evaluar la capacidad de generalización del modelo, se construye un conjunto de prueba adicional con imágenes descargadas de internet (Google Images). Estas imágenes representan condiciones más realistas: fondos variables, distintas orientaciones, iluminación heterogénea.

El ground truth de este conjunto se genera mediante el módulo de anotación interactivo (`tools/annotator.py`), donde cada miembro del equipo anota manualmente la zona de logo esperada. Las anotaciones se fusionan usando `tools/iou_merge.py`, que calcula el IoU inter-anotador y exporta la caja promedio como ground truth.

---

## 5. Metodología

### 5.1 Arquitectura

Se utiliza **YOLOv8n** (nano) como modelo base:
- Backbone: CSPDarknet con cuello FPN/PAN
- Cabeza: anchor-free, detección por distribución (DFL)
- Parámetros: ~3.2 M
- Inicialización: pesos preentrenados en COCO (`yolov8n.pt`)

Las capas del backbone se inician con pesos COCO y se permiten actualizar durante el entrenamiento. Dado el tamaño pequeño del dataset, se utilizan augmentaciones agresivas para regularizar.

### 5.2 Hiperparámetros de entrenamiento

| Parámetro       | Valor  | Justificación                              |
|-----------------|--------|--------------------------------------------|
| Epochs          | 100    | Con early stopping (patience=20)           |
| Batch size      | 4      | Limitado por RAM en CPU                    |
| Image size      | 640    | Resolución estándar YOLOv8                 |
| Optimizer       | AdamW  | Mejor convergencia con datasets pequeños   |
| lr₀             | 0.001  |                                            |
| lrf             | 0.01   | LR final = lr₀ × lrf                       |
| Warmup epochs   | 3      |                                            |
| Mosaic          | 1.0    | Aumentación clave para datasets pequeños   |
| Mixup           | 0.1    |                                            |
| Flip LR         | 0.5    |                                            |

### 5.3 Módulo de anotación interactiva (Validación IoU)

Para construir el conjunto de prueba con imágenes reales se desarrolló `tools/annotator.py`, una herramienta de escritorio con interfaz gráfica (tkinter) que permite a cada anotador:

1. Cargar una carpeta de imágenes descargadas de internet.
2. Dibujar un bounding box con arrastrar y soltar del mouse.
3. Seleccionar la clase (botellón o camiseta).
4. Confirmar o saltar cada imagen con atajos de teclado.
5. Las anotaciones se guardan en `test_set/annotations/<nombre>.json`.

Posteriormente, `tools/iou_merge.py` lee los JSONs de todos los anotadores, calcula el IoU promedio entre pares como métrica de acuerdo, y exporta las etiquetas YOLO para las imágenes que superan el umbral mínimo de acuerdo (por defecto IoU ≥ 0.3).

### 5.4 Módulo de demostración

La aplicación de demostración (`src/demo.py`) está implementada con Streamlit y permite:

1. Subir una imagen de producto (camisa o botellón).
2. Subir un logo (PNG con canal alpha recomendado).
3. Ejecutar inferencia con el modelo entrenado.
4. Componer el logo sobre la zona detectada respetando el canal de transparencia.
5. Visualizar y descargar el resultado.

La composición mantiene la relación de aspecto del logo, lo centra dentro del bounding box detectado, y aplica un nivel de opacidad configurable.

---

## 6. Resultados

> **Nota:** Esta sección se completará una vez finalizado el entrenamiento del modelo. Los valores presentados son marcadores de posición.

### 6.1 Métricas de entrenamiento

| Métrica              | Valor    |
|----------------------|----------|
| mAP@50 (val)         | —        |
| mAP@50-95 (val)      | —        |
| Precision (val)      | —        |
| Recall (val)         | —        |
| Epochs hasta early stop | —     |

### 6.2 Evaluación en conjunto de prueba

| Imagen       | Anotadores | IoU inter-anotador | IoU modelo vs GT |
|--------------|:----------:|:------------------:|:----------------:|
| (pendiente)  |            |                    |                  |

### 6.3 Ejemplos cualitativos

*(Agregar capturas de pantalla del demo con ejemplos de botellones y camisetas.)*

---

## 7. Análisis

> Esta sección se completará con los resultados del entrenamiento.

Puntos a analizar:
- Comparación de desempeño entre clases (bottles vs tshirts).
- Casos de falla: imágenes con fondos complejos, productos con logos ya impresos, o ángulos atípicos.
- Diferencia entre métricas en val (imágenes Makito, condiciones controladas) vs test (imágenes reales de internet).
- Grado de acuerdo inter-anotador en el conjunto de prueba como indicador de la dificultad de la tarea.

---

## 8. Conclusiones

> Esta sección se completará al finalizar el proyecto.

Aspectos a incluir:
- Viabilidad del enfoque de extraer ground truth desde imágenes de catálogo con marcas de color.
- Desempeño del Transfer Learning con datasets pequeños (~150 imágenes por clase).
- Utilidad del módulo de validación IoU como herramienta de construcción de ground truth colaborativo.
- Limitaciones del sistema y posibles mejoras (segmentación en lugar de bounding box, deformación perspectiva del logo, soporte para más categorías de productos).

---

## 9. Referencias

1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You only look once: Unified, real-time object detection.* CVPR 2016.
2. Jocher, G., et al. (2023). *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics
3. Pan, S. J., & Yang, Q. (2010). *A survey on transfer learning.* IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345–1359.
4. Everingham, M., et al. (2010). *The PASCAL Visual Object Classes (VOC) challenge.* International Journal of Computer Vision, 88(2), 303–338.
5. Lin, T. Y., et al. (2014). *Microsoft COCO: Common objects in context.* ECCV 2014.
6. Makito — Artículos Publicitarios Personalizados. https://www.makito.es

---

## Apéndice A — Estructura del Repositorio

```
proyecto_vision/
├── dataset/                  # Dataset YOLO (train/val)
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   └── labels/val/
├── test_set/                 # Conjunto de prueba con imágenes reales
│   ├── images/               # Imágenes descargadas de Google Images
│   ├── annotations/          # JSON por anotador (generado por annotator.py)
│   └── labels/               # Ground truth YOLO (generado por iou_merge.py)
├── data_mining/              # Pipeline de extracción de bounding boxes
├── tools/
│   ├── annotator.py          # Herramienta de anotación interactiva
│   └── iou_merge.py          # Fusión de anotaciones y exportación YOLO
├── src/
│   ├── train.py              # Entrenamiento YOLOv8n
│   ├── predict.py            # Inferencia por línea de comandos
│   └── demo.py               # Aplicación Streamlit de demostración
├── docs/
│   ├── plan.ipynb            # Plan original del proyecto
│   └── informe_final.md      # Este documento
└── requirements.txt
```

## Apéndice B — Instrucciones de Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt
pip install streamlit

# 2. Entrenar el modelo
python src/train.py

# 3. Construir conjunto de prueba (cada integrante corre esto)
python tools/annotator.py --name Francis
python tools/annotator.py --name Cesar
python tools/annotator.py --name Jose

# 4. Fusionar anotaciones y exportar ground truth
python tools/iou_merge.py

# 5. Evaluar en conjunto de prueba
python src/predict.py --source test_set/images/

# 6. Lanzar demo interactivo
streamlit run src/demo.py
```