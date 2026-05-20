# TFG: Análisis empírico comparativo de modelos fundacionales para segmentación de imágenes

Este repositorio contiene el código asociado al Trabajo de Fin de Grado *Implementación y Análisis empírico comparativo de soluciones Fundacionales de última generación para Segmentación de Imágenes*, desarrollado dentro del Grado en Ciencia e Ingeniería de Datos de la Universidad de Las Palmas de Gran Canaria.

El proyecto evalúa el rendimiento de la familia de modelos fundacionales *Segment Anything* (SAM, SAM 2, SAM 2.1 y SAM 3) sobre cinco conjuntos de datos pertenecientes a dominios visuales heterogéneos. Para cada combinación de modelo y conjunto de datos se mide la calidad geométrica de las predicciones (mIoU, Dice, Boundary IoU, HD95, entre otras) y su eficiencia operativa (latencia y consumo de VRAM). Además, se realiza una comparación de los modelos de la familia SAM frente a arquitecturas supervisadas clásicas (U-Net y YOLOv8-Seg) en el dominio médico (Kvasir-SEG).

## Estructura del repositorio

```
TFG-foundation-models-segmentation/
├── configs/                  # Configuración de rutas y de entrenamiento
│   ├── paths.yaml
│   └── training.yaml
├── notebooks/                # Cuadernos auxiliares para generar figuras
├── scripts/                  # Puntos de entrada de la experimentación
│   ├── run_all.py
│   ├── run_finetune.py
│   ├── run_finetune_text_eval.py
│   ├── run_zero_shot.py
│   └── train_classical.py
├── src/                      # Código fuente del proyecto
│   ├── data/                 # Datasets y particionado
│   ├── evaluation/           # Inferencia, métricas y evaluadores
│   └── models/               # Implementación de cada arquitectura
│       ├── classical/        # U-Net y YOLOv8-Seg
│       ├── sam/               # SAM 1 (fine tuning)
│       ├── sam2/              # SAM 2 (fine tuning)
│       ├── sam2_1/            # SAM 2.1 (delega en sam2)
│       └── sam3/              # SAM 3 (fine tuning)
├── pyproject.toml
├── requirements.txt
└── README.md
```

La carpeta `notebooks/` contiene los cuadernos utilizados para generar las figuras del análisis exploratorio y de los resultados que aparecen en la memoria del TFG. No forma parte del flujo experimental principal y se incluye únicamente con fines de reproducibilidad.

## Conjuntos de datos evaluados

| Dataset | Dominio | Tipo de anotación |
|---|---|---|
| Kvasir-SEG | Médico (endoscopia gastrointestinal) | Máscaras binarias + bounding boxes |
| ISIC 2016 | Médico (dermatoscopia) | Máscaras binarias |
| PASCAL-S | Natural (objetos salientes) | Máscaras binarias |
| RefCOCOg | Visión general con componente textual | Máscaras binarias + expresiones referenciales |
| Mapillary Vistas | Urbano (instancias en escena de calle) | Máscaras de instancia |

## Modelos incluidos

- **U-Net** (encoder ResNet-34, supervisado).
- **YOLOv8-Seg** (variante *nano*, supervisado).
- **SAM** (Base y Large), evaluados en *zero-shot* y *fine tuning* del *mask decoder*.
- **SAM 2** y **SAM 2.1** (Base y Large), evaluados en *zero-shot* y *fine tuning*.
- **SAM 3**, evaluado tanto con *prompts* geométricos como con descripciones textuales, en *zero-shot* y *fine tuning*. La evaluación textual del modelo *fine-tuneado* utiliza Grounding DINO como detector previo.

## Requisitos

- Python 3.10.
- GPU compatible con CUDA (los experimentos se ejecutaron sobre una NVIDIA RTX 3090 con CUDA 12.1).
- Aproximadamente 24 GB de VRAM para los modelos *Large* y SAM 3.

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/danilp10/TFG-foundation-models-segmentation.git
cd TFG-foundation-models-segmentation
```

### 2. Instalar PyTorch

PyTorch debe instalarse de forma específica según la versión de CUDA disponible en el sistema. Consultar la [guía oficial de instalación de PyTorch](https://pytorch.org/get-started/locally/) para obtener el comando correspondiente. A modo de referencia, la instalación utilizada en este proyecto, con CUDA 12.1, fue:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Si el sistema dispone de otra versión de CUDA (11.8, 12.4, etc.) o se desea ejecutar en CPU, el comando cambia. No instalar PyTorch desde `requirements.txt` directamente, ya que la versión que se instalaría por defecto puede no coincidir con la versión de CUDA del equipo.

### 3. Instalar el resto de dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar SAM 2 manualmente

SAM 2 no se distribuye a través de PyPI, por lo que es necesario clonar el repositorio oficial e instalarlo en modo editable:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

Se recomienda clonar SAM 2 **fuera del directorio de este proyecto** para evitar conflictos de importación con paquetes ya instalados.

### 5. Instalar Grounding DINO manualmente (solo si se evalúa SAM 3 con texto)

La evaluación de SAM 3 *fine-tuneado* con *prompts* textuales requiere Grounding DINO como detector previo. Se instala también clonando el repositorio oficial:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

A continuación, descargar los pesos de Grounding DINO (por ejemplo, `groundingdino_swint_ogc.pth`) desde el [repositorio oficial de releases](https://github.com/IDEA-Research/GroundingDINO/releases) y actualizar la ruta correspondiente en `configs/paths.yaml`.

### 6. Pesos preentrenados de SAM

Los pesos oficiales de SAM, SAM 2, SAM 2.1 y SAM 3 deben descargarse manualmente desde sus respectivos repositorios oficiales (`facebookresearch/segment-anything`, `facebookresearch/sam2` y Ultralytics) y colocarse en la ruta indicada en `configs/paths.yaml`.

## Configuración

Antes de ejecutar cualquier experimento, editar `configs/paths.yaml` para apuntar a la ubicación local de:

- Los conjuntos de datos.
- Los pesos preentrenados de cada modelo.
- Los directorios de salida para los resultados y los pesos *fine-tuneados*.

Los hiperparámetros del entrenamiento se centralizan en `configs/training.yaml`.

## Uso

### Evaluación *zero-shot*

```bash
python -m scripts.run_zero_shot --model sam3 --dataset kvasir
```

### Fine tuning

```bash
python -m scripts.run_finetune --model sam3 --dataset kvasir
```

### Evaluación textual de SAM 3 fine-tuneado

```bash
python -m scripts.run_finetune_text_eval --dataset refcocog
```

### Entrenamiento de las arquitecturas clásicas

```bash
python -m scripts.train_classical --model unet --dataset kvasir
python -m scripts.train_classical --model yolo --dataset kvasir
```

### Ejecución en cadena

El script `run_all.py` permite lanzar todos los experimentos previstos de forma secuencial:

```bash
python -m scripts.run_all --zero-shot       # Todos los modelos en zero-shot
python -m scripts.run_all --finetune        # Todos los modelos en fine tuning
python -m scripts.run_all --classical       # U-Net y YOLOv8-Seg
python -m scripts.run_all --text-eval       # SAM 3 fine-tuneado con texto
python -m scripts.run_all --all             # Todo lo anterior
```

## Resultados

Los resultados numéricos de cada experimento se almacenan como ficheros CSV en los directorios definidos en `configs/paths.yaml`. Por defecto:

- `resultados_zero_shot/` contiene un fichero por conjunto de datos con los resultados de todas las variantes evaluadas en *zero-shot*.
- `resultados_finetuned/` contiene los resultados tras el proceso de *fine tuning*.

## Autor

Daniel Talavera Hernández.
Trabajo de Fin de Grado del Grado en Ciencia e Ingeniería de Datos.
Universidad de Las Palmas de Gran Canaria, curso 2025/2026.

Tutores: Francisco Mario Hernández Tejera y Cristian David Estupiñán Ojeda.
