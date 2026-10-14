# CNN (DenseNet-121) vs. Transformers (DeiT-S vs. Swin-T) vs. Híbrido (CoAtNet-0) en radiografías de tórax

**Néstor Issel Hernández Rojo** · Proyecto final, Módulo 8: Aprendizaje Profundo · Maestría en Inteligencia Artificial, Universidad Panamericana

Comparación de cuatro familias de redes profundas para detectar 14 enfermedades en **ChestX-ray14** (112,120 radiografías de tórax), con la separación oficial por paciente y **el mismo protocolo para los cuatro modelos**. El artículo completo, en formato IEEE, está en [`docs/articulo_final.pdf`](docs/articulo_final.pdf).

## Los cuatro notebooks

Cada notebook es independiente: descarga los datos, construye el modelo, lo entrena con el 100 % de los pacientes, lo evalúa y guarda sus resultados. Están escritos en el estilo del curso: cada celda empieza con un bloque que dice qué hace y por qué, y las ideas clave se explican con sus fórmulas junto al código.

| Notebook | Familia | Parámetros | AUROC test (IC 95 %) | Notebook ejecutado en Kaggle |
|---|---|---|---|---|
| [`M8_DenseNet121-LeakyReLU.ipynb`](notebooks/M8_DenseNet121-LeakyReLU.ipynb) | CNN, implementada capa por capa (LeakyReLU 0.01) | 7.0 M | 0.805 (0.799–0.811) | [chestxray14-m8-densenet121-leakyrelu](https://www.kaggle.com/code/nestorhernandezrojo/chestxray14-m8-densenet121-leakyrelu) |
| [`M8_DeiT-S.ipynb`](notebooks/M8_DeiT-S.ipynb) | Transformer puro | 21.7 M | 0.785 (0.776–0.791) | [chestxray14-m8-deit-s](https://www.kaggle.com/code/nestorhernandezrojo/chestxray14-m8-deit-s) |
| [`M8_Swin-T.ipynb`](notebooks/M8_Swin-T.ipynb) | Transformer jerárquico | 27.5 M | 0.808 (0.803–0.813) | [chestxray14-m8-swin-t](https://www.kaggle.com/code/nestorhernandezrojo/chestxray14-m8-swin-t) |
| [`M8_CoAtNet-0.ipynb`](notebooks/M8_CoAtNet-0.ipynb) | Híbrido (convolución + atención) | 26.7 M | 0.809 (0.803–0.814) | [chestxray14-m8-coatnet-0](https://www.kaggle.com/code/nestorhernandezrojo/chestxray14-m8-coatnet-0) |

En los notebooks de Kaggle se ven todas las salidas de la corrida real: curvas, métricas, intervalos de confianza y costo.

## Cómo ejecutarlos

### En Kaggle (recomendado, con GPU)

1. Abre el notebook ejecutado del modelo en Kaggle (enlaces de la tabla) y haz clic en **Copy & Edit**.
2. Revisa que tenga como entrada el dataset público [`khanfashee/nih-chest-x-ray-14-224x224-resized`](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized), con las imágenes de 224×224, `Data_Entry_2017.csv` y las listas oficiales `train_val_list` y `test_list`.
3. En *Settings*, elige **GPU T4** y activa **Internet**, para que timm descargue los pesos de ImageNet si no los encuentra en los datos.
4. Ejecuta **Run All**. En una Tesla T4 tarda aproximadamente 40 minutos (DeiT-S), 52 (DenseNet-121), 79 (CoAtNet-0) y 85 (Swin-T).

### En una computadora propia

```
pip install -r requirements.txt
```

1. Descarga el mismo dataset de Kaggle y descomprímelo en una carpeta `data/` en la raíz de este repositorio. Los notebooks la buscan en `../data`, desde la carpeta `notebooks/`.
2. Abre el notebook. La función `buscar_archivo` localiza los archivos de datos.
3. Fuera de Kaggle se activa solo `MODO_PRUEBA`: pocas imágenes y una época, para revisar que todo corra. Para la corrida completa, pon `MODO_PRUEBA = False` en el PASO 2. Hace falta una GPU.

## Receta común (lo único que cambia es la arquitectura)

- **Preentrenamiento:** pesos de ImageNet-1k de la biblioteca [timm](https://github.com/huggingface/pytorch-image-models). Se ajusta toda la red y se cambia la capa final por una de 14 salidas.
- **Optimización:** AdamW con tasa de 10⁻⁴, weight decay de 0.05 y coseno; hasta 10 épocas con early stopping (paciencia de 3) según el AUROC de validación.
- **Pérdida:** BCE con peso √(N⁻/N⁺) para los casos positivos de cada enfermedad.
- **Aumento de datos:** recorte aleatorio (80–100 %), rotación de ±7° y brillo y contraste de ±10 %. No se voltea la imagen.
- **Entrenamiento:** lotes de 32 en precisión mixta, semilla 42 y cuDNN determinista.
- **Datos:** separación oficial por paciente. Validación: 10 % de los pacientes de entrenamiento. Test: la lista oficial, con 25,596 radiografías.
- **Métricas:** AUROC promedio de las 14 enfermedades (principal), con intervalo de confianza de 95 % por paciente (bootstrap). También F1 macro, accuracy, Brier score con calibración de Platt, costo computacional (parámetros, tiempo, memoria) y curvas de convergencia.

## Qué guarda cada notebook

En `saved_models/`:
- `<Modelo>.ckpt`: la mejor época;
- `preds_<Modelo>_{val,test}.csv`: probabilidades por radiografía;
- `costo_<Modelo>.json`: tiempos, memoria y épocas;
- `historia_<Modelo>.json`: pérdida y AUROC por época;
- `resultados_<Modelo>.json` y `resumen_<Modelo>.csv`: métricas finales.

## Requisitos

Python 3.10 o más reciente, PyTorch con GPU (CUDA), timm, torchvision, pandas, numpy, scikit-learn, matplotlib y Pillow (ver `requirements.txt`).
