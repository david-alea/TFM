# Análisis comparativo de arquitecturas de aprendizaje profundo para PLN

Código experimental del Trabajo Fin de Máster *Análisis comparativo de arquitecturas emergentes de aprendizaje profundo para el procesamiento del lenguaje natural* (Máster en Técnicas Estadísticas, USC/UDC).

Se comparan ocho arquitecturas de modelado de secuencias — FFN, LSTM, Transformer, Transformer+RoPE, Transformer+ALiBi, Transformer+RoPE+MoE, Mamba y Jamba — sobre tres tareas: clasificación de texto, modelado de lenguaje (WikiText-103) y razonamiento composicional (LRA ListOps).

Todo el código está pensado para ejecutarse **en Databricks**, sobre un clúster GPU de un solo nodo, usando `TorchDistributor` (PyTorch DDP) y MLflow para el registro de experimentos. No hay scripts `.py` ni `main`: cada notebook es autocontenido.

---

## Contenido del repositorio

| Fichero | Tarea | Patrón de ejecución |
|---|---|---|
| `Clasificacion_textos.ipynb` | Clasificación (20 Newsgroups y corpus PROGRESO) | Notebook parametrizado, lanzado como *Job* |
| `Parametros_clasificacion.ipynb` | Parámetros de la comparación de clasificación | Se usa como entrada de configuración para el *Job* de clasificación |
| `WikiText_Final.ipynb` | Modelado de lenguaje (WikiText-103) | Notebook todo-en-uno, ejecución interactiva |
| `LRA_ListOps_Final.ipynb` | Clasificación de secuencias (LRA ListOps) | Notebook todo-en-uno, ejecución interactiva |
| `datos_progreso.csv` | Resultados exportados desde MLflow | Entrada del análisis y de las figuras del TFM |

---

## Requisitos del entorno

- Clúster Databricks de **un nodo con varias GPU** (los resultados del TFM se obtuvieron con 8×A10G).
- Databricks Runtime **ML con GPU** (incluye PyTorch, MLflow y `pyspark.ml.torch.distributor`).
- Los notebooks de WikiText y LRA instalan `mamba-ssm` al principio:

  ```python
  %pip install --no-build-isolation "mamba-ssm[causal-conv1d]==2.2.2"
  dbutils.library.restartPython()
  ```

  Esa celda hay que ejecutarla **antes que ninguna otra**: reinicia el intérprete de Python y borra el estado del notebook. Si `mamba_ssm` no llega a importarse, los notebooks siguen funcionando pero omiten Mamba y Jamba.

- Los datasets se descargan de Hugging Face al primer uso y quedan cacheados en DBFS (`CACHE_DIR`). El corpus PROGRESO se lee desde CSV local (`df_ml_topic_*.csv`, `df_ml_genre_*.csv`).

---

## Cómo se ejecuta cada cosa

### 1. Notebooks todo-en-uno (WikiText y LRA ListOps)

Son los dos notebooks principales del estudio comparativo. Comparten estructura y se usan igual:

1. Ejecutar la celda de instalación de `mamba-ssm` y esperar al reinicio del intérprete.
2. **Editar la clase `CFG`** (sección *1) Config*). Es el único punto que se toca entre ejecuciones. En la práctica solo se cambia la longitud de secuencia:
   - WikiText: `BLOCK_SIZE` ∈ {128, 256, 512, 1024}
   - LRA ListOps: `MAX_SEQ_LEN` ∈ {512, 1024, 2048}

   El tamaño de lote se deriva automáticamente de `TOKENS_PER_GPU_TARGET / longitud`, de modo que el número de tokens por paso y GPU se mantiene constante al variar la longitud. El resto de hiperparámetros (`D_MODEL=256`, `N_LAYERS=4`, `N_HEADS=8`, `FFN_DIM=1024`, `LR=3e-4`, `SEED=42`) es común a todas las arquitecturas y no se modifica.
3. Ejecutar el notebook de arriba abajo. Las secciones intermedias solo definen cosas: dataset y tokenización, las ocho arquitecturas, la función de pérdida y las rutinas de DDP.
4. La última celda es la que lanza el entrenamiento:

   ```python
   TorchDistributor(num_processes=8, local_mode=True, use_gpu=True).run(ddp_worker)
   ```

   `num_processes` debe coincidir con el número de GPU del nodo.

Dentro de `ddp_worker` se entrenan **las ocho arquitecturas de forma secuencial** en una sola ejecución. Cada proceso construye el mismo modelo y lo envuelve en `DistributedDataParallel`; MLflow solo registra desde el rango 0, con un *run* padre por ejecución y un *run* anidado por arquitectura.

Un barrido completo consiste, por tanto, en ejecutar el notebook una vez por cada longitud de secuencia: 4 ejecuciones para WikiText y 3 para ListOps.

### 2. Notebook parametrizado (clasificación)

`Clasificacion_textos.ipynb` sigue el patrón opuesto: **una ejecución = una arquitectura con una configuración concreta**. Los parámetros se declaran como *widgets* de Databricks al principio del notebook:

```python
dbutils.widgets.text("model_type", "ffn")
dbutils.widgets.text("dataset", "20news")
dbutils.widgets.text("max_len", "256")
dbutils.widgets.text("lr", "0.001")
dbutils.widgets.text("seed", "13")
dbutils.widgets.text("num_processes", "2")
...
```

Esto permite dos modos de uso:

- **Interactivo**: fijar los valores en la barra de widgets y ejecutar el notebook.
- **Como Job** (el modo usado en el TFM): crear un *Job* de Databricks que apunte al notebook y lanzar una tarea por combinación de `model_type` × `max_len` × `lr` × `seed`. Cada tarea escribe su propio *run* en MLflow.

La celda final recoge los widgets en un diccionario `cfg` y lo pasa a `TorchDistributor(...).run(main_worker, cfg)`. El notebook incluye además dos celdas de *sanity check* (sin DDP) para comprobar formas de tensores y el efecto de la máscara de atención antes de lanzar un barrido largo.

---

## Registro de resultados

Todos los entrenamientos escriben en MLflow (métricas de pérdida, perplejidad o *accuracy*, `ms_per_step`, `tokens_per_sec` y pico de memoria GPU, más los pesos finales como artefacto).

---

## Notas de reproducibilidad

- La semilla es fija (`SEED=42`, o el widget `seed` en clasificación), pero los *kernels* CUDA de Mamba no son deterministas bit a bit: pueden aparecer diferencias pequeñas entre repeticiones.
- `USE_AMP=False` de forma deliberada: la precisión mixta da problemas con los *kernels* de Mamba y con parte de los bloques Transformer.
- `USE_GRAD_CHECKPOINTING=True` es necesario para que el Transformer quepa en memoria con las longitudes mayores.
- El entrenamiento no usa *scheduler* de tasa de aprendizaje ni *weight tying*, para que la comparación entre arquitecturas dependa solo del *backbone*.
