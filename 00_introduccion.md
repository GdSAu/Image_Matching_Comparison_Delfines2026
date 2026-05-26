# Introducción al Tutorial: Emparejamiento de Imágenes

## El problema que vamos a resolver

Imagina que tomas dos fotos del mismo edificio desde ángulos distintos.
A simple vista puedes reconocer que son del mismo lugar: ves la misma ventana,
el mismo arco, la misma mancha en la pared. Tu cerebro lo hace sin esfuerzo.

Ahora bien, ¿cómo le enseñas a una computadora a hacer lo mismo?

Ese es el problema del **emparejamiento de imágenes** (*image matching*): dado un par de
fotografías que comparten contenido visual, encontrar automáticamente qué píxeles
de una imagen corresponden al mismo punto físico del mundo real en la otra imagen.

```
   Imagen 1                        Imagen 2
  ┌──────────┐                   ┌──────────┐
  │    *     │ ←── misma esquina ──→  *     │
  │  *       │                   │      *   │
  │      *   │ ←── misma ventana ──→ *      │
  └──────────┘                   └──────────┘
         correspondencias encontradas
```

El resultado es un conjunto de pares de puntos (**correspondencias**), donde cada par
indica: "este punto en la imagen 1 y este otro punto en la imagen 2 son el mismo
lugar del mundo".

---

## ¿Por qué importa este problema?

El emparejamiento de imágenes es un bloque fundamental en una gran cantidad de
aplicaciones del mundo real. Sin él, ninguna de las siguientes tecnologías funcionaría:

### Reconstrucción 3D y mapeo
Cuando Google Street View o un dron capturan cientos de fotos de un lugar y las
convierten en un modelo 3D navegable, el primer paso es encontrar correspondencias
entre todas las imágenes para saber dónde estaba la cámara en cada toma.
Este proceso se llama **Structure from Motion (SfM)**.

### Realidad Aumentada (AR)
Para que un objeto virtual "se pegue" de forma convincente a una superficie del
mundo real, el sistema necesita saber exactamente cómo se movió la cámara entre
fotogramas. Eso requiere emparejar puntos entre imágenes consecutivas en tiempo real.

### Robótica y vehículos autónomos
Un robot que navega por un entorno desconocido necesita construir un mapa mientras
se localiza dentro de él (**SLAM** — Simultaneous Localization and Mapping). El
emparejamiento de imágenes le permite reconocer lugares que ya visitó y corregir
su estimación de posición.

### Búsqueda visual y reconocimiento de objetos
"¿Dónde está este producto en la foto?" o "¿cuál es el cuadro del museo que estoy
viendo?" son preguntas que se resuelven encontrando correspondencias entre una
imagen de consulta y una base de datos de imágenes de referencia.

### Medicina e industria
El registro de imágenes médicas (alinear una radiografía con una resonancia del mismo
paciente) y la inspección industrial automatizada (detectar defectos comparando piezas)
también dependen de variantes de este problema.

---

## El reto técnico

El problema suena sencillo, pero en la práctica es difícil por varias razones:

| Dificultad | Ejemplo |
|---|---|
| **Cambio de punto de vista** | La misma ventana se ve diferente desde 45° que desde 90° |
| **Cambio de iluminación** | El mismo edificio de día y de noche parece otro lugar |
| **Escala** | Una foto de cerca y otra de lejos del mismo objeto |
| **Oclusiones** | Un árbol tapa parte del edificio en una foto pero no en la otra |
| **Superficies sin textura** | Una pared blanca lisa no tiene puntos identificables |
| **Escenas repetitivas** | Un pasillo con puertas idénticas cada 3 metros |

Un buen algoritmo de image matching debe ser robusto a todos estos factores
simultáneamente.

---

## La evolución de las soluciones

### Métodos clásicos (1999 – 2015)

Durante muchos años el estándar de la industria fue **SIFT** (*Scale-Invariant
Feature Transform*, Lowe, 1999). La idea general de SIFT —y de todos los métodos
clásicos— es:

1. **Detectar** puntos "interesantes" en la imagen (esquinas, bordes fuertes).
2. **Describir** la apariencia local alrededor de cada punto con un vector numérico
   (el *descriptor*) construido a mano usando gradientes de intensidad.
3. **Emparejar** descriptores de las dos imágenes buscando los más parecidos.

SIFT funcionó muy bien durante más de una década. Pero tiene limitaciones importantes:
es lento, sus descriptores son grandes (128 floats) y su robustez a cambios extremos
de perspectiva o iluminación tiene un techo claro.

### Métodos basados en Deep Learning (2017 – presente)

Con el auge del aprendizaje profundo, surgieron métodos que **aprenden** a detectar
y describir puntos a partir de datos, en lugar de usar reglas escritas a mano.
Las redes neuronales pueden aprender a ser invariantes a los cambios que importan
(iluminación, perspectiva, escala) mientras preservan la discriminabilidad de los
descriptores.

El salto cualitativo más grande llegó con el emparejamiento: en lugar de simplemente
buscar el vecino más cercano en el espacio de descriptores, modelos como **SuperGlue**
y su sucesor **LightGlue** usan mecanismos de **atención** (transformers) para
considerar el contexto global de todos los puntos simultáneamente, aprendiendo a
resolver ambigüedades que el emparejamiento clásico no puede.

---

## Las herramientas que vamos a usar y por qué

### Python + PyTorch

**Python** es el lenguaje dominante en investigación y desarrollo de IA. Prácticamente
todos los modelos del estado del arte se publican con código en Python.

**PyTorch** es el framework de deep learning más usado en investigación (según Papers
With Code, más del 75% de los artículos de ML usan PyTorch). Es la base sobre la que
corren todos los modelos de este tutorial. Aprender a usar PyTorch es una habilidad
directamente transferible al mercado laboral.

---

### ALIKED — El extractor de características

**ALIKED** (*Adaptive and Lightweight Keypoint Detection*, Zhao et al., 2023) es el
modelo que usaremos para el **Paso 1** del pipeline: detectar puntos de interés y
calcular su descriptor.

¿Por qué ALIKED y no SIFT u otro método clásico?

- **Aprendido con datos reales**: sus descriptores son más robustos a cambios de
  iluminación y perspectiva que los descriptores escritos a mano.
- **Ligero**: fue diseñado explícitamente para ser eficiente. Funciona bien en CPU
  y en GPUs de gama media, a diferencia de modelos más grandes como SuperPoint.
- **Descriptores compactos**: genera vectores de 128 dimensiones, el mismo tamaño
  que SIFT pero mucho más discriminativos.
- **Variante `aliked-n16rot`**: el modelo que usaremos fue entrenado para ser
  invariante a rotaciones arbitrarias, lo que lo hace robusto en escenas donde
  la cámara puede estar orientada en cualquier dirección.
- **Relevancia industrial**: ALIKED fue parte de la solución ganadora del
  **Kaggle Image Matching Challenge 2024**, una de las competencias más exigentes
  de la comunidad de visión computacional.

---

### LightGlue — El emparejador

**LightGlue** (Lindenberger et al., 2023) es el modelo que usaremos para el **Paso 2**:
tomar los descriptores que extrajo ALIKED de ambas imágenes y decidir qué pares
corresponden al mismo punto del mundo.

¿Por qué LightGlue y no el emparejamiento por vecino más cercano (brute-force)?

El emparejamiento por fuerza bruta simplemente busca, para cada descriptor de la
imagen 1, el descriptor más parecido en la imagen 2. Esto produce muchos errores en:

- **Texturas repetitivas**: si hay 10 ventanas idénticas, ¿a cuál le corresponde
  cada punto?
- **Puntos en regiones sin textura**: el descriptor es poco discriminativo y hay
  muchos candidatos casi iguales.

LightGlue resuelve esto usando un **transformer con atención cruzada**: en lugar de
comparar cada par de forma aislada, el modelo observa *todos* los puntos de las dos
imágenes a la vez y aprende a usar el contexto global para resolver ambigüedades.
Si sé que el punto A de la imagen 1 está a la izquierda de B y C, puedo usar esa
información para encontrar su correspondencia aunque el descriptor por sí solo sea
ambiguo.

Además, LightGlue tiene una característica clave: **descarta puntos automáticamente**
si no encuentra una correspondencia confiable, en lugar de forzar una pareja incorrecta.
Esto reduce drásticamente los falsos positivos.

---

### RANSAC — El filtro geométrico

Incluso LightGlue puede producir algunos emparejamientos incorrectos.
**RANSAC** (*Random Sample Consensus*) es un algoritmo clásico que usamos como
último filtro para eliminar estos errores.

La idea es elegante: si dos imágenes fueron tomadas por una cámara real del mismo
lugar, las correspondencias correctas deben ser **geométricamente consistentes**.
Es decir, deben poder explicarse mediante una transformación geométrica válida
(la **Matriz Fundamental**, que codifica la relación espacial entre las dos vistas).

RANSAC encuentra la transformación que es consistente con el **mayor número de
correspondencias** y descarta las que no la cumplen (outliers). Los pares que sí
la cumplen se llaman **inliers** y son los que realmente corresponden al mismo punto
del mundo.

---

### Kornia — La biblioteca que conecta todo

**Kornia** es una biblioteca de visión computacional construida sobre PyTorch que
implementa decenas de algoritmos —incluyendo ALIKED y LightGlue— de forma lista
para usar, con una API uniforme y bien documentada.

¿Por qué usamos Kornia y no los repositorios originales de cada modelo?

Cada modelo de investigación se publica con su propio repositorio, sus propias
convenciones de código, sus propias dependencias y su propio formato de entrada/salida.
Integrar varios modelos de repositorios distintos en un mismo pipeline requiere
mucho trabajo de adaptación.

Kornia nos da:

- **Instalación estándar**: `pip install kornia`. Sin clonar repositorios ni
  resolver conflictos de dependencias.
- **API uniforme**: todos los modelos siguen la misma convención de entrada/salida
  (diccionarios de tensores PyTorch), lo que facilita combinarlos.
- **Descarga automática de pesos**: Kornia descarga los modelos pre-entrenados
  automáticamente en la primera ejecución desde un repositorio confiable.
- **Mantenimiento activo**: la biblioteca es mantenida por un equipo dedicado,
  lo que garantiza compatibilidad con versiones nuevas de PyTorch.

---

### Gradio — La interfaz web

**Gradio** es la herramienta que nos permite convertir el pipeline en una
aplicación usable por cualquier persona, sin que el usuario final necesite conocer
Python ni la terminal.

¿Por qué Gradio y no una aplicación web tradicional (Flask, FastAPI + HTML)?

Construir una interfaz web completa con Flask o FastAPI requiere HTML, CSS,
JavaScript y conocimiento de comunicación frontend-backend. Eso está fuera del
alcance de este tutorial.

Gradio nos da una interfaz funcional y visualmente atractiva con unas pocas líneas
de Python, permitiéndonos enfocarnos en lo que importa: el pipeline de visión
computacional. Es ampliamente usada en la comunidad de IA para publicar demos de
modelos y ya la encontrarás en el sitio **Hugging Face Spaces**, donde miles de
modelos tienen demos públicos construidas con ella.

---

## El pipeline completo de un vistazo

```
  Imagen 1  ──┐
               ├──► ALIKED ──► Keypoints + Descriptores ──┐
  Imagen 2  ──┘                                            │
                                                           ▼
                                                       LightGlue
                                                           │
                                                           ▼
                                               Correspondencias brutas
                                                           │
                                                           ▼
                                                        RANSAC
                                                           │
                                                           ▼
                                               Correspondencias filtradas
                                               (inliers geométricamente
                                                consistentes)
                                                           │
                                                           ▼
                                                    Visualización
```

---

## Lo que vas a aprender al terminar este tutorial

Al completar los tres scripts de este tutorial habrás trabajado con habilidades
que son directamente aplicables en la industria y en investigación:

1. **Usar modelos de Deep Learning pre-entrenados** sin entrenarlos desde cero,
   que es como se trabaja el 95% de las veces en la industria.

2. **Entender el ecosistema PyTorch + Kornia**: cómo se representan las imágenes
   como tensores, qué significa la forma `(B, C, H, W)` y cómo fluyen los datos
   a través de un pipeline de modelos.

3. **Aplicar filtrado geométrico con RANSAC**, un algoritmo presente en casi todos
   los sistemas de visión computacional del mundo real.

4. **Construir una demo interactiva con Gradio**, la herramienta estándar para
   publicar aplicaciones de IA rápidamente.

5. **Leer documentación técnica** de modelos de investigación y saber cómo
   integrarlos en tu propio código.

---

## Siguientes pasos

```
00_setup_check.py      ← Verifica que todo esté instalado correctamente
01_aliked_lightglue.py ← Implementa el pipeline completo paso a paso
02_gradio_app.py       ← Convierte el pipeline en una aplicación web
```

Empieza con:

```bash
python 00_setup_check.py
```
