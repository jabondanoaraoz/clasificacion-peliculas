# Clasificación de Géneros de Películas con NLP

Proyecto de Machine Learning para clasificar géneros cinematográficos (multi-etiqueta) a partir de sinopsis en inglés. Dada la trama de una película, el modelo predice la probabilidad de que pertenezca a cada uno de los 24 géneros posibles.

---

## Problema

Una película puede pertenecer a múltiples géneros simultáneamente (Drama + Crime + Thriller). El reto es predecir esas combinaciones a partir únicamente del texto de la sinopsis, en un corpus con fuerte desbalanceo: Drama y Comedy dominan ampliamente, mientras que News, Film-Noir y Short tienen menos de 100 ejemplos.

## Dataset

- **Fuente:** González & Arevalo (2017) — [arXiv:1702.01992](https://arxiv.org/abs/1702.01992)
- **Entrenamiento:** 7,895 películas × 5 variables
- **Test:** 3,383 películas × 4 variables
- **Variables:** `year`, `title`, `plot` (sinopsis), `genres`, `rating`
- **Géneros:** 24 clases — Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, News, Romance, Sci-Fi, Short, Sport, Thriller, War, Western

## Estructura del Proyecto

```
📁 clasificacion-peliculas/
├── data/
│   ├── dataTraining.csv
│   └── dataTesting.csv
├── 01_exploracion_datos.ipynb
├── 02_ingenieria_caracteristicas.ipynb
├── 03_modelado.ipynb
├── 04_resultados_conclusiones.ipynb
└── README.md
```

## Stack Tecnológico

| Categoría | Herramientas |
|---|---|
| Análisis de datos | pandas, numpy |
| Visualización | matplotlib, seaborn |
| NLP | NLTK (stopwords, lematización WordNet), scikit-learn (TF-IDF, CountVectorizer)|
| Machine Learning | scikit-learn (Logistic Regression, Random Forest, Ridge, Naive Bayes, GridSearchCV)|
| Gradient Boosting | LightGBM |
| Deep Learning | TensorFlow / Keras (redes densas, BiLSTM, GloVe embeddings) |

## Resultados

| Modelo | Representación | ROC-AUC macro |
|--------|----------------|---------------|
| Logistic Regression (trigramas + lematización) | TF-IDF | 0.888 |
| Feature Union (TF-IDF + longitud) | TF-IDF | 0.886 |
| Ridge Classifier | TF-IDF | 0.884 |
| LR baseline | TF-IDF | 0.884 |
| Red Neuronal Densa | TF-IDF | 0.880 |
| Naive Bayes (ComplementNB) | CountVectorizer | 0.876 |
| BiLSTM + GloVe 100d | GloVe | 0.874 |
| LightGBM | TF-IDF | 0.830 |
| Random Forest | TF-IDF | 0.820 |

**Hallazgo principal:** El modelo TF-IDF con trigramas + Logistic Regression OneVsRest (ROC-AUC 0.888) es el mejor modelo en todos los enfoques evaluados.

## Cómo ejecutar

```
pip install -r requirements.txt

jupyter notebook
```

Ejecutar los notebooks en orden: `01` → `02` → `03` → `04`.

> **GloVe (opcional):** El modelo BiLSTM del notebook 03 requiere `glove.6B.100d.txt` en `data/`. Descarga desde [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/).

---

## Contexto Académico

Proyecto desarrollado como parte del programa de **Maestría en Inteligencia Analítica de Datos (MIAD)** de la Universidad de los Andes, Colombia. Equipo de trabajo de 4 personas: Tatiana Garcia, Omar Albarracin, Edwin Ramirez y Joaquín Abondano.

---

## Autor de este repositorio

**Joaquín Abondano Araoz** - Data Analytics · AI & Automation · Strategic Planning

[Website](https://joaquinabondano.com) · [LinkedIn](https://linkedin.com/in/joaquin-abondano) · [GitHub](https://github.com/jabondanoaraoz)
