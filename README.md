# Clinical Cancer Staging: Titan SOTA Pipeline 

  **Choose your language / Elige tu idioma:** [🇺🇸 English Version](#-english-version) | [🇪🇸 Versión en Español](#-versión-en-español)
  
---

## 🇺🇸 English Version

### 1. Project Overview
This project implements an advanced Natural Language Processing (NLP) pipeline designed to classify raw clinical cancer pathology reports into their precise TNM stages (T1, T2, T3, T4). The final architecture, designated as the **Titan SOTA Ensemble**, moves beyond traditional text classification to achieve state-of-the-art (SOTA) performance through generative data augmentation, specialized medical transformers, and robust ensemble strategies.

### 2. The Engineering Journey: From Baseline to SOTA

Our pipeline is the result of iterative hypothesis testing and empirical failure analysis:

1. **Exploratory Data Analysis (EDA):** We identified two critical bottlenecks: an extreme class imbalance (severe T4 cases were rare) and sequence lengths often exceeding 1,000 words.
2. **Classic Machine Learning:** We established baselines using TF-IDF vectorization paired with models like Logistic Regression and **XGBoost**. While XGBoost performed admirably on tabularized text frequencies, it fundamentally lacked the semantic depth to interpret complex oncological nuances.
3. **Deep Learning from Scratch:** We attempted to train CNNs and LSTMs with basic embeddings. This phase proved that sequence models without pre-trained medical domain knowledge fail to capture the highly specific vocabulary of pathology.
4. **Transition to Transformers (BERT & Clinical-BERT):** Moving to pre-trained models yielded massive improvements in semantic understanding. However, standard BERT is strictly capped at 512 tokens. We implemented a hybrid **BERT + CNN** head to extract local spatial features from the embeddings, but the truncation of long documents remained a critical flaw.

### 3. The Breakthrough: Final SOTA Architecture

To shatter the performance ceiling and address the T4 classification bottleneck, we engineered the definitive **Titan Ensemble**, integrating the following components:

* **Clinical-Longformer Backbone:** Replaced BERT to handle sequence windows of up to 4096 tokens, ensuring no critical diagnostic information at the tail of the reports is truncated.
* **BioGPT Generative Augmentation:** Implemented a pre-trained biomedical generative transformer (BioGPT) to synthesize realistic, "Organ-Aware" clinical reports for the underrepresented T4 class, mathematically neutralizing the dataset imbalance prior to training.
* **3-Fold Stratified Ensemble:** Deployed a Cross-Validation framework to train three distinct "expert" models on 100% of the dataset, leveraging Soft Voting for highly stable final predictions.
* **Focal Label Smoothing Loss:** A custom loss function engineered to simultaneously penalize model overconfidence and aggressively target hard-to-classify edge cases.
* **Test-Time Augmentation (TTA) & Dynamic Thresholding:** During inference, reports are evaluated via multi-view augmentation. Furthermore, Out-of-Fold (OOF) probabilities dynamically calibrate the mathematical threshold required to flag a T4 tumor, eliminating guesswork.

---

## 🇪🇸 Versión en Español

### 1. Resumen del Proyecto
Este proyecto implementa un pipeline avanzado de Procesamiento de Lenguaje Natural (PLN) diseñado para clasificar informes de patología clínica en sus etapas TNM exactas (T1, T2, T3, T4). La arquitectura final, designada como **Titan SOTA Ensemble**, supera la clasificación de texto tradicional para alcanzar un rendimiento del estado del arte (SOTA) mediante el aumento generativo de datos, transformers médicos especializados y estrategias de ensamble robustas.

### 2. El Camino de Ingeniería: Del Baseline al SOTA

Nuestro pipeline es el resultado de la prueba iterativa de hipótesis y el análisis empírico de fallos:

1. **Análisis Exploratorio de Datos (EDA):** Identificamos dos cuellos de botella críticos: un desequilibrio extremo de clases (los casos graves T4 eran muy escasos) y longitudes de secuencia que frecuentemente superaban las 1.000 palabras.
2. **Machine Learning Clásico:** Establecimos líneas base (baselines) usando vectorización TF-IDF emparejada con Regresión Logística y **XGBoost**. Aunque XGBoost rindió de manera sólida con frecuencias de texto tabuladas, carecía de la profundidad semántica para interpretar matices oncológicos complejos.
3. **Deep Learning desde Cero:** Intentamos entrenar CNNs y LSTMs con embeddings básicos. Esta fase demostró que los modelos de secuencia sin conocimiento pre-entrenado del dominio médico no logran capturar el vocabulario altamente específico de la patología.
4. **Transición a Transformers (BERT y Clinical-BERT):** El salto a modelos pre-entrenados produjo mejoras masivas en la comprensión semántica. Sin embargo, el modelo BERT estándar está estrictamente limitado a 512 tokens. Implementamos una cabecera híbrida **BERT + CNN** para extraer características espaciales locales, pero el truncamiento de documentos largos seguía siendo un fallo crítico.

### 3. El Avance Definitivo: Arquitectura Final SOTA

Para romper el techo de rendimiento y solucionar el cuello de botella en la clasificación de T4, diseñamos el **Titan Ensemble** definitivo, integrando los siguientes componentes:

* **Backbone Clinical-Longformer:** Reemplazó a BERT para manejar ventanas de secuencia de hasta 4096 tokens, asegurando que ninguna información diagnóstica crítica al final de los informes se pierda.
* **Aumento Generativo con BioGPT:** Implementamos un transformer generativo biomédico (BioGPT) para sintetizar informes clínicos realistas y "Organ-Aware" para la clase T4 subrepresentada, neutralizando matemáticamente el desequilibrio antes del entrenamiento.
* **Ensamble Stratified de 3 Folds:** Desplegamos un marco de Validación Cruzada para entrenar tres modelos "expertos" distintos con el 100% de los datos, aprovechando el *Soft Voting* para predicciones finales ultra estables.
* **Focal Label Smoothing Loss:** Una función de pérdida personalizada diseñada para penalizar simultáneamente el exceso de confianza del modelo y atacar agresivamente los casos límite difíciles de clasificar.
* **Test-Time Augmentation (TTA) y Umbrales Dinámicos:** Durante la inferencia, los informes se evalúan mediante aumentos de múltiples vistas. Además, las probabilidades Out-of-Fold (OOF) calibran dinámicamente el umbral matemático requerido para marcar un tumor T4, eliminando las suposiciones a ciegas.

---

##  References / Referencias Científicas

1. Li, Y., et al. (2022). *Clinical-Longformer and Clinical-BigBird: Transformers for long clinical sequences*. [arXiv:2201.11838](https://arxiv.org/abs/2201.11838).
2. Luo, R., et al. (2022). *BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining*. Briefings in Bioinformatics.
3. Lin, T. Y., et al. (2017). *Focal Loss for Dense Object Detection*. IEEE International Conference on Computer Vision (ICCV).
