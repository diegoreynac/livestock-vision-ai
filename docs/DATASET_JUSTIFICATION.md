# Dataset Justification

## 1. Purpose

Este documento justifica el uso del dataset seleccionado para el proyecto "Livestock Weight Estimation using Computer Vision and Deep Learning". La justificación se apoya en los resultados reales de la auditoría realizada con los módulos COCOAudit y DatasetStatistics del proyecto, y en la información verificada sobre composición, cobertura y anotaciones. El objetivo es proporcionar una base técnica y científica para su uso en la tesis, diferenciando claramente los datos disponibles de las tareas que requerirán preparación adicional del dataset.

## 2. Dataset Description

El dataset auditado consta de 4,910 imágenes de ganado bovino con 4,749 anotaciones COCO cargadas en el repositorio. La auditoría identificó 161 imágenes sin anotación, lo que confirma que no todas las imágenes son directamente útiles para tareas supervisadas basadas en anotaciones. La información de peso, sexo y tipo de vista está presente en la colección, y su estructura de anotación incluye subgrupos B2 y B4.

## 3. Dataset Composition

### 3.1 Images

- Total de imágenes: 4,910
- Imágenes sin anotación: 161
- Imágenes con anotaciones COCO cargadas: 4,749

### 3.2 Animal Weight

Los valores de peso validados en el dataset son:
- Peso mínimo: 61 kg
- Peso máximo: 481 kg
- Peso promedio: 167.3395 kg
- Mediana: 167.5 kg
- Desviación estándar: 38.164 kg
- Q1: 144 kg
- Q3: 191 kg
- IQR: 47 kg

Este conjunto muestra una dispersión moderada del peso, con un rango amplio que abarca desde animales relativamente ligeros hasta ejemplares de casi 500 kg.

### 3.3 Sex Distribution

La distribución por sexo verificada es la siguiente:
- Hembras: 4,252 imágenes
- Machos: 658 imágenes

Esta relación indica un desbalance importante hacia hembras, lo cual es relevante para la construcción de subconjuntos y para la evaluación de modelos de clasificación de sexo.

### 3.4 Camera Views

El dataset contiene dos vistas principales casi equilibradas:
- Rear: 2,457 imágenes
- Side: 2,453 imágenes

La disponibilidad de vistas posteriores y laterales ofrece diversidad visual para métodos de extracción de características y estimación de peso, aunque la auditoría no reporta segmentaciones COCO válidas.

### 3.5 COCO Annotations

Estructura B2:
- Rear
- Rear_2
- Side
- Side_2

Estructura B4:
- Rear
- Side

Cobertura validada de anotaciones COCO:
- BBox disponible y válido: 865 imágenes
- BBox ausente: 3,884 imágenes
- BBox inválido: 0
- Segmentación disponible: 0
- Segmentación ausente: 4,749 imágenes

Estas cifras muestran que la mayoría de las imágenes no cuentan con bounding boxes anotadas, pero la integridad de los BBoxes presentes es alta.

### 3.6 Keypoints

La auditoría detectó cardinalidades de keypoints variables en las anotaciones:
- 4
- 6
- 9
- 23

Cobertura de keypoints:
- Keypoints válidos: 4,748
- Keypoints inválidos: 1

Aunque los keypoints son mayoritariamente válidos, su cardinalidad variable introduce complejidad para el aprendizaje de modelos basados en poses o geometría.

## 4. Suitability for the Research Problem

La idoneidad del dataset para cada tarea se debe separar entre lo que ya está disponible y lo que exige preparación adicional.

### 4.1 Weight Estimation

- Información disponible: valores de peso para los animales y pares imagen-peso.
- Preparación requerida: selección de un subconjunto con anotaciones confiables y estabilidad visual homogénea; validación de relaciones entre vista, pose y peso.
- Comentario: el dataset es relevante para estimación de peso porque provee las etiquetas de peso objetivo, pero la ausencia de segmentaciones y la cobertura parcial de BBoxes implican que la extracción de características visuales puede ser más difícil si se requiere localización precisa del animal.

### 4.2 Object Detection

- Información disponible: un subconjunto de 865 imágenes con BBoxes COCO válidos.
- Preparación requerida: la mayoría de las imágenes carece de bounding boxes, por lo que la tarea de detección demandará generar o ampliar anotaciones si se pretende entrenar un detector robusto con todo el dataset.
- Comentario: para detección de ganado, la dataset puede usarse inicialmente como conjunto de validación o entrenamiento limitado, pero no como un dataset de detección completo sin anotaciones adicionales.

### 4.3 Sex Classification

- Información disponible: etiquetas de sexo en 4,910 imágenes, con 4,252 hembras y 658 machos.
- Preparación requerida: técnicas de balanceo de clases, muestreo estratificado o ponderación en la función de pérdida para mitigar el desbalance.
- Comentario: el dataset es adecuado para clasificación de sexo, pero el fuerte desbalance debe considerarse en el diseño experimental y en la interpretación de métricas.

### 4.4 Computer Vision Features

- Información disponible: múltiples vistas y keypoints en la mayoría de las imágenes.
- Preparación requerida: definición de una representación consistente de keypoints de cardinalidad variable, y posiblemente extracción de características globales para imágenes sin segmentación.
- Comentario: la diversidad de vistas y la presencia de keypoints habilitan experimentos de extracción de características, aunque la falta de segmentación limita los métodos que dependan de máscaras de objeto.

## 5. Comparison with Alternative Datasets

No se incorporan comparaciones con datasets específicos que no hayan sido verificados por el proyecto. En general, muchos datasets de imágenes de ganado disponibles en la literatura se centran en detección, clasificación de raza o segmentación, pero no siempre incluyen etiquetas de peso ni metadata de sexo para cada imagen.

Para el problema específico de estimación de peso en bovinos, este dataset presenta una característica diferencial relevante: la disponibilidad de valores de peso emparejados con imágenes, lo cual no es un atributo habitual en conjuntos de datos de visión por computador para ganado.

## 6. Dataset Advantages

Las ventajas reales respaldadas por la auditoría son:
- Información de peso disponible en el dataset.
- Metadata de sexo para cada registro, permitiendo clasificación de sexo.
- Múltiples vistas (rear y side) casi equilibradas.
- Anotaciones COCO cargadas que facilitan el uso de pipelines existentes.
- Presencia de keypoints válidos en 4,748 registros.
- Un tamaño moderado de 4,910 imágenes que permite análisis exploratorio y experimentación inicial.
- Posibilidad de estudiar representaciones visuales basadas en vista, pose y medidas de peso.

## 7. Dataset Limitations

Las limitaciones demostradas por la auditoría son:
- 161 imágenes sin anotación, lo que reduce el efectivo usable para tareas supervisadas.
- Cobertura parcial de bounding boxes: sólo 865 BBoxes válidos frente a 3,884 imágenes sin BBox.
- Ausencia completa de segmentaciones COCO válidas en todas las imágenes.
- Cardinalidad variable de keypoints (4, 6, 9 y 23), lo que complica el uso uniforme de anotaciones de pose.
- Distribución desbalanceada por sexo, con un predominio de hembras sobre machos.
- La documentación de vistas B2/B4 sugiere estructuras de anotación heterogéneas que requieren tratamiento específico.

## 8. Impact of Dataset Limitations

- Dataset split: las particiones de entrenamiento, validación y prueba deberán construirse respetando el desbalance de sexo y la heterogeneidad de anotaciones. Se recomienda un muestreo estratificado por sexo y vista, y posiblemente crear subconjuntos separados según la presencia de BBoxes.
- Entrenamiento: los modelos de detección y segmentación no pueden aprovechar todo el dataset sin anotaciones adicionales. El entrenamiento de estimación de peso puede utilizar más imágenes, pero requiere control de calidad adicional para evitar ruido de vista y pose.
- Detection: la baja cobertura de BBoxes limita la capacidad de entrenar detectores a partir de todo el conjunto; se puede utilizar el subconjunto de 865 BBoxes válidos para entrenar detectores iniciales y el resto sólo para tareas no localizadas.
- Sex classification: el desbalance de 4,252 hembras y 658 machos puede sesgar un clasificador si no se corrige con técnicas de balanceo o ponderación.
- Weight estimation: la ausencia de segmentación y la falta de BBoxes en la mayoría de las imágenes implica que los modelos deben basarse en características globales de la imagen o en keypoints, y no en recortes segmentados exclusivos.
- Clustering: la cardinalidad variable de keypoints y la distribución de vistas hacen que los análisis de agrupamiento requieran normalización previa y selección de características robustas a vistas múltiples.
- Data augmentation: las transformaciones geométricas y de color son posibles, pero deben aplicarse con cuidado al subconjunto que contiene keypoints y BBoxes válidos para evitar inconsistencia en las anotaciones.
- Comparación de modelos: las evaluaciones deben documentar explícitamente qué subconjunto se usa para cada tarea y qué limitaciones de anotación afectan cada experimento.

## 9. Final Justification

El dataset sigue siendo adecuado para la investigación en estimación de peso de ganado y tareas asociadas, siempre que se construyan subconjuntos específicos para cada objetivo y se documente claramente cada limitación. Su valor principal reside en la disponibilidad de etiquetas de peso emparejadas con imágenes, metadata de sexo y múltiples vistas, mientras que sus limitaciones exigen un diseño experimental cuidadoso y una validación robusta de los subconjuntos utilizados.

## 10. References

Este documento se construyó a partir de los resultados internos de la auditoría COCOAudit y de las estadísticas del dataset generadas por DatasetStatistics. No se incluyen referencias externas en esta versión técnica porque no se utilizaron fuentes verificables adicionales.