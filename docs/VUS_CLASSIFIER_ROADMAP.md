# Roadmap: Clasificador de Variantes de Importancia Desconocida (VUS) 🧬

**Objetivo**: Transformar el detector de anomalías V16.0 en una herramienta de triaje clínico para variantes genéticas humanas (TP53 inicialmente).

## 1. Definición del Problema
Muchas variantes identificadas mediante secuenciación genómica son clasificadas como **VUS (Variants of Uncertain Significance)**. La medicina genómica requiere herramientas computacionales que trasciendan la simple correlación estadística para evaluar el impacto estructural real de una variante.

## 2. Fases del Proyecto

### Fase I: Adquisición y Procesamiento (Data Engineering)
- [ ] Obtener dataset de variantes de TP53 desde **ClinVar** (Etiquetas: *Pathogenic, Benign, VUS*).
- [ ] Implementar parser para transformar VCF/TSV a secuencias FASTA (ventana de 9 nucleótidos centrada en la mutación).
- [ ] Normalización y tokenización hacia índices ternarios compatibles con V16.0.

### Fase II: Evaluación y Calibración (Validation)
- [ ] Proyectar el dataset clínico en el espacio latente del modelo V16.0 (Fine-tuned en TP53).
- [ ] Calcular umbrales óptimos mediante Curvas ROC y F1-Score para maximizar la detección de variantes *Pathogenic*.
- [ ] Evaluar la capacidad del modelo para separar variantes *Benign* de *Pathogenic*.

### Fase III: Clasificador VUS (Application)
- [ ] Desarrollo de una interfaz de consulta: `Input: Secuencia variante -> Output: Score de Probabilidad de Anomalía`.
- [ ] Clasificación de secuencias VUS desconocidas basada en su proximidad al *manifold* patogénico o benigno.
- [ ] Reporte final: Interpretación geométrica de por qué una variante es clasificada como patogénica (ej. "distorsión en nivel v=0").

## 3. Entregables Esperados
1.  Pipeline automatizado de ClinVar a Embedding.
2.  Reporte de rendimiento: Precisión, Sensibilidad (TPR) y Especificidad (TNR) en variantes clínicas reales.
3.  Módulo funcional `VUSClassifier` para despliegue.
