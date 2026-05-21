# Visualizando el Espacio Latente 3-Ádico 🎨

Uno de los aspectos más potentes de este proyecto es la capacidad de **ver** las matemáticas. Debido a que utilizamos geometría hiperbólica, nuestros "mapas" de los datos se ven y se comportan de forma diferente a los mapas euclidianos estándar.

## 🛰️ El Pipeline de Visualización

Durante el entrenamiento, el `VisualizationPipeline` genera automáticamente varios tipos de mapas. Estos se guardan como archivos HTML interactivos en `runs/visualizations/<nombre_del_experimento>/`.

### 1. Disco de Poincaré Nativo
Este es nuestro "estándar de oro" en visualización.
- **Qué muestra**: El espacio latente proyectado directamente en un disco 2D.
- **Preservación Radial**: La distancia desde el centro representa la **valuación 3-ádica**. Los puntos en el centro son los más "fundamentales" (divisibles por muchas potencias de 3), mientras que los puntos en el borde son los más "específicos".
- **Dirección**: El ángulo representa la **estructura de prefijos** (los primeros dígitos de la operación ternaria).
- **Renders Recientes**:
    - **[Disco de Poincaré V7.2 Large](v7_large_poincare.html)**: El baseline de gran escala recomendado.
    - **[Disco de Poincaré V14.1 Ring Completion](v14_ring_completion_poincare.html)**: Modelo avanzado que explora la completitud de anillos y una alta consistencia algebraica.
- **Características**:
    - **Bordes del Árbol**: Líneas finas que conectan "padres" con "hijos" en el árbol 3-ádico.
    - **Sombreado de Prefijos**: Regiones coloreadas que muestran dónde viven las diferentes clases de prefijos.

### 2. UMAP y PaCMAP Hiperbólicos
Los algoritmos tradicionales como UMAP suelen utilizar la distancia euclidiana. Nosotros los hemos modificado para utilizar la **Matriz de Distancia Hiperbólica**.
- **UMAP (Uniform Manifold Approximation and Projection)**: Excelente para ver el "esqueleto" general de los datos.
- **PaCMAP**: Mejor para equilibrar los clusters locales con la estructura global del árbol.
- **Por qué es importante**: Al usar distancias hiperbólicas, nos aseguramos de que la naturaleza "curva" del espacio se preserve incluso cuando lo aplanamos a 2D o 3D.

### 3. Homología Persistente
Utilizamos la topología para comprobar si la IA está aprendiendo "agujeros" o "componentes conectados" en los datos.
- **Números de Betti**: Rastreamos $H_0$ (componentes conectados). Idealmente, a medida que la IA aprende la estructura del árbol, el número de componentes conectados debería alinearse con la ramificación 3-ádica.

## 🛠️ Cómo Generar Visualizaciones

### Durante el Entrenamiento
Las visualizaciones están activadas por defecto en las configuraciones YAML:
```yaml
visualization:
  max_per_level: 500     # Tamaño de la submuestra por nivel de valuación
  persist_every: 50      # Generar nuevos archivos cada 50 épocas
  save_html: true
```

### Generación Manual
Puedes ejecutar los scripts de análisis para generar visualizaciones específicas:
- **`scripts/analysis/create_evolution_animation.py`**: Crea un vídeo que muestra cómo el espacio latente se "despliega" durante el entrenamiento.
- **`scripts/analysis/visualize_algebraic_trajectories.py`**: Muestra cómo se mueve el espacio latente cuando realizas operaciones algebraicas como $n \to n+1$ o $n \to 3n$.

## 🕵️ Cómo "Leer" un Mapa de Poincaré

Cuando abras una visualización HTML interactiva:

1.  **El Centro es la Raíz**: El punto único en el centro mismo (normalmente $v=9$) es el índice "0", la raíz del árbol 3-ádico.
2.  **Los Anillos son Valuaciones**: Verás anillos concéntricos de puntos. Cada anillo corresponde a una valuación 3-ádica diferente.
3.  **Las Cuñas son Prefijos**: Los puntos en la misma "rebanada de pastel" suelen compartir los mismos dígitos iniciales.
4.  **Zoom**: En la bola de Poincaré, la "acción" ocurre cerca del borde. Haz zoom en los bordes para ver la ramificación detallada de los niveles $v=0$ y $v=1$.

---

*"Una imagen vale más que mil valuaciones."*
