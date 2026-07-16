# Pipeline de filogenia real: Cytochrome C + comparación de baselines

## Contexto

`docs/plans/EXTERNAL-VALIDATION-ROADMAP.md` deja documentado un problema de
fondo: todas las métricas de "jerarquía" reportadas hasta ahora (Spearman
0.8335, ARI=1.0) miden si las losses hacen lo que fueron escritas para hacer
— `v_3(índice)` se calcula del índice y se inyecta directamente como target,
así que el resultado es casi tautológico. El roadmap identifica dos cosas que
faltan para que esto sea una demostración real en vez de un ejercicio
autoconsistente:

1. Mover a un dominio real cuya jerarquía sea conocida **de forma
   independiente**, no inyectada en la loss.
2. Comparar contra baselines (Euclidiano, hiperbólico genérico) en la misma
   tarea — comparación que hoy no existe en ningún lado del repo.

Elegimos citocromo C como dominio porque: es el dataset clásico de reloj
molecular (Fitch & Margoliash), es corto (~104-110 aa, pocas ventanas por
especie), tiene cobertura real desde bacterias hasta humanos en
UniProt/NCBI, y reutiliza el encoding de hidropatía que ya existe en
`scripts/data/peptide_encoding.py` (ya usado por 4 scripts). El diseño de
experimento acordado son **3 condiciones** (Euclidiano / Hiperbólico
genérico / P-ádico actual) evaluadas post-hoc contra distancia taxonómica
real — sin tocar `src/losses/` para esta primera fase (la variante D,
"reapuntar" `target_radius` a la taxonomía real vía el `valuation_type`
plegable que ya existe en `src/losses/combined.py:120`, queda deliberadamente
fuera de alcance hasta tener resultados de las 3 primeras).

**Objetivo final medible:** ¿la distancia hiperbólica/p-ádica entre especies
correlaciona con su distancia taxonómica real mejor de lo que lo hace un VAE
euclidiano plano, en secuencias nunca usadas para fijar el target de ninguna
loss?

---

## Fase 1 — Datos reales con procedencia auditable

**Problema a evitar:** el roadmap ya documenta que `validate_humsavar_tp53.py`
usa una secuencia "tipeada a mano en un comentario". Esta vez todo dato debe
venir de una fuente descargable y verificable.

- **Nuevo:** `scripts/data/fetch_cytochrome_c.py` — descarga secuencias reales
  de citocromo C vía UniProt REST API para una lista curada de ~30-40
  especies que cubran un rango taxonómico amplio (bacterias, hongos,
  plantas, invertebrados, vertebrados incluyendo humano). Guarda FASTA crudo
  + manifest committeable (especie, accession, fecha de descarga, URL) en
  `data/cytochrome_c/manifest.csv` — el manifest se commitea aunque
  `data/` esté gitignoreado para las secuencias, así queda auditable sin
  necesidad de volver a bajar nada para revisar procedencia.
- **Nuevo:** `scripts/data/fetch_taxonomy.py` — para la misma lista de
  especies, obtiene el linaje taxonómico completo (dominio→especie) vía la
  API de NCBI Taxonomy. Calcula una matriz de distancia par-a-par por
  "profundidad del último ancestro común en rangos Linneanos" (ej. mismo
  género = distancia 1, mismo dominio nada más = distancia 7) — es una
  ultramétrica genuina, calculable con comparación de strings de linaje, sin
  dependencias nuevas. Guarda `data/cytochrome_c/taxonomic_distance.npy` +
  `taxonomy_lineage.json`.

**Riesgo flageado:** ambos scripts requieren acceso a red en tiempo de
ejecución (UniProt/NCBI). No se puede validar esto en modo plan; se
verificará al ejecutar.

---

## Fase 2 — Alineamiento y encoding a ventanas de 9 residuos

Las 96+ ventanas de 9 residuos por especie que salen de un tiling naive no
son comparables entre especies (posición i en la especie X no es
homóloga a posición i en la especie Y sin alinear). Se necesita alinear
antes de definir las ventanas.

- **Nuevo:** `scripts/data/align_cytochrome_c.py` — alinea cada secuencia
  contra una referencia (humano) con `Bio.Align.PairwiseAligner`
  (Biopython — confirmar que está en requirements, si no, agregarlo).
  Define los límites de ventana sobre las coordenadas de la referencia
  alineada, así "ventana k" es la misma posición estructural para todas las
  especies.
- **Reutiliza sin cambios:** `scripts/data/peptide_encoding.py`
  (`encode_peptide_window`, `AA_MAP`) para mapear cada ventana alineada a un
  vector de 9 dígitos ternarios {-1,0,1} — el mismo encoding que ya usan
  `scan_human_proteome.py`, `qspr_bioactivity_scoring.py`, etc.
- **Nuevo:** `scripts/data/prepare_cytochrome_c_dataset.py` — junta todo:
  por especie, produce ~10-12 ventanas alineadas, las convierte a índices
  ternarios (0-19682) vía `TERNARY.from_ternary` (mismo patrón que
  `seq_to_ternary_index` en `prepare_codon_data.py`), y guarda
  `data/cytochrome_c/indices.pt` + un mapa `window_id → (species, window_idx)`
  para poder reagregar embeddings por especie después del entrenamiento.
  El formato de salida (`torch.save(tensor(indices), ...)`) sigue el mismo
  patrón que `prep_human_tp53.py`.

---

## Fase 3 — Tres condiciones de entrenamiento

Las 3 condiciones entrenan sobre el mismo `data/cytochrome_c/indices.pt`
(vía `data.indices_path`, mecanismo que **ya existe y ya soporta datasets de
tamaño arbitrario** — confirmado en `src/training/bootstrap.py:92-99`,
`DataAuditor.prepare_data` no asume `N=19683` en ningún punto).

### Condición A — Euclidiano plano (código nuevo, aislado)
El motor de entrenamiento actual (`src/train.py`) trae consigo maquinaria
específica de la curricula p-ádica (StateNet, Lagrangian dual ascent, LR
controller, grokking detector) que no aplica a un VAE plano — forzarla
sería más trabajo que escribir el loop directo.

- **Nuevo:** `src/models/vae_baseline.py` — `TernaryVAEEuclideanBaseline`:
  reutiliza `EncoderHead`/decoder de `src/models/vae.py` tal cual, pero
  se salta `DualHyperbolicProjection` por completo — reparametrización
  N(0,I) estándar, decodifica directo del espacio tangente.
- **Nuevo:** `scripts/applications/train_euclidean_baseline.py` — loop de
  entrenamiento simple (reconstrucción cross-entropy + KL gaussiano
  estándar, Adam, sin curricula) sobre `data/cytochrome_c/indices.pt`.

### Condición B — Hiperbólico genérico (sin código nuevo)
Arquitectura actual (`TernaryVAEV6Controllable`) + preset YAML que apaga
**todas** las losses p-ádicas (`radial`, `monotonic`, `rich_hierarchy.
hierarchy_weight=0`, `angular_coherence.enabled=false`,
`algebraic_*.enabled=false`, `valuation_prior.enabled=false`,
`geodesic`/`rank` desactivadas) dejando solo `rich_hierarchy.coverage_weight`
(reconstrucción) + `hyperbolic_kl` activos. Esto es 100% config-driven —
el patrón de habilitar/deshabilitar por loss ya existe en cada bloque YAML
(ver `surrogate_property.enabled: false` en `v24.0_tangent_fix.yaml`).
- **Nuevo:** `src/presets/cytochrome_c_B_hyperbolic_generic.yaml`

### Condición C — P-ádico actual, sin cambios de arquitectura
Mismo `train.py` + mismo `TernaryVAEV6Controllable`, config casi idéntica a
`v24.0_tangent_fix.yaml` pero apuntando `data.indices_path` al dataset de
citocromo C. Esta es la prueba de "transferencia pura": el índice ternario
de cada ventana de citocromo C no tiene relación causal con la especie de
origen, así que si esta condición correlaciona con taxonomía real sería
sorprendente (y hay que decirlo así en el resultado, sin sobre-vender).
- **Nuevo:** `src/presets/cytochrome_c_C_padic.yaml`

**Nota de tamaño de dataset:** ~30-40 especies × ~10-12 ventanas ≈ 300-480
muestras. Verificar en la ejecución si el batch size / hiperparámetros por
defecto de `v24.0_tangent_fix.yaml` siguen siendo razonables a esta escala
(probablemente sí, pero con menos pasos por época).

---

## Fase 4 — Evaluación: ¿recuperan filogenia real?

- **Nuevo:** `scripts/analysis/evaluate_phylogeny_recovery.py`:
  1. Para cada checkpoint (A/B/C), calcular embeddings de todas las
     ventanas (`model.get_mu_representations` / `get_hyperbolic_representations`,
     igual que `probe_foreign_genome.py`).
  2. Agregar las ~10-12 ventanas por especie a **un punto por especie**
     (media en espacio tangente vía `logmap0`/`expmap0`, ya usados en todo
     el codebase, para B/C; media aritmética simple para A).
  3. Calcular matriz de distancia par-a-par entre especies (Poincaré para
     B/C, Euclidiana para A) y correlacionarla (Spearman) contra
     `taxonomic_distance.npy` de la Fase 1.
  4. **Test de Mantel** (permutación, no Spearman naive) para el p-valor —
     los pares de una matriz de distancia no son independientes, así que
     un Spearman ingenuo infla la significancia. Reutilizar el patrón
     auto-escéptico de `scripts/validation/check_zero_count_semantics.py`
     (hipótesis nula explícita + baseline de permutación aleatoria).
  5. Bootstrap sobre especies para intervalo de confianza de la correlación
     de cada condición, y comparar A vs B vs C.
  6. Hold-out: separar ~20% de especies que nunca entren en el
     entrenamiento, reportar la correlación también solo sobre esas.
  Donde aplique, reutilizar `representation_probe_suite`/
  `retrieval_ablation_suite` de `scripts/analysis/project_audit.py`
  (`project_audit.py:341`, `:534`) en vez de reimplementar estadística —
  el roadmap ya señala esa máquina como la más reutilizable para esto.

---

## Fase 5 — Reporte honesto

Actualizar `docs/plans/EXTERNAL-VALIDATION-ROADMAP.md` con el resultado
(cualquiera sea) siguiendo el mismo formato que ya usa el documento para
`check_zero_count_semantics.py`: hipótesis, método, resultado numérico,
veredicto explícito. Si el resultado es negativo (ninguna condición
correlaciona con taxonomía real), documentarlo igual — es la clase de
resultado que el roadmap pide, no uno a evitar.

---

## Archivos nuevos (resumen)

| Archivo | Rol |
|---|---|
| `scripts/data/fetch_cytochrome_c.py` | Descarga secuencias reales + manifest |
| `scripts/data/fetch_taxonomy.py` | Linaje NCBI + matriz de distancia taxonómica |
| `scripts/data/align_cytochrome_c.py` | Alineamiento contra referencia humana |
| `scripts/data/prepare_cytochrome_c_dataset.py` | Ventanas alineadas → `indices.pt` |
| `src/models/vae_baseline.py` | `TernaryVAEEuclideanBaseline` (condición A) |
| `scripts/applications/train_euclidean_baseline.py` | Loop de entrenamiento condición A |
| `src/presets/cytochrome_c_B_hyperbolic_generic.yaml` | Config condición B |
| `src/presets/cytochrome_c_C_padic.yaml` | Config condición C |
| `scripts/analysis/evaluate_phylogeny_recovery.py` | Evaluación A/B/C + Mantel + bootstrap |

Reutilizados sin cambios: `scripts/data/peptide_encoding.py`,
`src/core/ternary.py` (`TERNARY.from_ternary`), `src/train.py`,
`src/models/vae.py` (`TernaryVAEV6Controllable`), patrones de
`scripts/analysis/probe_foreign_genome.py` y `project_audit.py`.

---

## Verificación

1. `fetch_cytochrome_c.py` / `fetch_taxonomy.py`: correr y confirmar que el
   manifest tiene N especies reales con accessions válidos (no placeholders).
2. `prepare_cytochrome_c_dataset.py`: confirmar shape de `indices.pt` y que
   el mapa especie↔ventana cuadra.
3. Entrenar las 3 condiciones (pocas épocas primero, smoke test, luego full
   run) y confirmar convergencia básica (reconstrucción no se estanca).
4. Correr `evaluate_phylogeny_recovery.py` y revisar que el test de Mantel
   produzca un p-valor y no solo una correlación cruda.
5. Confirmar que el resultado (positivo o negativo) queda escrito en el
   roadmap con el mismo nivel de rigor que las entradas existentes.
