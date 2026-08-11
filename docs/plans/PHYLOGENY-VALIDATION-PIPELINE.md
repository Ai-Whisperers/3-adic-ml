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

---

## Estado actual (2026-07-17)

**Fases 1-4: completas y pusheadas a `origin/main`** (commits `966ffd1`,
`dd294ce`), todo verificado en CPU (esta máquina no tiene GPU con potencia
suficiente para entrenar):

- **Fase 1** (datos): 41 especies bajadas de UniProt vía Pfam PF00034 +
  taxonomía NCBI. 2 especies (E. coli, B. subtilis) excluidas de Fase 2 en
  adelante — sus únicos hits reviewed en UniProt son proteínas no-homólogas
  (peroxidasa/COX2), detectado y flageado por el propio script de fetch.
- **Fase 2** (alineamiento): 39 especies × 11 ventanas = 429 ventanas
  alineadas contra la referencia humana. **Nota importante**: 62% de los
  índices ternarios colisionan entre especies (encoding de hidropatía de
  solo 3 símbolos). El split por fila de ventana (fuga de especies) se
  **arregló el 2026-08-11**: `DataAuditor.prepare_data` ahora acepta
  `group_map_path` y separa por especie entera — activado por defecto en
  los presets de citocromo (B/C vía `data.group_map_path` en YAML, A vía
  `--group-map-path`). Con seed=42, las 3 condiciones separan las mismas
  6 especies de validación. Las colisiones de índice entre especies siguen
  existiendo (43/429 filas de val comparten contenido codificado con train
  bajo otra especie) — el auditor las reporta como warning, son inherentes
  al encoding, no al split.
- **Fase 3** (entrenamiento): las 3 condiciones corren limpio en CPU a
  escala smoke-test (60 épocas). Se encontró y arregló un colapso de
  posterior real en la Condición A (`train_euclidean_baseline.py`):
  faltaba `free_bits` (igual que B/C ya usan) — sin eso, val_acc quedaba en
  0% indefinidamente. Ya arreglado y verificado (val_acc subió a ~80% en
  el smoke test).
- **Fase 4** (evaluación): `scripts/analysis/evaluate_phylogeny_recovery.py`
  escrito, revisado a fondo (8 ángulos de búsqueda + verificación),
  10 hallazgos reales arreglados — incluye un guard contra p-valores
  espurios cuando el modelo colapsa, y un chequeo de colapso direccional de
  VAE-B que **ya detectó un colapso real** en el checkpoint smoke-test de
  la Condición C (`mean_pairwise_cosine_similarity=0.9999`).

**Bloqueado, necesita GPU:** los full runs (varios cientos de épocas por
condición, como V21-V24) nunca corrieron — todo lo de arriba fue a escala
smoke-test (60 épocas), que ni siquiera alcanza para que el currículum de
la Condición C active sus losses p-ádicas (`warmup_epochs: 100-200`). Los
números de correlación/Mantel obtenidos hasta ahora **no son el resultado
real**, solo prueban que el pipeline corre de punta a punta sin errores.

**Próximo paso exacto** (retomar en la máquina con GPU):
```bash
git pull
python3 scripts/applications/train_euclidean_baseline.py --epochs 500 \
  --checkpoint-out runs/cytochrome_c_A_euclidean/full.pt
python3 src/train.py --config src/presets/cytochrome_c_B_hyperbolic_generic.yaml   # subir epochs en el YAML primero
python3 src/train.py --config src/presets/cytochrome_c_C_padic.yaml               # subir epochs en el YAML primero
python3 scripts/analysis/evaluate_phylogeny_recovery.py \
  --run-a-checkpoint runs/cytochrome_c_A_euclidean/full.pt \
  --run-b-dir runs/cytochrome_c_B_hyperbolic_generic_<timestamp> \
  --run-c-dir runs/cytochrome_c_C_padic_<timestamp>
```
Después: Fase 5 (documentar el resultado real, positivo o negativo, en
`docs/plans/EXTERNAL-VALIDATION-ROADMAP.md`).

## Actualización 2026-07-17: full runs ejecutados, Fase 5 completa

La nota "bloqueado, necesita GPU" de arriba estaba desactualizada — esta
máquina sí tiene una RTX 3050 (la misma que documenta CLAUDE.md como target
de hardware), simplemente no se había chequeado `torch.cuda.is_available()`
antes. Se subieron `training.epochs` a valores de full run (A=500, B=500,
C=1000 — con `warmup_epochs`/`phase_start_epoch` del currículum de C
finalmente con margen para activar) y se corrieron las 3 condiciones + Fase
4 de punta a punta en background (~20 min total en GPU).

También se agregó `raw_encoding_baseline` (Condición 0, sin modelo) a
`evaluate_phylogeny_recovery.py` — distancia Euclidiana directa sobre las
secuencias alineadas codificadas por hidropatía, sin VAE — después de que
revisar la fuga de datos train/val de Fase 3 destapara algo más importante:
esa señal trivial, sola, ya correlaciona ~0.72 con la taxonomía real. Sin
ese control, un resultado positivo de A/B/C sería imposible de interpretar.

**Resultado real (no smoke test):** ninguna de las 3 condiciones supera el
baseline sin modelo (raw=0.7228 vs. A=0.6285, B=0.6538, C=0.4955 — C queda
último). Detalle completo, metodología y qué significa en
`docs/plans/EXTERNAL-VALIDATION-ROADMAP.md` (sección "Result (2026-07-17):
cytochrome c phylogeny") — ese documento es la Fase 5 (reporte honesto) que
pedía este plan.

**Checkpoints + config + resultados completos, públicos:**
https://huggingface.co/geestaltt/3-adic-vae-cytochrome-c

**Hallazgo 2026-07-17 (revisión de la fuga de datos), ya corregido:** al
investigar la fuga de fila train/val de Fase 3 (confirmada real pero
confinada a las métricas de monitoreo de Fase 3 — `evaluate_phylogeny_recovery.py`
usa `indices.pt` completo, sin split, así que la Fase 4 no está contaminada
por eso) apareció algo más importante: la colisión de índices entre especies
(73.9% de las 429 ventanas, ya reportada por `index_collision_report`)
correlaciona **por sí sola, sin ningún modelo entrenado**, con la taxonomía
real: `Spearman(similitud trivial por codificación, distancia taxonómica) =
0.72` (n=741 pares, p≈1e-117 — motivo obvio: secuencia conservada ↔
pariente cercano, es la premisa de la filogenia molecular). Esto pone un
piso alto: cualquier resultado de A/B/C que no supere ese ~0.72 no
demuestra que la arquitectura aprendió nada, solo que codificación +
distancia ya recuperan filogenia trivialmente.

Agregado `raw_encoding_baseline` (Condición 0) a
`evaluate_phylogeny_recovery.py`: distancia Euclidiana directa sobre las
secuencias alineadas codificadas por hidropatía, sin VAE, corre siempre
(no necesita ningún checkpoint) y ahora el script imprime un veredicto
explícito por condición ("BEATS baseline" / "does NOT beat baseline").
Verificado end-to-end: con los checkpoints smoke-test de 60 épocas (Fase 3),
ninguna de las 3 condiciones supera el baseline todavía (A=0.649, B=0.625,
C=0.451 vs baseline=0.723) — resultado esperado dado que ninguna llegó a
converger; sirve como prueba de que el veredicto funciona antes de gastar
cómputo GPU en los full runs.

## Actualización 2026-08-11: fuga de fila train/val arreglada + val_loss consistente

Los dos pendientes documentados arriba quedaron arreglados (en la máquina
CPU, en paralelo a los full runs de la máquina GPU — los full runs de
2026-07-17 se entrenaron todavía con el split por fila; recordar que esa
fuga estaba confinada a las métricas de monitoreo de Fase 3, la evaluación
de Fase 4 usa `indices.pt` completo sin split):

- **Split por especie en entrenamiento:** `DataAuditor.prepare_data` ahora
  acepta `group_map_path`/`group_key` y separa por especie entera en vez
  de por fila. Activado por defecto en las 3 condiciones (B/C vía
  `data.group_map_path` en YAML, A vía `--group-map-path`). Con seed=42,
  las 3 condiciones separan las mismas 6 especies de validación. Las
  colisiones de índice entre especies siguen existiendo (43/66 filas de
  val comparten contenido codificado con train bajo otra especie) — el
  auditor las reporta como warning, son inherentes al encoding. Tests en
  `tests/test_data_auditor_group_split.py`. Complementario al holdout de
  9 especies de `select_holdout_species.py` (Condición D): aquel saca
  especies del dataset entero; esto evita fuga en el split train/val
  interno de cualquier run. Ojo si se re-entrena la Condición C: su
  currículum activa las losses stage-2 con `coverage_threshold: 0.80`
  medido sobre validación — con especies nunca vistas en val ese umbral
  puede tardar más épocas en alcanzarse.
- **`evaluate()` en `train_euclidean_baseline.py`** ahora aplica
  `kl_weight` igual que la loss de entrenamiento
  (`val_loss = recon + kl_weight * kl`); el número impreso es comparable
  a `train_loss`. Detectado 2026-07-17.
