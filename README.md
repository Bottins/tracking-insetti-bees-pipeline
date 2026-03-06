# Tracking Insetti (Bees Pipeline)

Pipeline video per tracking insetti/api con preprocessing, tracking multi-oggetto (Kalman), heatmap e analisi comportamentale.

## Struttura attesa

```text
Tracking Insetti/
  bees_pipeline.py
  requirements.txt
  data/
    raw_videos/
  output/
```

## Naming video

Formato atteso:

- `N-0,001(1).MOV`
- `N-0,01(2).MOV`
- `O-0,1(3).MOV`

Dove:

- `N`/`O` = tesi/gruppo
- concentrazione con virgola (es. `0,001`, `0,01`, `0,1`)
- replica tra parentesi

## Config path (portabile)

Il codice ora usa path relativi al progetto:

- input default: `data/raw_videos`
- output default: `output`

Override opzionale via variabili ambiente:

- `BEES_INPUT_DIR`
- `BEES_OUTPUT_DIR`

## Setup

```bash
pip install -r requirements.txt
```

## Esecuzione

```bash
python bees_pipeline.py
```

## Output generato

- `output/processed/` video pre/post processati
- `output/csv/` traiettorie
- `output/heatmaps/` mappe densita
- `output/analisi/` metriche comportamentali
- `output/tracking_master.csv`

## Nota privacy e dimensione repo

Video originali e output pesanti sono stati rimossi dalla repo e sostituiti da struttura/template.
## Research Profile

- Research keywords: multi-object tracking, Kalman filtering, behavioral quantification, video analytics, heatmaps.
- Positioning: computer-vision research project for experimental behavioral analysis.
- Open-source status: this repository is open source and intended for reproducible research and education.

