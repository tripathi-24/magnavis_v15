# Script Doc: `src/equivalent_source_interpolation.py`

## Purpose

Research/demo script for equivalent-source interpolation and upward continuation on magnetic anomaly data.

## How to run

```bash
python src/equivalent_source_interpolation.py
```

## Inputs

- Example magnetic dataset loaded through geoscience libraries (ensaio/harmonica workflow).

## Outputs

- Fitted equivalent-source model.
- Interpolated grids and upward-continued fields.
- Diagnostic plots.

## Main functionality

1. Loads sample data.
2. Projects/crops region of interest.
3. Fits equivalent-source model.
4. Generates grid predictions and upward continuation.
5. Renders comparison plots.

## Caveats

- Not part of main Magnavis runtime app pipeline.
- Requires heavy geospatial/scientific dependencies.

## Example usage

Use to study how equivalent-source methods can reconstruct smoother magnetic surfaces from sparse observations.

