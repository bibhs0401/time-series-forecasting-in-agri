# time-series-forecasting-in-agri

## Setup

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Current Pipeline

The raw Google Trends downloads live in `rawdata/g1` through `rawdata/g12`.
Generated outputs are written under `outputs/`.

Run the stages in this order:

```powershell
python -m src.data.stitch_panel
python -m src.data.validate_panel
python -m src.data.validate_stitch
python -m src.features.weather_features
python -m src.features.build_tensor
python -m src.graph.adjacency_static
```

For modeling, build feature groups and graph layers inside each rolling-origin
training fold so scalers and data-driven graph edges are fit only on training
data.
