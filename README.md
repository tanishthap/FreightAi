## Setup Instructions

1. Clone the repo

2. Install dependencies:
   pip install -r requirements.txt

3. Download dataset, file are too big to upload here:
   https://www.kaggle.com/datasets/usdot/freight-analysis-framework

4. Place files in data folder:
  /data/

5. Run the app:
   streamlit run app.py

# FreightAI — Logistics Decision-Support Dashboard

A polished, ML-powered prototype for supply chain professionals to predict shipment
transit time and risk level before execution, built with Streamlit, GeoPandas, and scikit-learn.

---

## Project structure

```
freightai/
├── app.py               ← Main Streamlit dashboard
├── requirements.txt     ← Python dependencies
├── README.md
└── data/
    ├── FAF4_4_State.csv                         ← FAF4 freight dataset
    ├── CFS_AREA_shapefile_010215.shp            ← Shapefile (geometry)
    ├── CFS_AREA_shapefile_010215.shx            ← Shapefile index
    ├── CFS_AREA_shapefile_010215.dbf            ← Shapefile attributes
    ├── CFS_AREA_shapefile_010215.prj            ← Projection info
    ├── CFS_AREA_shapefile_010215.cpg            ← Encoding info
    ├── CFS_AREA_shapefile_010215.sbn            ← Spatial index
    ├── CFS_AREA_shapefile_010215.sbx            ← Spatial index
    └── CFS_AREA_shapefile_010215.shp.xml        ← Metadata
```

> All shapefile component files must be in the **same `data/` folder** for GeoPandas to load them correctly.

---

## Setup

### 1. Clone / copy this folder
```bash
cd freightai
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # Mac / Linux
venv\Scripts\activate             # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note for Windows users:** GeoPandas on Windows sometimes requires extra steps.
> The easiest route is:
> ```bash
> conda install geopandas
> pip install streamlit plotly scikit-learn
> ```

### 4. Run the dashboard
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## How it works

### Data (FAF4)
The Freight Analysis Framework 4 (FAF4) dataset from the Bureau of Transportation Statistics
contains origin-destination-commodity-mode freight flows at the U.S. state level.
Key fields used:
- `dms_orig` / `dms_dest` — state FIPS codes
- `dms_mode` — transport mode (1=Truck, 2=Rail, 3=Water, 4=Air, …)
- `tons_2012` — freight weight in thousand tons
- `tmiles_2012` — ton-miles (weight × distance proxy)
- `value_2012` — shipment value in million dollars

### Machine Learning
Two models are trained at startup (cached after first run):

| Model | Type | Target |
|---|---|---|
| Transit Time | RandomForestRegressor | Hours derived from ton-miles / mode speed |
| Risk Level | RandomForestClassifier | Low / Medium / High from distance + mode + value |

Features: `origin FIPS`, `destination FIPS`, `mode code`, `approx. distance (miles)`, `value per ton`.

### Map
The CFS Area shapefile is loaded with GeoPandas and dissolved to state polygons.
A Plotly choropleth highlights the selected origin (blue) and destination (red),
with an animated arc between state centroids.

---

## Expanding this prototype

Future versions could add:
- Real-time weather and traffic API integration
- Historical delay database lookup
- Carrier performance scoring
- Cost optimization across mode combinations
- FAF4 Regional-level (132 zones) granularity using `FAF4_Regional.csv`
