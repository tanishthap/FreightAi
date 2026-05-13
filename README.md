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





# Dashboard Walkthrough (Screenshots)

## 1. Landing Overview
The FreightAI landing section introduces the three core workflows of the platform:

- **Predict** — generate shipment transit-time and risk predictions
- **History** — store and review previous shipment analyses
- **Batch Upload** — process multiple shipments at once using CSV uploads

This section communicates the platform’s goal of combining machine learning with an AI-assisted logistics decision layer.
<img width="1358" height="643" alt="Screenshot 2026-05-13 at 12 22 01 PM" src="https://github.com/user-attachments/assets/03406505-e275-4225-aa78-ebb0798b1e69" />

---
## . Historical Predictions
The History tab stores previously generated shipment predictions for future review.

Users can:
- Revisit prior shipment analyses
- Compare historical routes
- Track prediction consistency
- Export records for reporting

This creates a lightweight shipment decision archive directly inside the dashboard.

---
## 2. Shipment Prediction Interface
The main dashboard allows users to configure a shipment profile by selecting:

- Origin state
- Destination state
- Transportation mode
- Commodity type
- Trade type

The interface then runs two ML models:

1. Transit-time prediction
2. Shipment risk classification

An AI Intelligence Layer can also be enabled to generate operational insights and recommendations.

<img width="1286" height="714" alt="Screenshot 2026-05-13 at 12 22 15 PM" src="https://github.com/user-attachments/assets/815d9706-ed06-4a94-bb77-73f6d29eb2ff" />
---

## 3. Destination State Selection
This screenshot demonstrates the searchable state dropdown menu used to select shipment destinations.

Example:
- Origin = California
- Destination = New York

The app supports all U.S. states and dynamically updates predictions based on route distance and logistics conditions.
<img width="1299" height="704" alt="Screenshot 2026-05-13 at 12 22 26 PM" src="https://github.com/user-attachments/assets/52203bea-894f-49fc-8efd-42c06c6ff913" />

---
## 4. Transportation Mode Selection
Users can compare different shipping methods, including:

- Truck
- Rail
- Water
- Air
- Pipeline
- Multiple Modes

Different transportation modes produce significantly different:
- Transit times
- Cost estimates
- Risk classifications

For example:
- Air freight typically lowers transit time
- Rail and water often reduce cost
- Truck routes may increase congestion-related risk
<img width="277" height="360" alt="Screenshot 2026-05-13 at 12 22 32 PM" src="https://github.com/user-attachments/assets/5f84999e-ffe8-4b33-b557-27a378de1492" />

---

## 5. Commodity Selection
The commodity selector allows users to classify shipment type, including:

- Electronics
- Machinery
- Pharmaceuticals
- Fuel oils
- Mixed freight
- Agricultural goods

Commodity type affects:
- Estimated shipping cost
- Risk level
- Operational recommendations

For example, temperature-sensitive or hazardous commodities may increase operational risk.
<img width="1269" height="714" alt="Screenshot 2026-05-13 at 12 22 41 PM" src="https://github.com/user-attachments/assets/29156147-f0d9-4635-9413-f8a990a14ff0" />

---

## 6. California → New York Prediction (Initial Results)
This example shows a truck shipment from California to New York.

FreightAI predicts:
- ~69.8 transit hours (~2.9 days)
- High shipment risk
- Long-distance freight corridor exposure

The map visualization displays:
- Shipment origin
- Destination
- Animated route arc between states

The dashboard also estimates shipping cost ranges and highlights alternative transportation modes.
<img width="1000" height="717" alt="Screenshot 2026-05-13 at 12 22 54 PM" src="https://github.com/user-attachments/assets/068365d9-8638-4d25-a54d-3b0752031c52" />

---

## 7. Cost & Risk Analysis
This section expands the prediction results with:
- Estimated shipping cost
- Cost per mile
- Shipment risk confidence score
- Alternative mode comparisons

In this scenario:
- Rail and water transportation offer lower estimated costs
- Truck transport remains higher risk due to long-haul exposure and corridor congestion

The dashboard visually compares potential savings by transportation mode.
<img width="687" height="718" alt="Screenshot 2026-05-13 at 12 23 08 PM" src="https://github.com/user-attachments/assets/6307f1fe-2724-4b7c-81b7-76c875f0f29b" />


## 8. AI Intelligence Layer Explanation
The AI Intelligence Layer generates operational logistics commentary based on:
- Weather patterns
- Freight corridor congestion
- Commodity sensitivity
- Historical transportation patterns

For the California → New York shipment:
- The AI warns about possible Rocky Mountain weather delays
- Recommends dynamic routing adjustments
- Suggests intermodal transportation alternatives

This combines ML predictions with human-readable logistics recommendations.

---
## 9. Rule-Based Risk Explanation
The dashboard provides explainable AI outputs describing why a shipment receives a certain risk level.

Key risk drivers identified:
- Long shipping distance
- High-volume freight corridor
- Long-haul trucking constraints

The recommendation engine suggests:
- Adding delivery-time buffers
- Monitoring delays closely
- Considering lower-risk transportation alternatives

This improves transparency and decision support for logistics planners.




<img width="1111" height="716" alt="Screenshot 2026-05-13 at 12 23 24 PM" src="https://github.com/user-attachments/assets/3f304110-4d17-4be3-846c-d840a2e73ffc" />
---


## 11. Hawaii → New York Shipment Scenario
This scenario intentionally tests an unrealistic logistics configuration:

- Origin = Hawaii
- Destination = New York
- Mode = Truck

FreightAI predicts:
- Extremely long transit time
- Very high shipping cost
- High operational risk

The route visualization immediately highlights the impracticality of the shipment.

<img width="668" height="746" alt="Screenshot 2026-05-13 at 12 23 45 PM" src="https://github.com/user-attachments/assets/d14cf595-4840-4394-b664-80c6bae9546d" />

---

## 12. AI Governance & Validation Check
The AI Governance Layer detects logical transportation issues.

In this case:
- Truck transportation from Hawaii to New York is physically impossible without ocean transit

The system automatically flags:
- Invalid routing assumptions
- Unrealistic shipment planning
- Elevated logistics risk

This demonstrates FreightAI’s ability to combine machine learning with rule-based operational validation.

<img width="669" height="745" alt="Screenshot 2026-05-13 at 12 24 03 PM" src="https://github.com/user-attachments/assets/f94cc843-7861-4ba1-8fa2-17edfbef0dd5" />

---

## 13. Batch Upload Workflow
The Batch Upload tab allows users to upload CSV files containing multiple shipment records.

Required CSV fields include:
- Origin
- Destination
- Transportation mode
- Commodity
- Trade type

This workflow supports large-scale shipment analysis for logistics teams and supply-chain planners.

<img width="1278" height="671" alt="Screenshot 2026-05-13 at 12 24 12 PM" src="https://github.com/user-attachments/assets/3b63938f-b6f0-44ab-9bfe-2ef74ee0cf8a" />

---


## 14. Batch Prediction Results
After uploading a CSV file, FreightAI processes all shipment rows simultaneously.

The dashboard displays:
- Total processed shipments
- Successful vs failed predictions
- Transit times
- Estimated days
- Risk levels for each shipment

Color-coded risk labels make it easy to identify:
- High-risk routes
- Medium-risk shipments
- Lower-risk transportation plans

Results can also be exported as CSV for reporting and operational planning.

<img width="936" height="742" alt="Screenshot 2026-05-13 at 12 24 24 PM" src="https://github.com/user-attachments/assets/f2293e25-a2bf-422b-bbdc-217ebe2e8b25" />

---

---

# Conclusion

FreightAI demonstrates how machine learning and explainable AI can support smarter logistics planning before shipments are executed. By combining predictive modeling, route visualization, operational risk analysis, and AI-generated recommendations, the platform helps users evaluate shipment feasibility, cost efficiency, and transportation risk in real time.

The dashboard highlights how different transportation modes, commodity types, and route distances impact delivery performance and operational reliability. Features such as batch processing, historical prediction tracking, and AI governance checks further extend the platform into a practical decision-support tool for supply chain professionals.

Although this project is a prototype, it establishes a strong foundation for future logistics intelligence systems that could integrate:
- Real-time weather and traffic APIs
- Carrier performance analytics
- Dynamic routing optimization
- Live supply-chain monitoring
- Regional freight forecasting

FreightAI ultimately showcases how AI-powered decision systems can improve visibility, reduce operational uncertainty, and support more informed transportation planning across modern supply chains.

---
