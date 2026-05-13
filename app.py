"""
FreightAI - Flask Backend
Loads trained ML models and serves predictions via web UI.
Features: single prediction, prediction history, batch CSV, LLM explanation layer.
"""
import csv
import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ------------------------------------------------------------------
# Load models + artifacts once at startup
# ------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
MODELS_DIR = os.path.abspath(MODELS_DIR)
print(f"Loading models from: {MODELS_DIR}")

regressor  = joblib.load(f'{MODELS_DIR}/regressor.joblib')
classifier = joblib.load(f'{MODELS_DIR}/classifier.joblib')
with open(f'{MODELS_DIR}/feature_columns.json') as f:
    FEATURE_COLUMNS = json.load(f)

distance_lookup = pd.read_csv(f'{MODELS_DIR}/distance_lookup.csv')
volume_lookup   = pd.read_csv(f'{MODELS_DIR}/volume_lookup.csv')

with open(f'{MODELS_DIR}/metrics.json') as f:
    METRICS = json.load(f)

print(f"Loaded regressor  ({type(regressor).__name__})")
print(f"Loaded classifier ({type(classifier).__name__})")
print(f"Feature columns: {len(FEATURE_COLUMNS)}")

# ------------------------------------------------------------------
# In-memory prediction history
# ------------------------------------------------------------------
prediction_history = []

# ------------------------------------------------------------------
# LLM config — set ANTHROPIC_API_KEY or GEMINI_API_KEY env var
# ------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

def get_llm_narrative(rule_explanation, inputs, extra=None):
    """Returns {'text': str, 'governance': str} on success, {'error': str} on failure, or None if no key."""
    if ANTHROPIC_API_KEY:
        result = _call_claude(rule_explanation, inputs, extra)
    elif GEMINI_API_KEY:
        result = _call_gemini(rule_explanation, inputs, extra)
    else:
        return None
    # Parse governance section from response
    if result and 'text' in result:
        raw = result['text']
        if '===GOVERNANCE===' in raw:
            parts = raw.split('===GOVERNANCE===', 1)
            result['text'] = parts[0].strip()
            gov = parts[1].strip()
            # Only surface governance when LLM flags an issue
            if gov.upper().startswith('FLAG'):
                result['governance'] = gov[4:].strip().lstrip(':').strip()
            else:
                result['governance'] = None
        else:
            result['governance'] = None
    return result

def _build_llm_prompt(explanation, inputs, extra=None):
    extra = extra or {}
    import datetime
    month = datetime.datetime.now().strftime('%B')
    transit = extra.get('transit_hours', 'unknown')
    distance = extra.get('distance', 'unknown')
    risk = extra.get('risk_level', 'unknown')
    mode = inputs.get('mode', '?')
    commodity = inputs.get('commodity', 'freight')
    confidence = extra.get('confidence', 'unknown')

    return f"""You are a logistics intelligence analyst. Be direct and specific. No markdown. No filler phrases like "you should consider" or "it's worth noting."

Shipment: {commodity} from {inputs.get('origin','?')} to {inputs.get('destination','?')} by {mode}
Trade: {inputs.get('trade_type','Domestic')} | {distance} mi | Predicted: {transit}h | Risk: {risk} ({confidence}% confidence)
Month: {month}

Respond in EXACTLY two sections separated by ===GOVERNANCE===

SECTION 1: Intelligence brief. 3-4 sentences max. Start with the cost estimate, no preamble.
- Estimated cost range for this route and mode
- One geographic or seasonal risk specific to this corridor in {month}
- One commodity-specific consideration for {commodity} (handling, regulation, temperature, hazmat, weight)
- One actionable tip a junior planner wouldn't know

===GOVERNANCE===

Start with exactly PASS or FLAG.

PASS if the prediction is reasonable. Write nothing else after PASS.

FLAG followed by 1-2 sentences if ANY of these apply:
- Route is physically impossible (e.g., truck/rail from Hawaii or Alaska to mainland requires ocean crossing; water between landlocked states with no navigable waterway)
- Mode is incompatible with commodity (pipeline can only carry liquids, gas, slurry; air has weight/hazmat restrictions)
- Implied speed is >30% off expected (truck 40-50mph, rail 20-30mph, air 350-500mph, water 12-18mph)
- Model confidence below 45% means the classification is unreliable
- Risk level contradicts obvious route characteristics (Low risk on 2500+ mile route is suspicious)

Be blunt about what's wrong. Name the specific issue."""

def _call_claude(explanation, inputs, extra=None):
    import urllib.request
    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 600,
        "messages": [{"role": "user", "content": _build_llm_prompt(explanation, inputs, extra)}]
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return {'text': data['content'][0]['text']}
    except urllib.error.HTTPError as e:
        msg = f"HTTP {e.code}"
        if e.code == 429: msg = "Rate limit exceeded — try again in a minute"
        elif e.code == 401: msg = "Invalid API key"
        print(f"Claude API error: {e}")
        return {'error': msg}
    except Exception as e:
        print(f"Claude API error: {e}")
        return {'error': str(e)}

def _call_gemini(explanation, inputs, extra=None):
    import urllib.request, urllib.error
    prompt = _build_llm_prompt(explanation, inputs, extra)
    # Models ordered by free-tier generosity (RPM / RPD)
    models = ['gemini-3.1-flash-lite', 'gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-3-flash']
    last_error = None
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}]
        }).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                text = data['candidates'][0]['content']['parts'][0]['text']
                print(f"Gemini OK with model: {model}")
                return {'text': text}
        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code} on {model}"
            if e.code == 429:
                print(f"Gemini 429 on {model}, trying next...")
                continue
            elif e.code == 401:
                return {'error': 'Invalid API key'}
            elif e.code == 404:
                print(f"Gemini 404 on {model} (model not found), trying next...")
                continue
            else:
                print(f"Gemini error on {model}: HTTP {e.code}")
                continue
        except Exception as e:
            last_error = str(e)
            print(f"Gemini error on {model}: {e}")
            continue
    return {'error': f'All Gemini models unavailable — {last_error}'}

# ------------------------------------------------------------------
# Static reference data
# ------------------------------------------------------------------
STATE_NAMES = {
    1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
    8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia',
    12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
    18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana',
    23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan',
    27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri', 30: 'Montana',
    31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey',
    35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota',
    39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania',
    44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota',
    47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia',
    53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming',
}
STATE_CODES = {v.lower(): k for k, v in STATE_NAMES.items()}

MODE_NAMES = {1: 'Truck', 2: 'Rail', 3: 'Water', 4: 'Air',
              5: 'Multiple Modes', 7: 'Pipeline'}
MODE_CODES = {v.lower(): k for k, v in MODE_NAMES.items()}

COMMODITY_NAMES = {
    1: 'Live animals/fish', 2: 'Cereal grains', 3: 'Other ag products',
    4: 'Animal feed', 5: 'Meat/seafood', 6: 'Milled grain products',
    7: 'Other foodstuffs', 8: 'Alcoholic beverages', 9: 'Tobacco products',
    10: 'Building stone', 11: 'Natural sands', 12: 'Gravel',
    13: 'Nonmetallic minerals', 14: 'Metallic ores', 15: 'Coal',
    16: 'Crude petroleum', 17: 'Gasoline', 18: 'Fuel oils',
    19: 'Coal-n.e.c.', 20: 'Basic chemicals', 21: 'Pharmaceuticals',
    22: 'Fertilizers', 23: 'Chemical products', 24: 'Plastics/rubber',
    25: 'Logs', 26: 'Wood products', 27: 'Newsprint/paper',
    28: 'Paper articles', 29: 'Printed products', 30: 'Textiles/leather',
    31: 'Nonmetal mineral prods', 32: 'Base metals', 33: 'Articles-base metal',
    34: 'Machinery', 35: 'Electronics', 36: 'Motorized vehicles',
    37: 'Transport equip.', 38: 'Precision instruments', 39: 'Furniture',
    40: 'Miscellaneous mfg prods', 41: 'Waste/scrap', 43: 'Mixed freight',
}
COMMODITY_CODES = {v.lower(): k for k, v in COMMODITY_NAMES.items()}

TRADE_NAMES = {1: 'Domestic', 2: 'Import', 3: 'Export'}
TRADE_CODES = {v.lower(): k for k, v in TRADE_NAMES.items()}

STATE_REGIONS = {
    9:'NE',23:'NE',25:'NE',33:'NE',44:'NE',50:'NE',34:'NE',36:'NE',42:'NE',
    17:'MW',18:'MW',26:'MW',39:'MW',55:'MW',19:'MW',20:'MW',27:'MW',29:'MW',31:'MW',38:'MW',46:'MW',
    10:'S',11:'S',12:'S',13:'S',24:'S',37:'S',45:'S',51:'S',54:'S',1:'S',21:'S',28:'S',47:'S',5:'S',22:'S',40:'S',48:'S',
    4:'W',8:'W',16:'W',30:'W',32:'W',35:'W',49:'W',56:'W',2:'W',6:'W',15:'W',41:'W',53:'W',
}

STATE_CENTROIDS = {
    1:(32.8,-86.8),2:(64.2,-153.0),4:(34.3,-111.7),5:(34.9,-92.4),6:(37.2,-119.6),
    8:(39.0,-105.5),9:(41.6,-72.7),10:(39.0,-75.5),11:(38.9,-77.0),12:(28.6,-82.5),
    13:(32.6,-83.5),15:(20.3,-156.4),16:(44.4,-114.5),17:(40.0,-89.2),18:(39.9,-86.3),
    19:(42.1,-93.5),20:(38.5,-98.4),21:(37.5,-85.3),22:(31.1,-92.0),23:(45.4,-69.3),
    24:(39.1,-76.8),25:(42.2,-71.8),26:(44.3,-85.4),27:(46.3,-94.3),28:(32.7,-89.7),
    29:(38.4,-92.3),30:(47.0,-109.6),31:(41.5,-99.8),32:(39.3,-116.6),33:(43.7,-71.6),
    34:(40.3,-74.5),35:(34.4,-106.1),36:(42.9,-75.5),37:(35.6,-79.4),38:(47.5,-100.5),
    39:(40.4,-82.7),40:(35.5,-97.5),41:(43.9,-120.6),42:(40.6,-77.2),44:(41.7,-71.5),
    45:(33.9,-80.9),46:(44.4,-100.2),47:(35.7,-86.7),48:(31.1,-97.6),49:(39.3,-111.7),
    50:(44.1,-72.7),51:(37.8,-78.2),53:(47.4,-121.5),54:(38.5,-80.9),55:(44.3,-89.6),
    56:(42.8,-107.3),
}

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def get_distance(orig, dest, mode):
    match = distance_lookup[
        (distance_lookup['dms_orig']==orig)&(distance_lookup['dms_dest']==dest)&(distance_lookup['dms_mode']==mode)]
    if len(match) > 0:
        return float(match['distance_miles'].iloc[0])
    if orig in STATE_CENTROIDS and dest in STATE_CENTROIDS:
        lat1,lon1 = STATE_CENTROIDS[orig]; lat2,lon2 = STATE_CENTROIDS[dest]
        gc = haversine_miles(lat1,lon1,lat2,lon2)
        factor = {1:1.25,2:1.15,3:1.4,4:1.0,5:1.2,7:1.15}.get(mode,1.25)
        return float(max(gc*factor, 50))
    return 500.0

def get_corridor_volume(orig, dest, mode):
    match = volume_lookup[
        (volume_lookup['dms_orig']==orig)&(volume_lookup['dms_dest']==dest)&(volume_lookup['dms_mode']==mode)]
    if len(match) > 0:
        return float(match['corridor_volume'].iloc[0])
    return volume_lookup['corridor_volume'].median()

def build_feature_row(orig, dest, mode, sctg, trade):
    orig_region = STATE_REGIONS.get(orig, 'S')
    dest_region = STATE_REGIONS.get(dest, 'S')
    distance = get_distance(orig, dest, mode)
    volume = get_corridor_volume(orig, dest, mode)
    log_vol = float(np.log1p(volume))
    row = {c: 0 for c in FEATURE_COLUMNS}
    for col in [f'orig_{orig_region}',f'dest_{dest_region}',f'mode_{mode}',f'sctg_{sctg}',f'trade_{trade}']:
        if col in row: row[col] = 1
    row['distance_miles'] = distance
    row['log_volume'] = log_vol
    return pd.DataFrame([row], columns=FEATURE_COLUMNS), distance, volume

# ------------------------------------------------------------------
# Rule-based explanation
# ------------------------------------------------------------------
def build_explanation(orig, dest, mode, sctg, trade,
                       transit_hours, risk_level, distance, volume):
    orig_name = STATE_NAMES.get(orig, f'State {orig}')
    dest_name = STATE_NAMES.get(dest, f'State {dest}')
    mode_name = MODE_NAMES.get(mode, 'Unknown')
    commodity_name = COMMODITY_NAMES.get(sctg, 'general freight')
    trade_name = TRADE_NAMES.get(trade, 'Domestic')

    drivers = []
    vol_pct = (volume_lookup['corridor_volume'] <= volume).mean() * 100
    dist_pct = (distance_lookup['distance_miles'] <= distance).mean() * 100

    if distance > 1500:
        drivers.append({'factor':'Long distance','detail':f'{distance:,.0f} mi route ({dist_pct:.0f}th percentile)'})
    elif distance > 800:
        drivers.append({'factor':'Medium distance','detail':f'{distance:,.0f} mi route'})
    if vol_pct > 75:
        drivers.append({'factor':'High-volume corridor','detail':f'Top {100-vol_pct:.0f}% by throughput (congestion risk)'})
    elif vol_pct < 25:
        drivers.append({'factor':'Low-volume corridor','detail':'Limited historical data; outcomes more variable'})
    if mode == 4:
        drivers.append({'factor':'Air freight','detail':'Weather-sensitive; delays often cascade'})
    elif mode == 3:
        drivers.append({'factor':'Water transport','detail':'Port congestion and weather impact transit'})
    elif mode == 5:
        drivers.append({'factor':'Intermodal transfers','detail':'Handoffs between modes add variability'})
    elif mode == 1 and distance > 1500:
        drivers.append({'factor':'Long-haul trucking','detail':'Hours-of-Service regulations extend transit'})
    if trade != 1:
        drivers.append({'factor':f'{trade_name} shipment','detail':'Customs and border processing add time'})

    recommendations = []
    if risk_level == 'High':
        recommendations.append(f'Build a {max(int(transit_hours*0.25),6)}-hour buffer into the delivery commitment.')
        if mode == 1 and distance > 1000:
            recommendations.append('Evaluate intermodal (rail + truck) as a lower-risk alternative.')
        recommendations.append('Monitor this shipment closely; escalate at first sign of delay.')
    elif risk_level == 'Medium':
        recommendations.append(f'Add a modest buffer (~{max(int(transit_hours*0.15),3)} hours) to the planned ETA.')
        recommendations.append('Standard monitoring cadence is appropriate.')
    else:
        recommendations.append('Planned transit time is reliable; no additional buffer needed.')
        recommendations.append('Suitable for time-sensitive customer commitments.')

    risk_phrase = {'High':'high risk of delay','Medium':'moderate risk profile','Low':'low risk profile'}[risk_level]
    narrative = (f'This {trade_name.lower()} shipment of {commodity_name.lower()} '
        f'from {orig_name} to {dest_name} by {mode_name.lower()} '
        f'is projected to take approximately {transit_hours:.1f} hours '
        f'({transit_hours/24:.1f} days) and carries a {risk_phrase}. ')
    if drivers:
        narrative += f'Primary risk factors are {" and ".join(d["factor"].lower() for d in drivers[:2])}.'
    else:
        narrative += 'No significant risk factors identified.'
    return {'narrative': narrative, 'drivers': drivers, 'recommendations': recommendations}

# ------------------------------------------------------------------
# Cost estimation (industry-average rates)
# ------------------------------------------------------------------
# Sources: DAT Freight & Analytics, FreightWaves SONAR, BTS commodity flow data
MODE_COST_PER_MILE = {
    1: (2.20, 3.10),   # Truck (FTL): $2.20-$3.10/mi
    2: (0.80, 1.40),   # Rail: $0.80-$1.40/mi (carload equiv.)
    3: (0.40, 0.90),   # Water/barge: $0.40-$0.90/mi
    4: (5.50, 9.00),   # Air cargo: $5.50-$9.00/mi
    5: (1.80, 2.80),   # Intermodal: $1.80-$2.80/mi
    7: (0.15, 0.35),   # Pipeline: $0.15-$0.35/mi
}

# Risk surcharge: High-risk corridors cost more (insurance, expedite fees)
RISK_SURCHARGE = {'Low': 1.0, 'Medium': 1.08, 'High': 1.18}

# Trade overhead (customs, brokerage, port fees)
TRADE_OVERHEAD = {1: 0, 2: 850, 3: 650}  # Domestic, Import, Export

def estimate_cost(distance, mode, risk_level, trade):
    """Estimate shipping cost range with breakdown."""
    lo, hi = MODE_COST_PER_MILE.get(mode, (2.0, 3.0))
    surcharge = RISK_SURCHARGE.get(risk_level, 1.0)
    trade_fee = TRADE_OVERHEAD.get(trade, 0)

    base_lo = distance * lo
    base_hi = distance * hi
    # Apply risk surcharge
    adj_lo = base_lo * surcharge + trade_fee
    adj_hi = base_hi * surcharge + trade_fee

    # Compare with alternatives
    alternatives = {}
    for alt_mode, (alt_lo, alt_hi) in MODE_COST_PER_MILE.items():
        if alt_mode == mode:
            continue
        alt_cost = distance * ((alt_lo + alt_hi) / 2) * surcharge + trade_fee
        mode_name = MODE_NAMES.get(alt_mode, 'Unknown')
        savings = ((adj_lo + adj_hi) / 2) - alt_cost
        alternatives[mode_name] = {
            'cost_est': round(alt_cost, 0),
            'savings': round(savings, 0),  # positive = you save by switching
        }

    # Sort alternatives by cost
    sorted_alts = sorted(alternatives.items(), key=lambda x: x[1]['cost_est'])

    return {
        'base_range': [round(base_lo, 0), round(base_hi, 0)],
        'adjusted_range': [round(adj_lo, 0), round(adj_hi, 0)],
        'risk_surcharge_pct': round((surcharge - 1) * 100, 0),
        'trade_overhead': trade_fee,
        'per_mile_range': [lo, hi],
        'alternatives': sorted_alts[:3],  # top 3 cheapest alternatives
    }

# ------------------------------------------------------------------
# Core prediction (shared by single + batch)
# ------------------------------------------------------------------
def run_single_prediction(orig, dest, mode, sctg, trade, use_llm=False):
    if orig == dest:
        raise ValueError('Origin and destination must differ.')
    row, distance, volume = build_feature_row(orig, dest, mode, sctg, trade)
    transit_hours = float(regressor.predict(row.values)[0])
    risk_level    = str(classifier.predict(row.values)[0])
    classes = list(classifier.classes_)
    probs = classifier.predict_proba(row.values)[0]
    risk_probs = {cls: float(p) for cls, p in zip(classes, probs)}

    explanation = build_explanation(orig, dest, mode, sctg, trade,
                                    transit_hours, risk_level, distance, volume)
    inputs_named = {
        'origin': STATE_NAMES.get(orig), 'destination': STATE_NAMES.get(dest),
        'mode': MODE_NAMES.get(mode), 'commodity': COMMODITY_NAMES.get(sctg),
        'trade_type': TRADE_NAMES.get(trade),
    }
    result = {
        'transit_hours': round(transit_hours,1), 'transit_days': round(transit_hours/24,2),
        'risk_level': risk_level,
        'risk_probs': {k: round(v,3) for k,v in risk_probs.items()},
        'distance_miles': round(distance,0), 'corridor_volume_kt': round(volume,1),
        'explanation': explanation, 'inputs': inputs_named,
        'mode_code': mode,
        'coords': {
            'origin': list(STATE_CENTROIDS.get(orig, (39.8, -98.6))),
            'destination': list(STATE_CENTROIDS.get(dest, (39.8, -98.6))),
        },
        'cost': estimate_cost(distance, mode, risk_level, trade),
    }
    if use_llm:
        top_prob = max(risk_probs.values()) * 100
        llm = get_llm_narrative(explanation, inputs_named, extra={
            'distance': round(distance, 0),
            'transit_hours': round(transit_hours, 1),
            'risk_level': risk_level,
            'confidence': round(top_prob, 1),
        })
        if llm and 'text' in llm:
            result['llm_narrative'] = llm['text']
            if llm.get('governance'):
                result['llm_governance'] = llm['governance']
        elif llm and 'error' in llm:
            result['llm_error'] = llm['error']
    return result

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/reference')
def reference():
    return jsonify({
        'states':      [{'code':k,'name':v} for k,v in sorted(STATE_NAMES.items(), key=lambda x:x[1])],
        'modes':       [{'code':k,'name':v} for k,v in MODE_NAMES.items()],
        'commodities': [{'code':k,'name':v} for k,v in sorted(COMMODITY_NAMES.items(), key=lambda x:x[1])],
        'trade_types': [{'code':k,'name':v} for k,v in TRADE_NAMES.items()],
        'metrics':     METRICS,
        'llm_available': bool(ANTHROPIC_API_KEY or GEMINI_API_KEY),
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        orig  = int(data['origin']); dest = int(data['destination'])
        mode  = int(data['mode']);   sctg = int(data.get('commodity',43))
        trade = int(data.get('trade_type',1))
        use_llm = bool(data.get('use_llm', False))
        result = run_single_prediction(orig, dest, mode, sctg, trade, use_llm=use_llm)
        prediction_history.append({
            'id': len(prediction_history)+1,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            **{k: result['inputs'][k] for k in ['origin','destination','mode','commodity','trade_type']},
            'transit_hours': result['transit_hours'], 'transit_days': result['transit_days'],
            'risk_level': result['risk_level'], 'distance_miles': result['distance_miles'],
        })
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400

# -- History --
@app.route('/api/history')
def get_history():
    return jsonify({'predictions': list(reversed(prediction_history))})

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    prediction_history.clear()
    return jsonify({'status': 'cleared'})

@app.route('/api/history/export')
def export_history():
    if not prediction_history:
        return jsonify({'error': 'No predictions in history'}), 400
    si = io.StringIO()
    fields = ['id','timestamp','origin','destination','mode','commodity','trade_type',
              'transit_hours','transit_days','risk_level','distance_miles']
    writer = csv.DictWriter(si, fieldnames=fields)
    writer.writeheader()
    for e in prediction_history: writer.writerow(e)
    return Response(si.getvalue(), mimetype='text/csv',
        headers={'Content-Disposition':'attachment; filename=freightai_history.csv'})

# -- Batch --
def _resolve_code(value, name_to_code, fallback=None):
    if value is None or str(value).strip() == '': return fallback
    try: return int(float(value))
    except (ValueError, TypeError): return name_to_code.get(str(value).strip().lower(), fallback)

@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No CSV file uploaded'}), 400
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a .csv'}), 400
        content = file.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(content))
        results, errors = [], []
        for i, row in enumerate(reader, 1):
            try:
                orig  = _resolve_code(row.get('origin'), STATE_CODES)
                dest  = _resolve_code(row.get('destination'), STATE_CODES)
                mode  = _resolve_code(row.get('mode'), MODE_CODES, 1)
                sctg  = _resolve_code(row.get('commodity'), COMMODITY_CODES, 43)
                trade = _resolve_code(row.get('trade_type'), TRADE_CODES, 1)
                if orig is None or dest is None:
                    errors.append({'row':i,'error':'Could not resolve origin or destination'}); continue
                if orig == dest:
                    errors.append({'row':i,'error':'Origin = destination'}); continue
                result = run_single_prediction(orig, dest, mode, sctg, trade)
                result['row_number'] = i
                results.append(result)
                prediction_history.append({
                    'id': len(prediction_history)+1,
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    **{k: result['inputs'][k] for k in ['origin','destination','mode','commodity','trade_type']},
                    'transit_hours': result['transit_hours'], 'transit_days': result['transit_days'],
                    'risk_level': result['risk_level'], 'distance_miles': result['distance_miles'],
                })
            except Exception as e:
                errors.append({'row':i,'error':str(e)})
        return jsonify({'total_rows':len(results)+len(errors),'successful':len(results),
                        'failed':len(errors),'results':results,'errors':errors})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict_batch/export', methods=['POST'])
def export_batch():
    try:
        data = request.get_json()
        results = data.get('results', [])
        if not results: return jsonify({'error':'No results'}), 400
        si = io.StringIO()
        fields = ['row','origin','destination','mode','commodity','trade_type',
                  'distance_miles','transit_hours','transit_days','risk_level',
                  'risk_high','risk_medium','risk_low','narrative']
        writer = csv.DictWriter(si, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'row':r.get('row_number',''), 'origin':r['inputs']['origin'],
                'destination':r['inputs']['destination'], 'mode':r['inputs']['mode'],
                'commodity':r['inputs']['commodity'], 'trade_type':r['inputs']['trade_type'],
                'distance_miles':r['distance_miles'], 'transit_hours':r['transit_hours'],
                'transit_days':r['transit_days'], 'risk_level':r['risk_level'],
                'risk_high':r['risk_probs'].get('High',0),
                'risk_medium':r['risk_probs'].get('Medium',0),
                'risk_low':r['risk_probs'].get('Low',0),
                'narrative':r['explanation']['narrative'],
            })
        return Response(si.getvalue(), mimetype='text/csv',
            headers={'Content-Disposition':'attachment; filename=freightai_batch_results.csv'})
    except Exception as e:
        return jsonify({'error':str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
