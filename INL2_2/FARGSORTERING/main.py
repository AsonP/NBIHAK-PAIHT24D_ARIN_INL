import cv2
import json
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Grundinställningar och filhantering
ASSETS_DIR = "assets"
COLOR_FILE = os.path.join(ASSETS_DIR, "colors.json")
MODEL_FILE = os.path.join(ASSETS_DIR, "color_model.json")

def safe_load_json(filename, default=None):
    """Säker JSON-inladdning med fallback"""
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return json.load(f)
        else:
            return default or {}
    except (json.JSONDecodeError, IOError) as e:
        print(f"Fel vid inläsning av {filename}: {e}")
        return default or {}

def safe_save_json(data, filename):
    """Säker JSON-sparning med konvertering av numpy-typer"""
    try:
        # Konvertera numpy-typer till vanliga Python-typer
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        # Använd rekursiv konvertering
        converted_data = json.loads(json.dumps(data, default=convert))

        with open(filename, "w") as f:
            json.dump(converted_data, f, indent=4)
    except IOError as e:
        print(f"Fel vid sparning av {filename}: {e}")

# Initiera färgdata och modelldata
color_data = safe_load_json(COLOR_FILE, {})
model_data = safe_load_json(MODEL_FILE, {"samples": [], "labels": []})

def get_camera_frame():
    """Hämta kameraframe med markeringar"""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    # Rita markeringar på kamerabilden
    cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), 1)        # Mitten
    cv2.circle(frame, (center_x - 20, center_y), 5, (255, 255, 255), 1)   # Vänster
    cv2.circle(frame, (center_x, center_y - 20), 5, (255, 255, 255), 1)   # Uppåt

    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def extract_color_features(rgb):
    """Extrahera detaljerade färgfeatures"""
    try:
        # Konvertera till vanliga Python-listor
        rgb = [int(x) for x in rgb]

        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        return list(hsv) + [
            np.mean(rgb),  # Ljusintensitet
            np.std(rgb)    # Färgvarians
        ]
    except Exception as e:
        print(f"Fel vid feature-extrahering: {e}")
        return None

def capture_advanced_color():
    """Capture and process color from camera"""
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2

        # Sampla fler punkter för mer robust färgbestämning
        sample_points = [
            frame[center_y, center_x],           # Mitt
            frame[center_y, center_x - 20],      # Vänster
            frame[center_y, center_x + 20],      # Höger
            frame[center_y - 20, center_x],      # Upp
            frame[center_y + 20, center_x]       # Ner
        ]

        # Konvertera till numpy-array
        sample_points = np.array(sample_points)

        # Enklare genomsnittsberäkning
        avg_color = np.mean(sample_points, axis=0).astype(int)
        return avg_color.tolist()
    except Exception as e:
        print(f"Fel vid färginsamling: {e}")
        return None

def train_color_model():
    """Träna färgklassificeringsmodell"""
    if not model_data['samples']:
        return None

    try:
        X = np.array(model_data['samples'])
        y = np.array(model_data['labels'])

        # Standardisera features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Träna KNN-klassificerare
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_scaled, y)

        return {
            'model': knn,
            'scaler': scaler
        }
    except Exception as e:
        print(f"Fel vid modellträning: {e}")
        return None

def identify_color_ml(color, trained_model):
    """Identifiera färg med maskininlärning"""
    if not trained_model or color is None:
        return "Ingen tränad modell eller ogiltig färg"

    try:
        features = extract_color_features(color)
        if features is None:
            return "Kunde inte extrahera features"

        scaled_features = trained_model['scaler'].transform([features])
        prediction = trained_model['model'].predict(scaled_features)

        return prediction[0]
    except Exception as e:
        print(f"Fel vid färgidentifiering: {e}")
        return "Identifieringsfel"

# Dash-applikation
app = dash.Dash(__name__)

app.layout = html.Div(style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'padding': '20px'}, children=[
    html.H1("Sopsortera med färgade påsar", style={'text-align': 'center'}),

    html.Div(style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'margin-bottom': '20px'}, children=[
        html.Img(id='camera-feed', style={'width': '400px', 'height': '300px', 'border': '2px solid black', 'margin-top': '20px'}),
        html.Button("Uppdatera Bild", id='refresh-button', n_clicks=0, style={'margin-top': '10px'}),
        html.Div(id='save-status', style={'margin-top': '10px'})  # Ensure this is included
    ]),

    html.Div(style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'margin-bottom': '20px'}, children=[
        html.H3("Lär upp en ny färg"),
        dcc.Input(id='color-category', type='text', placeholder='Kategori', style={'margin-bottom': '10px'}),
        html.Button("Spara", id='save-button', n_clicks=0)
    ]),

    html.Div(style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}, children=[
        html.H3("Identifiera en färg"),
        html.Button("Identifiera", id='identify-button', n_clicks=0),
        html.Div(id='identified-color', style={'margin-top': '10px'})
    ])
])

@app.callback(
    [Output('camera-feed', 'src'),
     Output('color-category', 'value')],
    Input('refresh-button', 'n_clicks')
)
def update_camera_feed(n_clicks):
    frame = get_camera_frame()
    if frame:
        return f'data:image/jpeg;base64,{frame}', ''  # Clear the text box
    return '', ''

@app.callback(
    Output('save-status', 'children'),
    Input('save-button', 'n_clicks'),
    State('color-category', 'value'),
    prevent_initial_call=True
)
def save_and_train_color(n_clicks, category):
    global color_data, model_data

    # Validera indata
    if not n_clicks or not category:
        return "Ange en kategori"

    try:
        color = capture_advanced_color()
        if not color:
            return "Kunde inte capture färg"

        # Spara färg i färgdatabasen
        color_data[category] = color
        safe_save_json(color_data, COLOR_FILE)

        # Uppdatera träningsdata
        features = extract_color_features(color)
        if features is None:
            return "Kunde inte extrahera features"

        model_data['samples'].append(features)
        model_data['labels'].append(category)
        safe_save_json(model_data, MODEL_FILE)

        return f"Färg sparad för: {category}"

    except Exception as e:
        print(f"Fel i save_and_train_color: {e}")
        return f"Fel: {str(e)}"

@app.callback(
    Output('identified-color', 'children'),
    Input('identify-button', 'n_clicks'),
    prevent_initial_call=True
)
def identify_color_improved(n_clicks):
    try:
        color = capture_advanced_color()
        if not color:
            return "Kunde inte capture färg"

        trained_model = train_color_model()
        if not trained_model:
            return "Ingen tränad modell"

        category = identify_color_ml(color, trained_model)
        return f"Identifierad kategori: {category}"

    except Exception as e:
        print(f"Fel i identify_color_improved: {e}")
        return f"Fel: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
