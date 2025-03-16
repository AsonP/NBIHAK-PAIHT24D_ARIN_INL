import cv2
import json
import numpy as np
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import base64
import os
from pyzbar.pyzbar import decode

# ✅ Hitta rätt sökväg till `ean_codes.json`
ASSETS_DIR = "assets"
JSON_FILE = os.path.join(ASSETS_DIR, "ean_codes.json")

# ✅ Ladda produktinformation från JSON
def load_products():
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return {item["EAN"]: item for item in data.get("products", [])}
    except FileNotFoundError:
        print("❌ Kunde inte hitta ean_codes.json i assets/")
        return {}

# ✅ Förbättra bild för bättre streckkodsläsning
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ✅ Läs EAN från en bild men visa originalet
def read_ean_from_image(image, product_data):
    if image is None:
        return None, "Ingen bild laddades."

    processed_image = preprocess_image(image)  # Förbättrad bild för analys
    barcodes = decode(processed_image)

    if not barcodes:
        return image, "Ingen streckkod hittades."  # 🟢 Visar originalet istället för gråskalan

    result = []
    for barcode in barcodes:
        ean_code = barcode.data.decode("utf-8")
        result.append(f"Upptäckt EAN: {ean_code}")

        if ean_code in product_data:
            product = product_data[ean_code]
            result.append(f"**Produkt:** {product['Info']}")
            result.append(f"**Förpackning:** {product['Förpackning']}")
            result.append(f"**Pant:** {product['Pant']}")
        else:
            result.append("❌ Produkten finns inte i databasen.")

    return image, "\n".join(result)  # 🟢 Returnerar originalbilden istället för svartvit

# ✅ Hantera kameraskanning med OpenCV
def scan_from_camera():
    product_data = load_products()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        return None, "❌ Kunde inte öppna kameran."

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, "❌ Kunde inte ta bild från kameran."

    return read_ean_from_image(frame, product_data)

# ✅ Dash-app med Bootstrap (Dark Mode)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    html.H1("📦 EAN-kod Skanner", className="text-center mt-4"),
    html.P("💡 Skanna en EAN-kod från en bild eller kamera.", className="text-center"),

    dbc.Row([
        dbc.Col([
            html.Label("📂 **Ladda upp en bild**", className="upload-label"),
            dcc.Upload(
                id="upload-image",
                children=html.Div(["Drag and drop eller klicka här för att ladda upp"]),
                className="upload-box",
                multiple=False
            ),
            html.Button("📷 Skanna med kamera", id="scan-camera", n_clicks=0, className="btn btn-success mt-2 scan-btn"),
        ], width=6),

        dbc.Col([
            html.H5("📸 Skannad bild:"),
            html.Img(id="image-output", className="image-preview"),
            html.H5("📋 **Resultat:**"),
            html.Pre(id="text-output", className="result-box")
        ], width=6)
    ], className="mt-4"),
])

# ✅ Callback för att ladda upp bilder och skanna
@app.callback(
    [Output("image-output", "src"), Output("text-output", "children")],
    [Input("upload-image", "contents"), Input("scan-camera", "n_clicks")],
    [State("upload-image", "filename")]
)
def update_output(upload_contents, n_clicks, filename):
    ctx = dash.callback_context
    product_data = load_products()

    if not ctx.triggered:
        return None, "Ladda upp en bild eller skanna med kamera."

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "upload-image" and upload_contents:
        content_type, content_string = upload_contents.split(",")
        decoded = base64.b64decode(content_string)
        image = np.array(cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR))
        processed_image, result = read_ean_from_image(image, product_data)

    elif trigger_id == "scan-camera" and n_clicks > 0:
        processed_image, result = scan_from_camera()

    else:
        return None, "Ingen bild tillgänglig."

    if processed_image is None:
        return None, result

    _, buffer = cv2.imencode(".jpg", processed_image)
    encoded_image = base64.b64encode(buffer).decode()

    return f"data:image/jpeg;base64,{encoded_image}", result

# ✅ Starta servern
if __name__ == "__main__":
    app.run_server(debug=True)