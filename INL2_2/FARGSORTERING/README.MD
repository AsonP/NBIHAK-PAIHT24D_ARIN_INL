# Sopsortera med Färgade Påsar

## Projektöversikt
Detta projekt är en Dash-applikation designad för att sortera avfall med hjälp av färgade påsar. Applikationen tar bilder från en kamera, bearbetar färginformationen och använder maskininlärning för att klassificera färgen i fördefinierade kategorier.
Den använder OpenCV, Dash och scikit-learn. Den tänar en K-Nearest Neighbors (KNN) modell för färgigenkänning


## Funktioner
- Fånga och bearbeta färginformation från ett livekameraflöde.
- Träna en maskininlärningsmodell för att känna igen och klassificera färger.
- Spara och ladda färgdata med JSON-filer.
- Användarvänligt gränssnitt för att lära systemet nya färger och identifiera befintliga.

## Användning
Placera färgdata i katalogen "assets".

## Kör applikationen:
python main.py
Öppna webbläsaren på http://localhost:8050 för att använda applikationen.

## Struktur
main.py: Huvudprogrammet
assets/colors.json: JSON-fil med färgdata
assets/color_model.json: Sparad maskininlärningsmodell

## Författare
Applikationen är utvecklad av Peter Arvidsson

