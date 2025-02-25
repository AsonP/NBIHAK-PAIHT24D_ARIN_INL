# 📌 Hotell-Chatbot
En enkel men kraftfull chatbot som hjälper användare att hitta hotell, shopping och sevärdheter i olika städer.

# 📜 Beskrivning
Denna applikation är en del av en skoluppgift och bygger på en chatbot som kan svara på frågor om:

* Hotell (genom att använda FAISS för att söka i hotellrecensioner)

* Shopping (genom fördefinierade svar i config.py)

* Sevärdheter (genom fördefinierade svar i config.py)

Chatboten använder OpenAI:s API för att ge dynamiska svar och har fallback-responser för att minska onödiga API-anrop.

# 🏗 Teknisk uppbyggnad
Projektet består av följande huvudfiler:

* main.py → Hanterar frågorna och skickar dem till rätt källa (FAISS, config.py eller OpenAI API).

* config.py → Innehåller fallback-responser och stödord för att minska hårdkodning i main.py.

* data/responses.json → Sparar återkommande svar för att minska API-anrop.

* data/Hotel_Reviews.csv → Dataset med hotellrecensioner som används för FAISS-sökningar.

# 🛠 Teknologier som används

* Python 3.10+

* FAISS (för vektorsökning av hotellrecensioner)

* LangChain (för AI-integrering och promptdesign)

* SentenceTransformers (för att skapa embeddingar från textdata)

* OpenAI API (för att generera svar där fallback-responser saknas)

# 📄 Licens
Detta projekt är licensierat under MIT-licensen men det är en skoluppgift och kommer endast att används i utbildningssyfte. 

