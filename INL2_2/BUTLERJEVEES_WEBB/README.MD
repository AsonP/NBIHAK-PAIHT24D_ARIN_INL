# Jevees – Din Sofistikerade AI-Butler


📌 Jevees är en intelligent och sofistikerad AI-butler byggd med modern teknik för att ge personliga och kontextuella svar på dina frågor. Projektet kombinerar språkmodeller, vektorsökning och ett användarvänligt webbgränssnitt för att skapa en effektiv och interaktiv upplevelse.

# Funktioner
Multispråkstöd: Jevees svarar på samma språk som din fråga (t.ex. svenska eller engelska).
Minneshantering: Håller koll på de senaste 5 meddelandena för att ge kontextuella svar.
Vektorbaserad kontext: Använder FAISS och SentenceTransformers för att hämta relevant bakgrundsinformation.
Webbgränssnitt: Byggt med Gradio för en enkel och visuellt tilltalande användarupplevelse.
Optimering av API-anrop: Implementerar cachning för att minska antalet API-anrop och förbättra prestanda.


# Tekniker och bibliotek
Python: Grundspråket för projektet.
LangChain: Hanterar språkmodellinteraktioner och minneskontext.
OpenAI GPT-4o: Kraftfull språkmodell för att generera svar.
SentenceTransformers: Skapar embeddings för vektorsökning.
FAISS: Effektiv vektordatabas för att hämta relevant information.
Gradio: Webbgränssnitt för interaktiv användning.
Caching: En JSON-baserad cache för att minska API-anrop.
Höjdpunkt: Minskning av API-anrop
En central del av detta projekt är att optimera resursanvändningen genom att minska antalet API-anrop till OpenAI:s språkmodell. Detta uppnås genom en cachningsmekanism:

När en användare ställer en fråga kontrolleras först om svaret redan finns i cachen (jevees_cache.json).
Om svaret finns i cachen returneras det direkt utan att ett API-anrop görs.
Om svaret inte finns genereras det av GPT-4o, sparas i cachen och returneras sedan.
Detta minskar både kostnader och latens, särskilt för återkommande frågor.


# Användning
Öppna webbgränssnittet i din webbläsare.
Skriv en fråga i textrutan (t.ex. "Vad är vädret idag?" eller "What’s the weather tomorrow?").
Jevees svarar direkt i chattfältet på samma språk som frågan.
Skriv "exit", "quit" eller "avsluta" för att avsluta konversationen.
Använd "Rensa konversationen"-knappen för att nollställa chatten.

# Exempel
Fråga: "Vad är vädret idag?"
Svar: "Idag är det soligt och fint!"
Fråga: "What’s the weather tomorrow?"
Svar: "Tomorrow it will be cloudy but warm."


# Licens
Detta projekt är licensierat under MIT-licensen – se LICENSE för detaljer.
