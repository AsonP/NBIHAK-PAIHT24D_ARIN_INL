import os
import logging
import pandas as pd
import time
import faiss
import numpy as np
import re
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from config import SYSTEM_PROMPT, FALLBACK_RESPONSES, SIGHTSEEING_LOCATIONS

# ✅ Ladda miljövariabler
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️ Varning: OPENAI_API_KEY saknas! Lägg till den i en .env-fil eller som en miljövariabel.")
    exit(1)

# ✅ Konfigurera loggning
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Filhantering för cachelagring av svar
RESPONSES_FILE = "data/responses.json"

def load_responses():
    """Läser in tidigare sparade svar från JSON-filen."""
    try:
        with open(RESPONSES_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_response(question, answer):
    """Sparar nya svar i JSON-filen."""
    responses = load_responses()
    responses[question] = answer
    with open(RESPONSES_FILE, "w", encoding="utf-8") as file:
        json.dump(responses, file, indent=4, ensure_ascii=False)

def get_cached_response(question):
    """Kollar om frågan redan har ett sparat svar i JSON-filen."""
    responses = load_responses()
    return responses.get(question, None)

# ✅ Ladda in CSV-filen med hotellrecensioner
def load_hotel_reviews(file_path):
    try:
        df = pd.read_csv(file_path, encoding="utf-8").iloc[:1000]
        if df.empty:
            logging.warning("⚠️ Varning: Filen är tom!")
            return None
        logging.info("✅ Hotellrecensioner laddade!")
        return df
    except FileNotFoundError:
        logging.error(f"🚨 Fel: Filen hittades inte vid sökvägen {file_path}")
    except pd.errors.ParserError:
        logging.error("🚨 Fel: Filen verkar ha ett ogiltigt format!")
    except UnicodeDecodeError:
        logging.error("🚨 Fel: Filen har felaktig kodning. Prova att spara den med UTF-8.")
    return None

# ✅ Ladda hotellrecensioner och skapa FAISS-index
hotel_data = load_hotel_reviews("data/Hotel_Reviews.csv")

# ✅ Förbättrad prompt-engineering
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{user_query}")
])

# ✅ Ladda SentenceTransformer för CPU-användning
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

def embed_texts(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)

# ✅ Funktion för att fråga chatboten
def ask_chatbot(user_query):
    """Försöker svara från JSON, FAISS eller API beroende på frågans kategori."""
    
    # ✅ Identifiera språk baserat på användarens fråga
    language = "swedish" if re.search(r"[åäöÅÄÖ]", user_query, re.IGNORECASE) else "english"
    
    # ✅ Shoppingfrågor hanteras direkt från config.py
    shopping_keywords = ["shopping", "store", "buy", "gift", "present", "mall"]
    if any(keyword in user_query.lower() for keyword in shopping_keywords):
        match = re.search(r'in ([A-Za-z]+)', user_query, re.IGNORECASE)
        city = match.group(1) if match else ("den här staden" if language == "swedish" else "this city")
        return FALLBACK_RESPONSES["shopping"][language].format(city=city)

    # ✅ Sightseeingfrågor hanteras från config.py
    sightseeing_keywords = ["sightseeing", "landmark", "attractions", "museum", "monument", "things to see"]
    if any(keyword in user_query.lower() for keyword in sightseeing_keywords):
        match = re.search(r'in ([A-Za-z]+)', user_query, re.IGNORECASE)
        city = match.group(1) if match else ("den här staden" if language == "swedish" else "this city")
        attractions = SIGHTSEEING_LOCATIONS.get(city, {}).get(language, "många intressanta platser" if language == "swedish" else "many interesting places")
        return FALLBACK_RESPONSES["sightseeing"][language].format(city=city, attractions=attractions)

    # ✅ Om inget annat hittades, gör API-anrop
    try:
        model = ChatOpenAI(api_key=OPENAI_API_KEY)
        prompt = prompt_template.format(user_query=user_query)
        response = model.invoke(prompt)

        if response.content.strip():
            save_response(user_query, response.content)
            return response.content
        else:
            return "Jag är inte säker på svaret just nu. Kolla resewebbplatser som Booking.com."

    except Exception as e:
        return f"⚠️ API-fel: {str(e)}. Kontrollera din API-nyckel eller saldo."

# ✅ Användargränssnitt
if __name__ == "__main__":
    print("🌍 Välkommen till hotell-chatboten! Ställ frågor om hotell i valfri stad.")
    while True:
        user_question = input("✈️ Fråga chatboten om hotell (eller skriv 'exit' för att avsluta): ")
        if user_question.lower() == 'exit':
            print("👋 Hej då!")
            break
        answer = ask_chatbot(user_question)
        print(f"🤖 Chatbot svarar: {answer}\n")