SYSTEM_PROMPT = "You are a helpful travel assistant. Answer questions about hotels, sightseeing, and shopping in a friendly and informative way."

FALLBACK_RESPONSES = {
    "shopping": {
        "swedish": "{city} har många bra shoppingmöjligheter! För lyxshopping kan du besöka stora varuhus som NK i Stockholm. "
                   "För souvenirer rekommenderas marknader och turistområden. Vill du ha tips på specifika butiker?",
        "english": "{city} has great shopping options! For luxury shopping, visit NK in Stockholm. "
                   "For souvenirs and local products, markets and tourist areas are recommended. Would you like store recommendations?"
    },
    "hotels": {
        "swedish": "Jag kan rekommendera några hotell i {city}. Här är några populära alternativ...",
        "english": "I can recommend some hotels in {city}. Here are some popular options..."
    },
    "sightseeing": {
        "swedish": "{city} har många sevärdheter att upptäcka! Populära platser inkluderar {attractions}. "
                    "Vill du ha mer detaljer om en specifik sevärdhet?",
        "english": "{city} has many sights to explore! Popular attractions include {attractions}. "
                   "Would you like more details on a specific attraction?"
    }
}

# Exempel på sevärdheter för olika städer
SIGHTSEEING_LOCATIONS = {
    "Stockholm": {
        "swedish": "Gamla Stan, Vasamuseet, Skansen, Djurgården",
        "english": "Gamla Stan, The Vasa Museum, Skansen, Djurgården"
    },
    "London": {
        "swedish": "Big Ben, Tower of London, British Museum, Hyde Park",
        "english": "Big Ben, Tower of London, British Museum, Hyde Park"
    },
    "Paris": {
        "swedish": "Eiffeltornet, Louvren, Notre-Dame, Montmartre",
        "english": "Eiffel Tower, The Louvre, Notre-Dame, Montmartre"
    }
}