import openai
import streamlit as st
import requests
import json
import chardet

# Greife auf den API-Schlüssel aus der Umgebungsvariable zu
api_key = st.secrets['OPENAI_API']

if not api_key:
    st.error("Kein API-Schlüssel gesetzt. Bitte setze die Umgebungsvariable OPENAI_API_KEY.")
else:
    openai.api_key = api_key
    st.write(f" ")

# URL of the trainingsdaten.json file in your GitHub repository
url = "https://raw.githubusercontent.com/Bernhard-Keller123/AventraGPT_MK/main/trainingdata.json"

# Funktion zum Laden der Trainingsdaten von GitHub
def lade_trainingsdaten_aus_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        content = response.content.decode('utf-8')
        if content.strip():  # Check if content is not empty
            try:
                json_data = json.loads(content)
                # Ensure the JSON data has the correct structure
                if "data" in json_data:
                    return json_data["data"]
                else:
                    st.error("JSON-Daten haben nicht die erwartete Struktur.")
                    return []
            except json.JSONDecodeError:
                st.error("Fehler beim Dekodieren der JSON-Daten.")
                return []
        else:
            return []  # Return empty list if JSON file is empty
    else:
        st.error("Fehler beim Laden der Trainingsdaten von GitHub.")
        return []

trainingsdaten = lade_trainingsdaten_aus_github(url)
chat_history = [{"role": "system", "content": td} for td in trainingsdaten]

def generiere_antwort(prompt):
    chat_history.append({"role": "user", "content": prompt})
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_history,
            max_tokens=600,
            n=1,
            stop=None,
            temperature=0.7
        )
        antwort = response.choices[0].message['content'].strip()
        chat_history.append({"role": "assistant", "content": antwort})
        return antwort
    except openai.error.OpenAIError as e:
        if "quota" in str(e):
            return "Du hast dein aktuelles Nutzungslimit überschritten. Bitte überprüfe deinen Plan und deine Abrechnungsdetails unter https://platform.openai.com/account/usage."
        return str(e)

# Streamlit App
st.title("AventraGPT_MK")

# Eingabefeld für den Prompt
prompt = st.text_input("Du: ")

# Schaltfläche zum Senden des Prompts
if st.button("Senden"):
    if prompt:
        antwort = generiere_antwort(prompt)
        st.text_area("AventraGPT:", value=antwort, height=200, max_chars=None)

# Datei-Upload für Trainingsdaten
uploaded_file = st.file_uploader("Trainingsdaten hochladen", type=["txt"])

# Schaltfläche zum Laden der Trainingsdaten
if st.button("Trainingsdaten laden"):
    if uploaded_file:
        try:
            # Versuche, die Datei zu lesen und die Kodierung zu erkennen
            raw_data = uploaded_file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            training_data = raw_data.decode(encoding)

            # Append new training data to existing data
            if training_data.strip():
                existing_data = lade_trainingsdaten_aus_github(url)
                existing_data.append(training_data)
                # Update JSON structure with new data
                updated_data = {
                    "message": "The repo AventraGPT_MK has been trained with the following data:",
                    "data": existing_data
                }
                with open('trainingdata.json', 'w') as f:
                    json.dump(updated_data, f, ensure_ascii=False, indent=4)

            st.success("Trainingsdaten erfolgreich geladen.")
        except Exception as e:
            st.error(f"Fehler beim Laden der Datei: {e}")

# Anzeige des Gesprächsverlaufs
st.subheader("Trainingsdaten und Gesprächsverlauf")
for eintrag in chat_history:
    if eintrag['role'] == 'user':
        st.write(f"Du: {eintrag['content']}")
    elif eintrag['role'] == 'assistant':
        st.write(f"LLM: {eintrag['content']}")
    elif eintrag['role'] == 'system':
        st.write(f"System: {eintrag['content']}")
