import json
import os
import requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SWISS_API_KEY")  # Or set this manually
API_BASE = "https://api.publicai.co/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}



def list_models():
    resp = requests.get(f"{API_BASE}/models", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def chat_completion(model_id: str, messages: list, temperature=1.0, top_p=1.0, n=1):
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
    }
    resp = requests.post(f"{API_BASE}/chat/completions", json=payload, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def interactive_chat(model_id: str):
    print("Start chatting! (type 'exit' to quit)\n")
    conversation = []  # Will append {"role": "...", "content": "..."} entries
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Ending conversation.")
            break
        conversation.append({"role": "user", "content": user_input})

        response = chat_completion(model_id, conversation)
        # This assumes the first choice is what you want
        reply = response["choices"][0]["message"]["content"]
        print(f"AI: {reply}\n")

        conversation.append({"role": "assistant", "content": reply})

def main():
    print("Listing models...")
    models_resp = list_models()
    models = models_resp.get("data", [])
    print("Available models:")
    for m in models:
        print(f" • {m.get('id')}")
    print()

    chosen = input("Enter model ID to use (or press Enter for the first): ").strip()
    model_id = chosen if chosen else (models[0].get("id") if models else None)
    if not model_id:
        print("No model available. Exiting.")
        return

    print(f"\nUsing model: {model_id}\n")
    interactive_chat(model_id)


if __name__ == "__main__":
    main()