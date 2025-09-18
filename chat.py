from dotenv import load_dotenv
import dspy
from synthetic_data.model_builder import build_lm, model_configs

load_dotenv()


def list_defined_models():
    return list(model_configs.keys())


class ChatSignature(dspy.Signature):
    """You are a helpful assistant that can answer questions."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


def interactive_chat(lm: dspy.LM):
    dspy.settings.configure(lm=lm)
    predict = dspy.Predict(ChatSignature)
    history = dspy.History(messages=[])

    print("Start chatting! (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Ending conversation.")
            break

        outputs = predict(question=user_input, history=history)
        print(f"AI: {outputs.answer}\n")

        # Maintain conversation context in history
        history.messages.append({"question": user_input, "answer": outputs.answer})


def main():
    print("Available models (from model_builder):")
    model_names = list_defined_models()
    for name in model_names:
        print(f" • {name}")
    print()

    chosen = input("Enter model name to use (or press Enter for the first): ").strip()
    model_name = (
        chosen if chosen in model_names else (model_names[0] if model_names else None)
    )
    if not model_name:
        print("No model available. Exiting.")
        return

    print(f"\nUsing model: {model_name}\n")
    lm = build_lm(model_name)
    interactive_chat(lm)


if __name__ == "__main__":
    main()
