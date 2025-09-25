# %%
import dspy
from dotenv import load_dotenv
import os

load_dotenv()

lm = dspy.LM(
    "openrouter/meta-llama/llama-4-scout:free", api_key=os.getenv("OPENROUTER_API_KEY")
)
dspy.configure(lm=lm)


class Base(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


model = dspy.Predict(Base)

# %%
output = model(question="what is the captial of argentina?")
print(output.answer)
