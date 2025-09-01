# %%
import os
import pandas as pd
import dspy

student_lm = dspy.LM(
    "ollama_chat/llama3.2:1b",
    api_base="http://localhost:11434",
    api_key="",
    temperature=2.0,
    max_tokens=100,
)
default_lm = dspy.LM(
    "ollama_chat/llama3.2:3b",
    api_base="http://localhost:11434",
    api_key="",
    # temperature=0.5,
)
dspy.configure(lm=default_lm)

file_path = os.path.join(os.path.dirname(__file__), "data", "logic", "logic_data.csv")

# read in data
data = pd.read_csv(file_path)

# %%


class LogicGrader(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: str = dspy.InputField(description="The ground truth reference text")
    answer: str = dspy.InputField(description="The student answer")

    label: bool = dspy.OutputField(
        description="True if the student answer is correct, False if the student answer is incorrect"
    )


class AnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    answer: str = dspy.OutputField(description="Something about the weather")


grader = dspy.ChainOfThought(LogicGrader)
answer_generator = dspy.ChainOfThought(AnswerGenerator)
answer_generator.set_lm(student_lm)

# %%

example_task = data.iloc[0]
student_answer = answer_generator(
    question=example_task["question"],
    # question="What is the capital of France?",
)

graded_task = grader(
    question=example_task["question"],
    reference=example_task["chunk_text"],
    answer=student_answer.answer,
)

print(f"Question: {example_task['question']}")
print(f"Student Answer: {student_answer.answer}")
print(f"Answer is predicted to be {graded_task.label}")
