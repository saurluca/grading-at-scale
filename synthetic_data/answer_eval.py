# %%
import os
import pandas as pd
import dspy

from dotenv import load_dotenv
from types import SimpleNamespace
load_dotenv()

cfg = SimpleNamespace(**{})


# Configuration constants
cfg.tasks_file_path = os.path.join(os.path.dirname(__file__), "../", "data", "privacy", "student_answers.csv")

teacher_lm = dspy.LM(
    "azure/gpt-4o",
    api_base=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2024-12-01-preview",
)

# read in data
tasks = pd.read_csv(cfg.tasks_file_path)

# %%

# tasks.tail()



