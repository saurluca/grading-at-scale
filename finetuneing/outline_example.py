# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field
import outlines
from enum import Enum

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizer and model
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)


class Rating(Enum):
    false = 0
    partially_correct = 0.5
    correct = 1


class GradingResult(BaseModel):
    reasoning: str = Field(
        ..., description="Brief explanation for the grade no more then 10 words"
    )
    score: Rating = Field(
        ..., description="Grade as a number: false, partially_correct, or correct"
    )


prompt_template = """question: {question} \n student answer: {answer} \n reference answer: {reference}"""


# %%

prompt = """ 
question: State at least 4 of the differences shown in the lecture between the UDP and TCP headers. 
student answer: The response correctly identifies four differences between TCP and UDP headers. 
reference answer: Possible Differences :
The UPD header (8 bytes) is much shorter than the TCP header (20-60 bytes)
The UDP header has a fixed length while the TCP header has a variable length
Fields contained in the TCP header and not the UDP header :
-Sequence number
-Acknowledgment number
-Reserved
-Flags/Control bits
-Advertised window
-Urgent Pointer
-Options + Padding if the options are
UDP includes the packet length (data + header) while TCP has the header length/data offset (just header) field instead
The sender port field is optional in UDP, while the source port in TCP is necessary to establish the connection. 
"""

# Simple classification
grading_result = model(
    prompt,
    GradingResult,
    max_new_tokens=200,
)

grading_result = GradingResult.model_validate_json(grading_result)
print(f"Reasoning: {grading_result.reasoning}")
print(f"Score: {grading_result.score}")
