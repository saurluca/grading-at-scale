# %%
NUM_DOMAINS = 5  # number of domains
NUM_ANSWER_TYPES = 3  # number of answers types, (1-correct, 0.5-partially_correct_incomplete, 0-incorrect)
num_questions_per_domain = 20  # number of questions per domain
num_answers_per_type = 5  # number of answers per question of one type

answers_per_question = NUM_ANSWER_TYPES * num_answers_per_type
answers_per_domain = num_questions_per_domain * answers_per_question
total_answers = NUM_DOMAINS * answers_per_domain
total_questions = NUM_DOMAINS * num_questions_per_domain


print(f"Total number of answers per question: {answers_per_question}")
print(f"Total number of answers per domain: {answers_per_domain}")
print(f"Total number of answers: {total_answers}")
print(f"Total number of questions: {total_questions}")

# %%

"""
Open Questions:

1. Choice of number of answers and questions



Sure:
- 3-way: 1-correct, 0.5-partially_correct_incomplete, 0-incorrect
- Number of responses per type should be balance.
- 5 Domains, logic, neuroscience, CNP, AI, Privacy
- Differentaite  between train, val and test set
    - unseen answers < unseen questions < unseen domains (increasing uncertainty)


Plan:
Create 20 questions for each domain.
Create small syntheitc dataset, check manualy and with machine if good
Create larger synthetic dataset, check manualy and provide to humans for validation

IF more data is needed later on, create more synthetic data without human validation. 

"""
