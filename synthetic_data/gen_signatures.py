import dspy
from typing import List


class CorrectAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: dspy.InputField = dspy.InputField(
        description="The correct reference text for this question",
        optional=True,
    )
    reference_answer: dspy.InputField = dspy.InputField(
        description="The correct reference answer",
        optional=True,
    )
    answer: str = dspy.OutputField(
        description="A short correct student answer that demonstrates understanding of the question. The answer should be accurate and well-reasoned."
    )


class PartialAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: dspy.InputField = dspy.InputField(
        description="The correct reference text for this question",
        optional=True,
    )
    reference_answer: dspy.InputField = dspy.InputField(
        description="The correct reference answer",
        optional=True,
    )
    answer: str = dspy.OutputField(
        description="A short partially correct student answer that demonstrates understanding of the question but is wrong."
    )


class IncorrectAnswerGenerator(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: dspy.InputField = dspy.InputField(
        description="The correct reference text for this question",
        optional=True,
    )
    reference_answer: dspy.InputField = dspy.InputField(
        description="The correct reference answer",
        optional=True,
    )
    answer: str = dspy.OutputField(
        description="A short incorrect student answer that shows misunderstanding or error in reasoning. Be creative. The answer should be plausible but wrong."
    )


# Per-question batched generators (many answers for one question per call)
class CorrectAnswerGeneratorPerQuestion(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: dspy.InputField = dspy.InputField(
        description="The correct reference text for this question",
        optional=True,
    )
    reference_answer: dspy.InputField = dspy.InputField(
        description="The correct reference answer",
        optional=True,
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers to generate for this question"
    )
    answers: List[str] = dspy.OutputField(description="List of correct student answers")


class PartialAnswerGeneratorPerQuestion(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: dspy.InputField = dspy.InputField(
        description="The correct reference text for this question",
        optional=True,
    )
    reference_answer: dspy.InputField = dspy.InputField(
        description="The correct reference answer",
        optional=True,
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers to generate for this question"
    )
    answers: List[str] = dspy.OutputField(
        description="List of partially correct student answers"
    )


class IncorrectAnswerGeneratorPerQuestion(dspy.Signature):
    question: str = dspy.InputField(description="The question text")
    reference: dspy.InputField = dspy.InputField(
        description="The correct reference text for this question",
        optional=True,
    )
    reference_answer: dspy.InputField = dspy.InputField(
        description="The correct reference answer",
        optional=True,
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers to generate for this question"
    )
    answers: List[str] = dspy.OutputField(
        description="List of incorrect student answers"
    )


# All-questions batched generators (many answers for all questions in one call)
class CorrectAnswerGeneratorAll(dspy.Signature):
    questions: List[str] = dspy.InputField(description="The list of questions")
    # references: List[str] = dspy.InputField(description="Optional context texts")
    references: dspy.InputField = dspy.InputField(
        description="The list of reference texts aligned with questions",
        optional=True,
    )
    reference_answers: dspy.InputField = dspy.InputField(
        description="The list of correct reference answers",
        optional=True,
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers per question"
    )
    answers: List[str] = dspy.OutputField(
        description="Flat list of answers in question-major order"
    )


class PartialAnswerGeneratorAll(dspy.Signature):
    questions: List[str] = dspy.InputField(description="The list of questions")
    # references: List[str] = dspy.InputField(description="Optional context texts")
    references: dspy.InputField = dspy.InputField(
        description="The list of reference texts aligned with questions",
        optional=True,
    )
    reference_answers: dspy.InputField = dspy.InputField(
        description="The list of correct reference answers",
        optional=True,
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers per question"
    )
    answers: List[str] = dspy.OutputField(
        description="Flat list of answers in question-major order"
    )


class IncorrectAnswerGeneratorAll(dspy.Signature):
    questions: List[str] = dspy.InputField(description="The list of questions")
    references: dspy.InputField = dspy.InputField(
        description="The list of reference texts aligned with questions",
        optional=True,
    )
    reference_answers: dspy.InputField = dspy.InputField(
        description="The list of correct reference answers",
        optional=True,
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers per question"
    )
    answers: List[str] = dspy.OutputField(
        description="Flat list of answers in question-major order"
    )
