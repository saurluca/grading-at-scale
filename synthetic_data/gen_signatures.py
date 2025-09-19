import dspy
from typing import List


class CorrectAnswerGenerator(dspy.Signature):
    """
    You are an excellent student answering a question.
    Your job is to provide a short and correct answer to the question at hand.
    Aim to achieve full points of the correct answer.
    Be creative. The answer should be accurate and well-reasoned.
    """

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
    """
    You are a mediocre student answering a question.
    Your job is to provide a short and partially correct answer to the question at hand.
    Aim to achieve half the points of the correct answer.
    Be creative. The answer should be plausible but wrong.
    """

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
    """
    You are a bad student answering a question.
    Your job is to provide a short and incorrect or contradictory answer to the question at hand.
    Aim to achieve zero points of the correct answer.
    Be creative. The answer should be wrong but still somewhat plausible.
    """

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
    """
    You are an excellent student answering a question.
    Your job is to provide multiple short and correct answers to the question at hand.
    Aim to achieve full points for each answer.
    Be creative. The answer should all be different from each other, accurate and well-reasoned.

    Generate exactly the number of answers specified.
    """

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
    """
    You are a mediocre student answering a question.
    Your job is to provide multiple short and partially correct answers to the question at hand.
    Aim to achieve half the points for each answer.
    The answers should be different from each other.
    Generate exactly the number of answers specified.
    """

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
        description="Number of answers to generate for this question"
    )
    answers: List[str] = dspy.OutputField(
        description="List of partially correct student answers"
    )


class IncorrectAnswerGeneratorPerQuestion(dspy.Signature):
    """
    You are a bad student answering a question.
    Your job is to provide multiple short and incorrect or contradictory answers to the question at hand.
    Aim to achieve zero points for each answer.
    Be creative. The answer should all be different from each other, wrong but still somewhat plausible.

    Generate exactly the number of answers specified.
    """

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
    """
    You are an excellent student answering multiple questions.
    Your job is to provide multiple short and correct answers to each question at hand in the order of the questions.
    Aim to achieve full points for each answer and each question.
    Be creative. The answer should all be different from each other, accurate and well-reasoned.

    Generate exactly the number of answers specified.
    """

    questions: List[str] = dspy.InputField(description="The list of questions")
    # references: List[str] = dspy.InputField(description="Optional context texts")
    references: dspy.InputField = dspy.InputField(
        description="The list of reference texts that the information for the questions are extracted from",
        optional=True,
    )
    reference_answers: dspy.InputField = dspy.InputField(
        description="The list of correct reference answers for the questions",
        optional=True,
    )
    number_of_answers_per_question: int = dspy.InputField(
        description="How many answers per question to generate"
    )
    answers: List[str] = dspy.OutputField(
        description="Flat list of correct answers to the questions in order"
    )


class PartialAnswerGeneratorAll(dspy.Signature):
    """
    You are a mediocre student answering multiple questions.
    Your job is to provide multiple short and partially correct answers to the questions at hand in the order of the questions.
    Aim to achieve half the points for each answer and each question.
    Be creative. The answer should all be different from each other, plausible but wrong.

    Generate exactly the number of answers specified.
    """

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
    """
    You are a bad student answering multiple questions.
    Your job is to provide multiple short and incorrect or contradictory answers to the questions at hand in the order of the questions.
    Aim to achieve zero points for each answer and each question.
    Be creative. The answer should all be different from each other, wrong but still somewhat plausible.

    Generate exactly the number of answers specified.
    """

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
