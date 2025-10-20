import dspy
from typing import List, Optional



class GraderSingle(dspy.Signature):
    """
    You are a university professor for a introductory class.
    Your job is to grade exercises and decide if the student answers are correct(2), partially correct(1), or incorrect(0).
    Return the corrsponding integer label for the grading, 0 for incorrect, 1 for partially correct, 2 for correct.
    """
        
    question: str = dspy.InputField(description="The question text")
    reference: Optional[str] = dspy.InputField(
        description="The ground truth reference text", optional=True
    )
    reference_answer: Optional[str] = dspy.InputField(
        description="The ground truth reference answer", optional=True
    )
    answer: str = dspy.InputField(description="The student answer")

    label: int = dspy.OutputField(
        description="2 if the student answer is correct, 1 if the student answer is partially correct, 0 if the student answer is incorrect"
    )


class GraderSingle_without_prompt(dspy.Signature):        
    question: str = dspy.InputField(description="The question text")
    reference: Optional[str] = dspy.InputField(
        description="The ground truth reference text", optional=True
    )
    reference_answer: Optional[str] = dspy.InputField(
        description="The ground truth reference answer", optional=True
    )
    answer: str = dspy.InputField(description="The student answer")

    label: int = dspy.OutputField(
        description="2 if the student answer is correct, 1 if the student answer is partially correct, 0 if the student answer is incorrect"
    )



class GraderPerQuestion(dspy.Signature):
    """
    You are a university professor for an introductory class.
    Your job is to grade exercises and decide if the student answers are correct(2), partially correct(1), or incorrect(0).
    Return the corrsponding integer labels as a list for the grading, 0 for incorrect, 1 for partially correct, 2 for correct.
    """

    question: str = dspy.InputField(description="The question text")
    reference: Optional[str] = dspy.InputField(
        description="The ground truth reference text for this question",
        optional=True,
    )
    reference_answer: Optional[str] = dspy.InputField(
        description="The ground truth reference answer", optional=True
    )
    answers: List[str] = dspy.InputField(description="The list of student answers")

    predicted_labels: List[int] = dspy.OutputField(
        description="Your labels for the provided answers, 0 for incorrect, 1 for partially correct, 2 for correct"
    )


class GraderAll(dspy.Signature):
    """
    You are a university professor for a introductory class.
    Your job is to grade exercises and decide if the student answers are correct(2), partially correct(1), or incorrect(0).
    Return the corrsponding integer labels as a list for the grading, 0 for incorrect, 1 for partially correct, 2 for correct.
    """

    questions: List[str] = dspy.InputField(description="Unique questions list")
    references: Optional[List[str]] = dspy.InputField(
        description="Reference texts aligned with questions (same order)", optional=True
    )
    reference_answers: Optional[List[str]] = dspy.InputField(
        description="Correct reference answers aligned with questions, same order as questions",
        optional=True,
    )
    counts_per_question: List[int] = dspy.InputField(
        description="Number of student answers for each question (same order as questions)"
    )
    answers_flat: List[str] = dspy.InputField(
        description="All student answers flattened in question-major order"
    )

    labels_flat: List[int] = dspy.OutputField(
        description="Labels flattened in question-major order, 0 for incorrect, 1 for partially correct, 2 for correct"
    )
