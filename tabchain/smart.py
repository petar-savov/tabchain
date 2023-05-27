import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def question_answering(question):
    """
    Answer a question based on a given context. Uses the 'Let's work this out in a step by step way to make sure that
    we have the right answer.' prompt for chain of thought reasoning - https://arxiv.org/pdf/2211.01910.pdf

    Parameters
    ----------
    question : str
        The question to answer.

    Returns
    -------
    str
        The answer to the question.
    """

    user_message = f"""Let's work this out in a step by step way to make sure that
          we have the right answer. --- {question}"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a question answering model."},
            {"role": "user", "content": user_message},
        ],
    )

    return response["choices"][0]["message"]["content"]


def smart_gpt(question):
    """
    Based on - https://www.youtube.com/watch?v=wVzuvf9D9BU

    The goal is to use a chain of questions and answers to improve performance
    on question answering and reasoning tasks.

    1. Generate 3 candidate answers to the question
    2. Allow the model to reflect on the candidate answers and identify flaws
    3. Ask the model to pick 1 improved final answer


    """

    candidate_answers = [question_answering(question) for _ in range(3)]

    reflexion_prompt = f"""List the flaws and faulty logic in each of the responses. Let's work this out
        in a step by step way to make sure that we have all the errors. --- {candidate_answers}"""

    reflexion_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a researcher tasked with investigating the 3 responses to\
                the question: '{question}'.",
            },
            {"role": "user", "content": reflexion_prompt},
        ],
    )

    reflexion_response = reflexion_response["choices"][0]["message"]["content"]

    resolver_prompt = f"""You are a resolver tasked with (1) finding which 3 responses to the question '{question}' the
        researcher thought as best. There are the 3 candiate(2) improving the answer, (3) printing the improved answer
        in full and only printing the improved answer. These are the original candidate answer --- {candidate_answers}
        --- Let's work this out in a step by step way to make sure that we have the right answer.
        --- {reflexion_response}"""

    resolver_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"You are a resolver tasked with finding the best answer to the question '{question}'",
            },
            {"role": "user", "content": resolver_prompt},
        ],
    )

    resolver_response = resolver_response["choices"][0]["message"]["content"]

    return resolver_response
