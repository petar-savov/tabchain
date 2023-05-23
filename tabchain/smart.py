import ast
import os
from typing import Literal

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from tabchain.nlp import question_answering


def question_answering(question, temperature=0.0):
    user_message = f"Let's work this out in a step by step way to make sure that\
          we have the right answer. --- {question}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a question answering model."},
            {"role": "user", "content": user_message},
        ],
    )

    return response["choices"][0]["message"]["content"]


def smart_gpt(question):
    candidate_answers = [question_answering(question) for _ in range(3)]

    reflexion_prompt = f"List the flaws and faulty logic in each of the responses. Let's work this out\
        in a step by step way to make sure that we have all the errors. --- {candidate_answers}"

    reflexion_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a researcher tasked with investigating the 3 responses to the question: '{question}'.",
            },
            {"role": "user", "content": reflexion_prompt},
        ],
    )

    reflexion_response = reflexion_response["choices"][0]["message"]["content"]

    resolver_prompt = f"You are a resolver tasked with (1) finding which 3 responses to the question '{question}' the researcher\
        thought as best (2) improving the answer, (3) printing the improved answer in full and only printing the improved answer.\
        Let's work this out in a step by step way to make sure that we have the right answer. Don't indicate that \
        this is a revised or improved answer, phrase it as if it's the original answer. --- {reflexion_response}"

    resolver_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a resolver."},
            {"role": "user", "content": resolver_prompt},
        ],
    )

    resolver_response = resolver_response["choices"][0]["message"]["content"]

    return resolver_response
