import ast
import os
from typing import Literal

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def sentiment_analysis(text: str | list[str] | dict[str, str], mode: Literal["discrete", "continuous"]):
    if isinstance(text, str):
        text = [text]

    joined = "[" + ", ".join([f'"{t}"' for t in text]) + "]"
    system_message = "You are a sentiment analyzer."

    if mode == "discrete":
        user_message = f"""For every sentence, tell me if it is positive, negative, or neutral.
            Remember, you can only use one word to describe the sentiment. For example,
            'good' is positive, 'bad' is negative, and 'okay' is neutral. Return the output
            as a list of strings, where each string is either 'positive', 'negative', or
            'neutral'. --- {joined}"""
    elif mode == "continuous":
        user_message = f"""For every sentence, tell me how positive or negative it is. Return the output
            f"as a list of numbers between -1 and 1, where -1 is the most negative, 0 is neutral,
            f"and 1 is the most positive. --- {joined}"""

    else:
        raise ValueError("mode must be 'discrete' or 'continuous'")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
    )

    result = response.choices[0]["message"]["content"]
    result = ast.literal_eval(result)

    return result


def summarisation(text, n_sentences=None, percentage=None):
    if n_sentences:
        user_message = f"Summarize the following text into {n_sentences} sentences: --- {text}"
    elif percentage:
        user_message = f"Summarize the following text into {percentage}% of the original text: --- {text}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a summarizer."}, {"role": "user", "content": user_message}],
    )

    return response["choices"][0]["message"]["content"]


def translation(text, target_language):
    user_message = f"Translate the following text into {target_language}: --- {text}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a translator."}, {"role": "user", "content": user_message}],
    )

    return response["choices"][0]["message"]["content"]


def classification(text: list[str], labels: list[str] = None, examples: dict[str, str] = None):
    label_addon, example_addon = "", ""

    if labels:
        joined_labels = "[" + ", ".join([f'"{t}"' for t in labels]) + "]"
        label_addon = f"into one of the following categories: {joined_labels}"

    if examples:
        joined_examples = "[" + ", ".join([f'"{k} : {v}"' for k, v in examples.items()]) + "]"
        example_addon = f"using the following correctly classified examples: {joined_examples}"

    user_message = f"Classify the following text {label_addon} {example_addon} Return your answer \
        as a list of labels. --- {text}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a classifier."}, {"role": "user", "content": user_message}],
    )

    return ast.literal_eval(response["choices"][0]["message"]["content"])
