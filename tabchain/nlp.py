import ast
import os
from typing import Literal

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def sentiment_analysis(text: str | list[str] | dict[str, str], mode: Literal["discrete", "continuous"]):
    """
    Assign sentiment to one or more pieces of text. Discrete mode returns a single word out of "positive", "negative",
    or "neutral". Continuous mode returns a number between -1 and 1, where -1 is the most negative, 0 is neutral, and
    1 is the most positive.

    Parameters
    ----------
    text : str | list[str] | dict[str, str]
        The text to assign sentiment to.
    mode : Literal["discrete", "continuous"]
        The mode to use for sentiment analysis. Discrete mode returns a single word out of "positive", "negative",
        or "neutral". Continuous mode returns a number between -1 and 1, where -1 is the most negative, 0 is neutral,
        and 1 is the most positive.

    Returns
    -------
    str | list[str]
        The sentiment of the text.

    """

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
    """
    Summarize a piece of text into a specified number of sentences or a specified percentage of the original text.

    Parameters
    ----------
    text : str
        The text to summarise.
    n_sentences : int, optional
        The number of sentences to summarise the text into.
    percentage : int, optional
        The percentage of the original text to summarise the text into.

    Returns
    -------
    str
        The summarised text.
    """

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
    """
    Translate a piece of text into a specified target language.

    Parameters
    ----------
    text : str
        The text to translate.
    target_language : str
        The target language to translate the text into.

    Returns
    -------
    str
        The translated text.
    """

    user_message = f"Translate the following text into {target_language}: --- {text}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a translator."}, {"role": "user", "content": user_message}],
    )

    return response["choices"][0]["message"]["content"]


def classification(text: list[str], labels: list[str] = None, examples: dict[str, str] = None):
    """
    Text classification.

    Parameters
    ----------
    text : list[str]
        A list of strings to classify.
    labels : list[str], optional
        A list of labels to classify the text into. If not provided, the model will generate labels.
    examples : dict[str, str], optional
        A dictionary of correctly classified examples. The keys are the labels and the values are the examples.

    Returns
    -------
    list[str]
        A list of labels for each piece of text.
    """

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
