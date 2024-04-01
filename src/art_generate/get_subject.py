import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
import pandas as pd

#thank you armaan for this code
csv_file = "1k.csv"
df = pd.read_csv(csv_file)
prompts = df["prompt"].tolist()


def fetch_response(prompt):
    system_message = (
        "Immediate action required: You need to simplify a text-to-image prompt. "
        "It is critical to eliminate all non-essential elements, particularly any stylistic components, artist names, and irrelevant details. "
        "Condense the prompt into one precise sentence, focusing exclusively on the indispensable subjects and actions. "
        "Under no circumstances should style components, mentions of artists, image styles, camera settings, or metadata like size and resolution be included in the summary. "
        "The word style can NEVER be used in the response. "
        "Your task is to extract and present only the fundamental subjects and actions. "
        "Components of style, vibe, lighting should strictly be avoided. We are looking for a clear, concise, and direct summary of the prompt. "
        "Non-compliance with these instructions will result in complete disregard of the unnecessary elements. Here is the prompt:"
    )

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Please remember the task: " + system_message},
            {"role": "user", "content": "```" + str(prompt) + "```"},
        ],
        stop=["\n"],
        max_tokens=100,
        temperature=0.0,
    )

    return response.choices[0].message.content


with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(fetch_response, prompt) for prompt in prompts]
    responses = [future.result() for future in as_completed(futures)]

new_df = pd.DataFrame()
new_df["response"] = responses

new_df.to_csv("1k_with_responses.csv", index=False)
