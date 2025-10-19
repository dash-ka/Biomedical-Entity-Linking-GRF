from openai import OpenAI
import os
from pathlib import Path

SELECT_PROMPT = """
You are an expert biocurator. Your task is to map an entity mention to the correct concept. You can use the context to disambiguate the meaning of the entity mention."
Context: "{context}"
Select which of the following concepts best represents the entity mention: "{mention}".
Concepts:
{concepts}

If uncertain between two concepts keep both. 
Format the output as the following JSON {{"id": <a list of selected concept identifiers.>}}
"""

API_KEY = "YOUR_OPENAI_API"
os.environ["OPENAI_API_KEY"] = API_KEY

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
def gen_selection(user):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        temperature=1,
        messages=[
            {"role": "user", "content": user}
            ]
        )
    return response.choices[0].message.content


main_folder = Path("submission/datasets/nlm-gene")

# load queries
with open(main_folder / "raw_queries.json", encoding="utf-8") as file:
    queries = json.load(file)

# load predicted_rankings
with open(main_folder / "predictions_grf_vector_aio.json") as file:
    predictions = json.load(file)

CUTOFF = 10
for query, candidates in tqdm(zip(queries, predictions), total=len(queries)):
    query_text = query.get("entity_text")
    context = query.get("sentence", "")
    concept_names = [candidates[c]["name"] for c in list(candidates)[:CUTOFF]]
    concept_ids = [candidates[c]["cui"].split(":")[-1] for c in list(candidates)[:CUTOFF]]
    concept_identifiers = "\n".join([": ".join(p) for p in zip(concept_ids, concept_names)])
     
    USER_MESSAGE = SELECT_PROMPT.format(context=context, mention=query_text, concepts=concept_identifiers)
    response = gen_selection(USER_MESSAGE)
    query["gpt4o_10"] = response

with open()
