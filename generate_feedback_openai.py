import os, json, argparse
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path


USER_PROMPT_SYNONYMS="""
You are an expert biocurator. Given the context: "{}", specify twenty exact synonyms of the concept **{}**. These should be either alternative names, morphological variants, or terminological equivalents referring to the same underlying concept (not broader or related terms).
Return your answer as a JSON object in the following format:
{{"synonyms": <exact synonyms here separated by a colon (e.g., "synonym1:synonym2:synonym3:synonym4:synonym5 etc.")>}}

If the concept is an acronym, always expand it by including the full name in parentheses.
"""

USER_PROMPT_DEF="""
You are an expert biocurator. Given the context: "{}", specify a single sentence definition of the concept **{}**. 
Return your answer as a JSON object in the following format:
{{"definition": <"a single sentence definition of the concept here">}}
"""

USER_PROMPT_STANDARD_NAME="""
You are an expert biocurator. Given the context: "{}", specify the standard concept name from {} for the concept **{}**. 
Return your answer as a JSON object in the following format:
{{"name": <"a standard community-recognized concept name here">}}
"""

dataset2terminology = {
     "nlm-gene": "the NCBI Gene terminology",
     "gnormplus": "the NCBI Gene terminology",
     "s800": "the NCBI Taxonomy terminology",
     "linnaeus": "the NCBI Taxonomy terminology",
     "ncbi-disease": "the MeSH disease terminology",
     "bc5cdr-disease": "the MeSH disease terminology",
     "bc5cdr-chemical": "the MeSH chemical terminology",
     "nlm-chemical": "the MeSH chemical terminology"
}

def generate_feedback(client, model_name, user_message, feedback_type):
        response = client.chat.completions.create(
            model=model_name,
            temperature=1,
            messages=[
                {"role": "user", "content": user_message}
                ],
            response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "entities",
                        "schema": {
                            "type": "object",
                            "properties": {
                                feedback_type: {"type": "string"}
                            },
                            "required": [feedback_type],
                            "additionalProperties": False
                        },
                    "strict": True
                    }
                }
            )

        return response.choices[0].message.content


def run(client, model_name, queries, target_terminology):
        
        for feedback_type in ["synonyms", "definition", "standard_name"]:  
            print(f"\n=== Generating {feedback_type} ===")
            for idx, item in tqdm(enumerate(queries), total=len(queries)):
                s, term = item["sentence"], item["entity_text"]
                
                if feedback_type == "synonyms":
                    user_message = USER_PROMPT_SYNONYMS.format(s, term)
                
                elif feedback_type == "definition":
                    user_message = USER_PROMPT_DEF.format(s, term)
                    
                elif feedback_type == "standard_name":
                    user_message = USER_PROMPT_STANDARD_NAME.format(s, target_terminology, term)
                    
                else:
                    raise ValueError(f"Unknown strategy: {feedback_type}")
            
                try:
                    response = generate_feedback(client, model_name, user_message, feedback_type=feedback_type)
                    response = json.loads(response)
                    
                    if feedback_type == "synonyms":
                        if len(response["synonyms"].split(":")) < 11:
                            print(f"Query {idx}: Generated ONLY {len(response['synonyms'].split(':'))} synonyms for {term}\n:", response["synonyms"], "\nRetrying...")
                            response2 = generate_feedback(client, model_name, user_message, feedback_type=feedback_type)
                            response2 = json.loads(response2)
                            if len(response2["synonyms"].split(":")) < 11:
                                print("Still not enough synonyms generated, saving anyway.")
                                joined =  ":".join([response["synonyms"].strip(":"), response2["synonyms"].strip(":")])
                                print("JOINED:", len(joined.split(":")))
                                item["synonyms"] = joined
                            else:
                                item["synonyms"] = response2["synonyms"]

                        else:
                            item["synonyms"] = response["synonyms"]
                            success_rate+=1

                    elif feedback_type == "definition":
                        if not response["definition"].strip():
                            print(f"Query {idx}: No definition generated for {term}\n:", response["definition"], "\nRetrying...")
                            response = generate_feedback(client, model_name, user_message, feedback_type=feedback_type)
                            response = json.loads(response)
                            if not response["definition"].strip():
                                print("Still no definition generated, saving anyway.")
                        item["definition"] = response["definition"]

                    elif feedback_type == "standard_name":
                        if not response["name"].strip():
                            print(f"Query {idx}: No definition generated for {term}\n:", response["name"], "\nRetrying...")
                            response = generate_feedback(client, model_name, user_message, feedback_type=feedback_type)
                            response = json.loads(response)
                            if not response["name"].strip():
                                print("Still no standard name generated, saving anyway.")
                        item["standard_name"] = response["name"]
                    else:
                        raise ValueError(f"Unknown strategy: {feedback_type}")
                        
                except Exception as e:
                        print(f"Error processing term: {term}")
                        print(f"Response: {response}")
                        print(e)
                        continue
            
            return queries


def main(args):
    model_name = args.get("model_name")
    dataset_name = args.get("dataset_name")
    queries_path = args.get("queries_path")
        
    # read queries
    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    # generate feedback
    client = OpenAI(api_key=args.get("OPENAI_API_KEY"))
    target_terminology = dataset2terminology.get(dataset_name)
    queries_with_feedback = run(client, model_name, queries, target_terminology)

    # write queries with feedback
    output_dir = Path(queries_path).parent / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "queries_with_feedback.json", "w", encoding="utf-8") as f:
        json.dump(queries_with_feedback, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--OPENAI_API_KEY", type=str, required=True, help="Your OpenAI API key.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to JSON file with queries.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=[
        "ncbi-disease", "bc5cdr-disease", "bc5cdr-chemical", "nlm-chemical",
        "nlm-gene", "gnormplus", "s800", "linnaeus"
    ])
    args = parser.parse_args()
    main(args)

    

    
