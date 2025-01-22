import json
import os
import time
from datasets import load_dataset
from tqdm import tqdm  # For progress bar
import logging
from g4f.client import Client  # Importing g4f client for API usage

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# #### Step 1: Authenticate with Hugging Face
def authenticate_huggingface():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    logging.info("Authenticated with Hugging Face successfully.")

# #### Step 2: Load dataset from Hugging Face
def load_dataset_from_huggingface(dataset_repo_id):
    try:
        dataset = load_dataset(dataset_repo_id, split="train")
        logging.info(f"Dataset loaded with {len(dataset)} entries")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# #### Step 3: Generate responses using g4f Client
def generate_responses_with_g4f(data, output_json_path):
    results = []
    stats = {
        "total_questions": 0,
        "total_time_seconds": 0,
        "average_time_per_question_seconds": 0,
        "detailed_times": []
    }

    # Ensure the output file exists and is properly encoded
    if not os.path.exists(output_json_path):
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)

    # Load existing results
    with open(output_json_path, "r", encoding="utf-8") as f:
        try:
            existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    processed_ids = {entry["id"] for entry in existing_results}

    total_start_time = time.time()

    client = Client()  # Initialize the g4f client

    for entry in tqdm(data, desc="Processing entries", unit="entry"):
        question_id = entry["id"]
        category = entry["category"]
        model_name = entry["model_name"]
        instruction = entry["instruction"]
        text = entry["text"]
        correct_answer = entry["correct_answer"]

        if question_id in processed_ids:
            continue

        try:
            question = f"{instruction}\n\n{text}"

            start_time = time.time()
            # Generate response using g4f client
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Specify the model
                messages=[{"role": "user", "content": question}],
                web_search=False  # Disable web search
            )
            end_time = time.time()

            # Extract the generated text
            answer = response.choices[0].message.content.strip()

            # Handle malformed server responses
            if isinstance(answer, dict):
                logging.warning(f"Malformed response for ID {question_id}: {answer}")
                answer = "Malformed response received from the server."

            # Check if the model repeats the question
            if answer == question.strip():
                logging.warning(f"Model repeated the question for ID {question_id}. Generating fallback response.")
                answer = "The model was unable to generate a distinct response. Please review the input."

            generation_time = end_time - start_time
            stats["detailed_times"].append({"id": question_id, "time_seconds": generation_time})

            result = {
                "id": question_id,
                "category": category,
                "test_model": model_name,
                "question": question,
                "answer": answer,
                "correct_answer": correct_answer,
                "generation_time_seconds": generation_time
            }
            results.append(result)

            existing_results.append(result)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, indent=4, ensure_ascii=False)

            logging.info(f"Processed question ID: {question_id} in {generation_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Error processing question ID {question_id}: {e}")
            continue

    total_end_time = time.time()
    stats["total_questions"] = len(stats["detailed_times"])
    stats["total_time_seconds"] = total_end_time - total_start_time
    stats["average_time_per_question_seconds"] = (
        stats["total_time_seconds"] / stats["total_questions"]
        if stats["total_questions"] > 0
        else 0
    )

    stats_output_path = output_json_path.replace(".json", "_stats.json")
    with open(stats_output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    logging.info(f"Total time for all generations: {stats['total_time_seconds']:.2f} seconds")
    logging.info(f"Average time per question: {stats['average_time_per_question_seconds']:.2f} seconds")

    return results, stats

# #### Main Script
if __name__ == "__main__":
    DATASET_REPO_ID = "lawful-good-project/sud-resh-benchmark"
    OUTPUT_JSON_PATH = "results.json"

    try:
        authenticate_huggingface()
        dataset = load_dataset_from_huggingface(DATASET_REPO_ID)
        generate_responses_with_g4f(dataset, OUTPUT_JSON_PATH)
    except Exception as e:
        logging.error(f"An error occurred: {e}")