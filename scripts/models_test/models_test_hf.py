import json
import os
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm  # For progress bar
import logging

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

# #### Step 3: Download model from Hugging Face
def load_model_and_tokenizer(model_repo_id):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
        model = AutoModelForCausalLM.from_pretrained(model_repo_id)

        if torch.cuda.is_available():
            model = model.to("cuda")
            logging.info("Model moved to GPU")
        else:
            logging.warning("GPU not available. Using CPU.")

        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        raise

# #### Step 4: Process the dataset and generate model responses
def generate_responses(data, tokenizer, model, output_json_path):
    results = []
    stats = {
        "total_questions": 0,
        "total_time_seconds": 0,
        "average_time_per_question_seconds": 0,
        "detailed_times": []
    }

    if not os.path.exists(output_json_path):
        with open(output_json_path, "w") as f:
            json.dump([], f)

    with open(output_json_path, "r") as f:
        try:
            existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    processed_ids = {entry["id"] for entry in existing_results}

    total_start_time = time.time()

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
            inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)

            if torch.cuda.is_available():
                inputs = {key: value.to("cuda") for key, value in inputs.items()}

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"], 
                    max_length=2048, 
                    num_beams=5, 
                    early_stopping=True
                )
            end_time = time.time()

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

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
            with open(output_json_path, "w") as f:
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
    with open(stats_output_path, "w") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    logging.info(f"Total time for all generations: {stats['total_time_seconds']:.2f} seconds")
    logging.info(f"Average time per question: {stats['average_time_per_question_seconds']:.2f} seconds")

    return results, stats

# #### Main Script
if __name__ == "__main__":
    DATASET_REPO_ID = "lawful-good-project/sud-resh-benchmark"
    MODEL_REPO_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    OUTPUT_JSON_PATH = "results.json"

    try:
        authenticate_huggingface()
        dataset = load_dataset_from_huggingface(DATASET_REPO_ID)
        tokenizer, model = load_model_and_tokenizer(MODEL_REPO_ID)
        generate_responses(dataset, tokenizer, model, OUTPUT_JSON_PATH)
    except Exception as e:
        logging.error(f"An error occurred: {e}")