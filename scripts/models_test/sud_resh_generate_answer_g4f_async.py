#!/usr/bin/env python3

import os
import json
import time
import asyncio
import logging
from typing import List, Tuple, Dict, Any
from datasets import load_dataset
from tqdm.asyncio import tqdm  # Use tqdm.asyncio for asynchronous progress bar
from g4f.client import AsyncClient
import backoff  # For implementing retry mechanism

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def normalize_id(question_id: str) -> str:
    """
    Normalize the question ID.
    Adjust this function to ensure IDs remain consistent.
    """
    return str(question_id).strip()


def authenticate_huggingface():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    logging.info("Authenticated with Hugging Face successfully.")


def load_dataset_from_huggingface(dataset_repo_id: str):
    try:
        dataset = load_dataset(dataset_repo_id, split="train")
        logging.info(f"Dataset loaded with {len(dataset)} entries")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise


def remove_duplicate_ids(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate entries based on their 'id' field.
    Keeps the first occurrence of each ID.
    """
    seen_ids = set()
    unique_data = []
    for entry in data:
        entry_id = normalize_id(entry["id"])
        if entry_id not in seen_ids:
            seen_ids.add(entry_id)
            unique_data.append(entry)
        else:
            logging.warning(f"Duplicate ID found and removed from output: {entry_id}")
    return unique_data


def validate_output(output_json_path: str, input_unique_ids: set) -> bool:
    """
    Validate that the output JSON contains all unique input IDs and no duplicates.
    """
    with open(output_json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Output JSON is malformed: {e}")
            return False

    output_ids = {normalize_id(entry["id"]) for entry in data if "id" in entry}
    missing_ids = input_unique_ids - output_ids
    duplicate_ids = len(data) != len(output_ids)

    if duplicate_ids:
        logging.warning("Duplicate IDs found in the output file. They will be removed.")
        return False

    if not missing_ids:
        logging.info("All unique input IDs are present in the output file.")
        return True
    else:
        logging.error(f"Missing IDs in the output file: {missing_ids}")
        return False


@backoff.on_exception(
    backoff.expo,
    (asyncio.TimeoutError, ConnectionError, Exception),
    max_tries=5,
    jitter=backoff.full_jitter,
    giveup=lambda e: isinstance(e, ValueError),  # Do not retry on ValueError
)
async def fetch_response(client: AsyncClient, model_name: str, question: str):
    """
    Fetch response from the API with retry mechanism.
    """
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question}],
        web_search=False
    )
    return response


async def process_entry(
    entry: Dict[str, Any],
    client: AsyncClient,
    processed_ids: set,
    existing_results: List[Dict[str, Any]],
    lock: asyncio.Lock,
    sem: asyncio.Semaphore,
    stats: Dict[str, Any],
    output_json_path: str,
    model_name: str
):
    raw_question_id = entry["id"]
    question_id = normalize_id(raw_question_id)
    category = entry.get("category", "")
    instruction = entry.get("instruction", "")
    text = entry.get("text", "")
    correct_answer = entry.get("correct_answer", "")

    if question_id in processed_ids:
        logging.info(f"ID {question_id} is already processed. Skipping.")
        return

    async with sem:
        try:
            question = f"{instruction}\n\n{text}"
            start_time = time.time()
            response = await fetch_response(client, model_name, question)
            end_time = time.time()
            answer = response.choices[0].message.content.strip() if response.choices else ""
            if isinstance(answer, dict):
                answer = "Malformed response received from the server."
            if answer == question.strip():
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
            async with lock:
                existing_results.append(result)
                processed_ids.add(question_id)
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(existing_results, f, indent=4, ensure_ascii=False)
            logging.info(f"Processed question ID: {question_id} in {generation_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Error processing question ID {question_id}: {e}")


async def generate_responses_with_g4f_async(data: List[Dict[str, Any]], model_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output_folder = "output"
    json_folder = os.path.join(output_folder, "jsons")
    stats_folder = os.path.join(output_folder, "stats")
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)
    output_json_path = os.path.join(json_folder, f"sud_resh_answers_{model_name}.json")
    stats_output_path = os.path.join(stats_folder, f"sud_resh_answers_{model_name}_stats.json")

    stats = {
        "total_questions": 0,
        "total_time_seconds": 0,
        "average_time_per_question_seconds": 0,
        "detailed_times": []
    }

    # Initialize output JSON file if it doesn't exist
    if not os.path.exists(output_json_path):
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)

    # Load existing results
    with open(output_json_path, "r", encoding="utf-8") as f:
        try:
            existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    # Remove duplicate IDs from existing results
    existing_results = remove_duplicate_ids(existing_results)

    # Save the deduplicated output
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=4, ensure_ascii=False)

    # Extract unique input IDs
    input_unique_ids = {normalize_id(entry["id"]) for entry in data}
    logging.info(f"Total unique input IDs: {len(input_unique_ids)}")

    # Extract already processed IDs
    processed_ids = {normalize_id(entry["id"]) for entry in existing_results if entry.get("id")}
    logging.info(f"Already processed IDs: {len(processed_ids)}")

    total_start_time = time.time()
    client = AsyncClient()
    lock = asyncio.Lock()
    # Get semaphore limit from environment variable or default to 20
    max_concurrent = int(os.getenv("MAX_CONCURRENT_REQUESTS", 60))
    sem = asyncio.Semaphore(max_concurrent)
    max_passes = 5  # Increased to allow multiple reprocessing attempts

    attempt = 0
    while attempt < max_passes:
        attempt += 1
        # Determine which IDs still need to be processed
        remaining_ids = input_unique_ids - processed_ids
        if not remaining_ids:
            logging.info("All unique input IDs have been processed.")
            break
        logging.info(
            f"Attempt {attempt}/{max_passes}: {len(remaining_ids)} IDs remaining to process."
        )
        # Prepare data for the remaining IDs
        remaining_data = [entry for entry in data if normalize_id(entry["id"]) in remaining_ids]

        tasks = []
        for entry in remaining_data:
            task = asyncio.create_task(
                process_entry(
                    entry,
                    client,
                    processed_ids,
                    existing_results,
                    lock,
                    sem,
                    stats,
                    output_json_path,
                    model_name
                )
            )
            tasks.append(task)

        # Use tqdm for asynchronous tasks
        for _ in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing entries", unit="entry"):
            await _

        # Remove any potential duplicates after processing
        existing_results = remove_duplicate_ids(existing_results)

        # Save the deduplicated output
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=4, ensure_ascii=False)

        logging.info(f"After attempt {attempt}, {len(processed_ids)} IDs have been processed.")

    total_end_time = time.time()
    stats["total_questions"] = len(processed_ids)
    stats["total_time_seconds"] = total_end_time - total_start_time
    stats["average_time_per_question_seconds"] = (
        stats["total_time_seconds"] / stats["total_questions"]
        if stats["total_questions"] > 0
        else 0
    )

    with open(stats_output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    logging.info(f"Total time for all generations: {stats['total_time_seconds']:.2f} seconds")
    logging.info(f"Average time per question: {stats['average_time_per_question_seconds']:.2f} seconds")
    logging.info(f"Total processed questions: {stats['total_questions']}")

    # Final validation
    is_valid = validate_output(output_json_path, expected_count=len(input_unique_ids))
    if not is_valid:
        logging.error("Output JSON does not meet the expected criteria. Some IDs might be missing or duplicated.")
    else:
        logging.info("Output JSON validation successful.")

    return existing_results, stats


async def main():
    DATASET_REPO_ID = "lawful-good-project/sud-resh-benchmark"
    MODEL_NAME = "llama-3.1-70b"
    try:
        authenticate_huggingface()
        dataset = load_dataset_from_huggingface(DATASET_REPO_ID)
        # Convert dataset to list of dictionaries
        data = [dict(entry) for entry in dataset]
        await generate_responses_with_g4f_async(data, MODEL_NAME)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise
