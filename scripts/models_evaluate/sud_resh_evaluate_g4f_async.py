import asyncio
import json
import os
import logging
import nest_asyncio
import shutil
import signal
from tqdm.asyncio import tqdm
from typing import List, Dict
from g4f.client import AsyncClient
from aiolimiter import AsyncLimiter
from datetime import datetime

# Apply the nest_asyncio patch to handle nested event loops
nest_asyncio.apply()

# ----------------------------------------------------------------------
# Configuration and Constants
# ----------------------------------------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to stderr
        logging.FileHandler("script.log", encoding="utf-8")  # Log to file
    ]
)
logger = logging.getLogger(__name__)

# Define the maximum number of concurrent tasks
MAX_CONCURRENT_TASKS = 20
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# Initialize a lock for saving progress
save_lock = asyncio.Lock()

# Rate Limiting Configuration
# Adjust the "max_calls" and "period" according to the API's rate limits
RATE_LIMIT = 60  # Maximum number of API calls
RATE_PERIOD = 60  # Time period in seconds
limiter = AsyncLimiter(max_rate=RATE_LIMIT, time_period=RATE_PERIOD)

# Flag to indicate if a shutdown has been requested
shutdown_requested = False

# ----------------------------------------------------------------------
# Backup Mechanism
# ----------------------------------------------------------------------

def create_backup(file_path: str):
    """
    Creates a timestamped backup of the specified file.
    """
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = f"{file_path}.backup.{timestamp}"
        try:
            shutil.copy(file_path, backup_path)
            logger.info(f"Backup created: '{backup_path}'")
        except Exception as e:
            logger.error(f"Failed to create backup for '{file_path}': {e}")

# ----------------------------------------------------------------------
# Graceful Shutdown Handler
# ----------------------------------------------------------------------

async def handle_shutdown(signal_number, frame):
    """
    Handles shutdown signals to gracefully terminate the script.
    """
    global shutdown_requested
    if not shutdown_requested:
        logger.info(f"Received shutdown signal ({signal_number}). Saving progress and exiting...")
        shutdown_requested = True
    else:
        logger.warning("Shutdown already in progress. Please wait...")

# ----------------------------------------------------------------------
# Processing Functions
# ----------------------------------------------------------------------

async def process_entry(
    client: AsyncClient,
    entry: dict,
    prompt_template: str,
    output_data_map: Dict[str, dict],
    output_json_filename: str
):
    """
    Process a single entry by formatting the prompt, sending it to the model,
    and parsing the response. Adds an 'evaluation' field to the entry.
    Saves progress after processing.
    """
    entry_id = entry.get("id")
    if not entry_id:
        logger.warning(f"Entry without 'id' found. Skipping entry: {entry}")
        return

    # Check if the entry has already been evaluated
    if entry_id in output_data_map:
        logger.debug(f"Entry with id {entry_id} already evaluated. Skipping.")
        return

    async with semaphore:  # Limit concurrency using the semaphore
        try:
            question = entry.get("question")
            answer = entry.get("answer")
            correct_answer = entry.get("correct_answer")

            if not all([question, answer, correct_answer]):
                logger.warning(f"Missing fields in entry ID {entry_id}. Skipping.")
                evaluation = {
                    "accuracy": None,
                    "completeness": None,
                    "clarity": None,
                    "comment": "Missing required fields."
                }
            else:
                # Format the prompt with the question, answer, and correct_answer
                formatted_prompt = prompt_template.format(
                    question=question,
                    answer=answer,
                    correct_answer=correct_answer
                )

                # Respect rate limiting
                async with limiter:
                    # Make the chat completion request
                    response = await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": formatted_prompt}],
                        stream=False  # We want a complete response at once
                    )

                # Extract the text response from the model
                if response.choices and response.choices[0].message.content:
                    evaluation_text = response.choices[0].message.content
                else:
                    evaluation_text = "No response generated by the model."

                logger.debug(f"Evaluation Text for entry ID {entry_id}: {evaluation_text}")

                # Parse the evaluation text into the required structure
                evaluation = parse_evaluation_text(evaluation_text)

            # Insert the evaluation into the output data map
            output_data_map[entry_id] = {**entry, "evaluation": evaluation}

            # Asynchronously save progress
            async with save_lock:
                await save_progress(output_json_filename, output_data_map)

            # Check if shutdown has been requested
            if shutdown_requested:
                raise asyncio.CancelledError

        except asyncio.CancelledError:
            logger.info(f"Processing of entry ID {entry_id} was cancelled.")
            raise
        except Exception as e:
            logger.error(f"Error processing entry ID {entry_id}: {e}")
            evaluation = {
                "accuracy": None,
                "completeness": None,
                "clarity": None,
                "comment": f"Processing error: {e}"
            }
            output_data_map[entry_id] = {**entry, "evaluation": evaluation}
            async with save_lock:
                await save_progress(output_json_filename, output_data_map)

def parse_evaluation_text(evaluation_text: str) -> dict:
    """
    Parse the evaluation text returned by the model into a structured dictionary.
    Supports JSON and line-based formats.
    """
    try:
        # Attempt to parse the response as JSON
        evaluation_data = json.loads(evaluation_text)
        evaluation = {
            "accuracy": evaluation_data.get("accuracy"),
            "completeness": evaluation_data.get("completeness"),
            "clarity": evaluation_data.get("clarity"),
            "comment": evaluation_data.get("comment"),
        }
    except json.JSONDecodeError:
        # If the response is not JSON, fall back to line-based parsing
        try:
            evaluation_lines = evaluation_text.strip().split("\n")
            evaluation = {
                "accuracy": int(evaluation_lines[0].split(":")[1].strip()),
                "completeness": int(evaluation_lines[1].split(":")[1].strip()),
                "clarity": int(evaluation_lines[2].split(":")[1].strip()),
                "comment": evaluation_lines[3].split(":")[1].strip(),
            }
        except (IndexError, ValueError) as e:
            # Handle cases where the response is not in the expected format
            logger.error(f"Error parsing evaluation text: {evaluation_text}. Error: {e}")
            evaluation = {
                "accuracy": None,
                "completeness": None,
                "clarity": None,
                "comment": "Parsing error: response format invalid."
            }
    return evaluation

async def save_progress(output_json_filename: str, data_map: Dict[str, dict]):
    """
    Save the current progress to the output JSON file.
    The data_map is a dictionary mapping entry IDs to their data.
    """
    try:
        # Convert the data_map back to a list
        data_list = list(data_map.values())

        # Write to a temporary file first for atomicity
        temp_filename = f"{output_json_filename}.tmp"
        with open(temp_filename, "w", encoding="utf-8") as out_file:
            json.dump(data_list, out_file, ensure_ascii=False, indent=4)

        # Replace the original file with the temporary file
        shutil.move(temp_filename, output_json_filename)

        logger.info(f"Progress saved to '{output_json_filename}'. Total evaluated: {len(data_map)}")
    except Exception as e:
        logger.error(f"Failed to save progress to '{output_json_filename}': {e}")


async def load_existing_output(output_json_filename: str) -> Dict[str, dict]:
    """
    Load existing output JSON data into a dictionary mapping entry IDs to data.
    If the file does not exist, return an empty dictionary.
    """
    if os.path.exists(output_json_filename):
        try:
            with open(output_json_filename, "r", encoding="utf-8") as out_file:
                existing_data = json.load(out_file)
            # Create a map for quick lookup
            data_map = {entry["id"]: entry for entry in existing_data if "id" in entry}
            logger.info(f"Loaded existing output from '{output_json_filename}'. Entries already evaluated: {len(data_map)}")
            return data_map
        except Exception as e:
            logger.error(f"Error loading existing output from '{output_json_filename}': {e}")
            return {}
    else:
        logger.info(f"No existing output file found at '{output_json_filename}'. Starting fresh.")
        return {}

# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------

async def main():
    """
    Main function to process the input JSON file, evaluate entries, and save the results.
    """
    global shutdown_requested
    try:
        # ----------------------------------------------------------------------
        # 1. Register Shutdown Signals
        # ----------------------------------------------------------------------
        loop = asyncio.get_running_loop()
        signals = (signal.SIGINT, signal.SIGTERM)
        for s in signals:
            try:
                loop.add_signal_handler(s, lambda s=s: asyncio.create_task(handle_shutdown(s, None)))
            except NotImplementedError:
                # Signal handlers are not implemented on some platforms (e.g., Windows)
                logger.warning(f"Signal handling for {s} is not implemented on this platform.")

        # ----------------------------------------------------------------------
        # 2. Read Input JSON File
        # ----------------------------------------------------------------------
        input_json_filename = "sud_resh_answers_gpt-4o.json"  # <-- replace with your actual JSON filename
        if not os.path.exists(input_json_filename):
            raise FileNotFoundError(f"Input JSON file '{input_json_filename}' not found.")

        with open(input_json_filename, "r", encoding="utf-8") as json_file:
            input_data = json.load(json_file)

        if not isinstance(input_data, list):
            raise ValueError("Input JSON file must contain a list of entries.")

        total_entries = len(input_data)
        logger.info(f"Total entries in input file: {total_entries}")

        # ----------------------------------------------------------------------
        # 3. Load Prompt Template from a .txt File
        # ----------------------------------------------------------------------
        prompt_template_filename = "prompt_template.txt"  # <-- replace with your actual template filename
        if not os.path.exists(prompt_template_filename):
            raise FileNotFoundError(f"Prompt template file '{prompt_template_filename}' not found.")

        with open(prompt_template_filename, "r", encoding="utf-8") as file:
            prompt_template = file.read()

        # ----------------------------------------------------------------------
        # 4. Determine the output filename based on 'test_model' value
        # ----------------------------------------------------------------------
        if input_data:
            test_model_value = input_data[0].get("test_model", "default_model")
            output_json_filename = f"sud_resh_{test_model_value}_evaluation.json"
        else:
            raise ValueError("Input JSON file is empty. Nothing to process.")

        # ----------------------------------------------------------------------
        # 5. Load Existing Output Data if Available
        # ----------------------------------------------------------------------
        output_data_map = await load_existing_output(output_json_filename)
        num_already_evaluated = len(output_data_map)
        logger.info(f"Entries already evaluated: {num_already_evaluated}")

        # ----------------------------------------------------------------------
        # 6. Create an Async G4F Client
        # ----------------------------------------------------------------------
        client = AsyncClient()  # Initialize the async client

        # ----------------------------------------------------------------------
        # 7. Identify Entries That Need Processing
        # ----------------------------------------------------------------------
        entries_to_process = [
            entry for entry in input_data
            if "id" in entry and entry["id"] not in output_data_map
        ]
        num_to_process = len(entries_to_process)
        logger.info(f"Entries to process (excluding already evaluated): {num_to_process}")

        if not entries_to_process:
            logger.info("All entries have already been evaluated. No processing needed.")
            return

        # ----------------------------------------------------------------------
        # 8. Process Entries Concurrently
        # ----------------------------------------------------------------------
        # Create a list of coroutines for processing
        tasks = [
            asyncio.create_task(process_entry(client, entry, prompt_template, output_data_map, output_json_filename))
            for entry in entries_to_process
        ]

        # Use tqdm to show progress
        try:
            for task in tqdm(asyncio.as_completed(tasks), total=num_to_process, desc="Processing entries"):
                await task
        except asyncio.CancelledError:
            logger.info("Shutdown requested. Cancelling remaining tasks...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("All tasks cancelled.")

        # ----------------------------------------------------------------------
        # 9. Final Save (optional, in case some entries were not saved)
        # ----------------------------------------------------------------------
        async with save_lock:
            await save_progress(output_json_filename, output_data_map)

        logger.info(f"Evaluation complete. Results saved to '{output_json_filename}'. Total evaluated: {len(output_data_map)}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Final save in case of unexpected exit
        if 'output_json_filename' in locals() and 'output_data_map' in locals():
            async with save_lock:
                await save_progress(output_json_filename, output_data_map)
        logger.info("Script terminated.")

# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Use asyncio.run() in standard Python environments
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
