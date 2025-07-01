import os
from pathlib import Path

import ollama


def process_new_text(input_file_path: Path, output_folder: Path):
    print(f"Processing file with ollama: {input_file_path}")
    # Read and process the new file
    with open(input_file_path, 'r') as file:
        content = file.read()
    result = ollama.generate(model='gemma3:1b', prompt=content)

    # Save the result
    result_file_path = output_folder / input_file_path.name
    with open(result_file_path, 'w') as result_file:
        result_file.write(str(result.response))

    print(f"Result saved to: {result_file_path}")