import os
import time
import shutil
from pathlib import Path

import click

from ollama_process import process_new_text
from whisper_process import process_new_audio


def check_folder_and_process_uprocessed_files(folder_path: Path, processing_function: callable, output_folder: Path):
    unprocessed_folder = folder_path / 'unprocessed'
    processed_folder = folder_path / 'processed'

    unprocessed_folder.mkdir(parents=True, exist_ok=True)
    processed_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    # List files in the unprocessed folder
    for file_path in unprocessed_folder.iterdir():
        if file_path.name.startswith('.'):
            continue

        # Process the file
        processing_function(file_path, output_folder)

        # Move the file to the processed folder
        shutil.move(file_path, processed_folder / file_path.name)
        print(f"Moved file to processed folder: {file_path.name}")


@click.command()
@click.option('--root-path', default='jobs', help='Root directory for jobs.')
def main(root_path):
    root_path = Path(root_path)
    check_folder_and_process_uprocessed_files(root_path / 'new_audio', process_new_audio,
                                              root_path / 'new_raw_text' / 'unprocessed')
    check_folder_and_process_uprocessed_files(root_path / 'new_raw_text', process_new_text, root_path / 'results')

    # Wait for 2 seconds
    time.sleep(2)


if __name__ == "__main__":
    main()