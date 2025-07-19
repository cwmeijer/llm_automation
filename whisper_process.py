import os
import time
from pathlib import Path

import click
from faster_whisper import WhisperModel


def process_new_audio(input_file_path: Path, output_folder: Path):
    print(f"Processing file: {input_file_path}")
    model_size = "large-v3"

    # Run on CPU with INT8
    with TimeBlock("Model loading"):
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    with TimeBlock("Transcription"):
        segments, info = model.transcribe(str(input_file_path), beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    with TimeBlock("Saving segments"):
        output_file_path = os.path.join(output_folder, input_file_path.name + ".txt")
        with open(output_file_path, "w") as output_file:
            for segment in segments:
                output_file.write(segment.text)
                print(segment.text)
        print(f"Segments saved to: {output_file_path}")


class TimeBlock:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.label}...")

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        print(f"{self.label} completed in %.2f seconds" % (end_time - self.start_time))


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_folder", type=click.Path(exists=True, writable=True, path_type=Path))
def main(input_file, output_folder):
    """Process audio files and save transcription results."""
    process_new_audio(input_file, output_folder)


if __name__ == "__main__":
    main()
