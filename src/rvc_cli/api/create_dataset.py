"""
Audio File Processor

This script provides functionality to process large audio files using ffmpeg.
It can extract specific clips based on an fspec string or split the audio into
fixed-size chunks.

Sample usage:

    import os
    from audio_processor import process_audio_file

    input_file = "/path/to/large_audio_file.mp3"
    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)

    # Option 1: Use fspec to extract specific clips
    fspec = '''
    0       28800   part_1
    28800   57600   part_2
    57600   86400   part_3
    '''
    process_audio_file(input_file, output_dir, fspec=fspec)

    # Option 2: Split into fixed-size chunks (e.g., 60-second chunks)
    process_audio_file(input_file, output_dir, chunk_size=60)
"""

import subprocess
from pathlib import Path
from typing import List, Tuple, Union


def parse_fspec(fspec_string: str) -> List[Tuple[float, float, str]]:
    """
    Parse an fspec string into a list of (start, end, name) tuples.

    :param fspec_string: A string containing the fspec information
    :type fspec_string: str
    :return: A list of tuples containing (start_time, end_time, clip_name)
    :rtype: List[Tuple[float, float, str]]
    """
    specs = []
    for line in fspec_string.strip().split("\n"):
        parts = line.split()
        if len(parts) == 3:
            start, end, name = parts
            specs.append((float(start), float(end), name))
    return specs


def extract_clip(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    start: float,
    duration: float,
) -> None:
    """
    Extract a clip from the input file using ffmpeg.

    :param input_file: Path to the input audio file
    :type input_file: Union[str, Path]
    :param output_file: Path to the output audio file
    :type output_file: Union[str, Path]
    :param start: Start time of the clip in seconds
    :type start: float
    :param duration: Duration of the clip in seconds
    :type duration: float
    """
    cmd = [
        "ffmpeg",
        "-ss",
        str(start),
        "-i",
        str(input_file),
        "-t",
        str(duration),
        "-c",
        "copy",
        "-y",  # Overwrite output file if it exists
        str(output_file),
    ]
    subprocess.run(cmd, check=True)


def process_audio_file(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    fspec: str = None,
    chunk_size: int = None,
) -> None:
    """
    Process an audio file by either extracting specific clips based on fspec or
    splitting it into fixed-size chunks.

    :param input_file: Path to the input audio file
    :type input_file: Union[str, Path]
    :param output_dir: Path to the output directory
    :type output_dir: Union[str, Path]
    :param fspec: fspec string defining clip extraction, defaults to None
    :type fspec: str, optional
    :param chunk_size: Size of chunks in seconds for splitting, defaults to None
    :type chunk_size: int, optional
    :raises ValueError: If neither fspec nor chunk_size is provided
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if fspec:
        specs = parse_fspec(fspec)
        for start, end, name in specs:
            output_file = output_path / f"{name}.mp3"
            duration = end - start
            extract_clip(input_path, output_file, start, duration)
            print(f"Extracted: {output_file}")
    elif chunk_size:
        # Get the duration of the input file
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())

        chunk_index = 0
        for start in range(0, int(duration), chunk_size):
            chunk_index += 1
            output_file = output_path / f"chunk_{chunk_index}_{input_path.stem}.mp3"
            extract_clip(input_path, output_file, start, chunk_size)
            print(f"Extracted: {output_file}")
    else:
        raise ValueError("Either 'fspec' or 'chunk_size' must be provided")
