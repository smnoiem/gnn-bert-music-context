"""
Download the clips within the MusicCaps dataset from YouTube.

Requires:
    - ffmpeg
    - yt-dlp
    - datasets[audio]
    - torchaudio
"""
import subprocess
import yt_dlp
import os
import logging
from pathlib import Path

from datasets import load_dataset, Audio


def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir='/tmp/musiccaps',
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    logger = logging.getLogger("music-context")
    command = [
        "yt-dlp",
        "--force-keyframes-at-cuts",
        "--no-warnings",
        "-x",
        "--audio-format",
        "wav",
        "-f",
        "bestaudio",
        "-o",
        str(output_filename),
        "--download-sections",
        f"*{start_time}-{end_time}",
        f"{url_base}{video_identifier}",
    ]

    attempts = 0
    while attempts < num_attempts:
        attempts += 1
        logger.info(
            "Downloading %s (attempt %d/%d)",
            video_identifier,
            attempts,
            num_attempts,
        )
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as err:
            logger.warning("Download failed for %s with exit code %d", video_identifier, err.returncode)
        else:
            if os.path.exists(output_filename):
                return True, "Downloaded"
            logger.warning("Downloader completed but did not create %s", output_filename)

    return False, f"Failed after {num_attempts} attempts"


def main(
    data_dir: str,
    sampling_rate: int = 44100,
    limit: int = None,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
):
    """
    Download the clips within the MusicCaps dataset from YouTube.

    Args:
        data_dir: Directory to save the clips to.
        sampling_rate: Sampling rate of the audio clips.
        limit: Limit the number of examples to download.
        num_proc: Number of processes to use for downloading.
        writer_batch_size: Batch size for writing the dataset. This is per process.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("music-context")
    logger.info("Loading MusicCaps dataset")
    ds = load_dataset('google/MusicCaps', split='train')
    if limit is not None:
        logger.info("Limiting to %d examples", limit)
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            status = False
            status, log = download_clip(
                example['ytid'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )

            logger.info(
                "Finished %s (%s-%s): success=%s, %s",
                example["ytid"],
                example["start_s"],
                example["end_s"],
                status,
                log,
            )

        example['audio'] = outfile_path
        example['download_status'] = status
        return example

    logger.info("Processing %d examples with %d worker process(es)", len(ds), num_proc)
    mapped = ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    )
    logger.info("Casting downloaded audio to %d Hz", sampling_rate)
    return mapped.cast_column('audio', Audio(sampling_rate=sampling_rate))


if __name__ == '__main__':
    ds = main(
        './music_data',
        sampling_rate=44100,
        limit=None,
        num_proc=16,
        writer_batch_size=1000,
    )