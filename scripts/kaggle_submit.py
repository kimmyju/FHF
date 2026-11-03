"""Kaggle Code Competition entrypoint: offline test CSV -> submission.parquet."""
import os
from pathlib import Path
from predict import submit


def main():
    checkpoint = os.environ.get('FHF_CHECKPOINT', '/kaggle/input/fhf-checkpoint')
    test = '/kaggle/input/competitions/name-that-llm/test.csv'
    output = '/kaggle/working/submission.parquet'
    for path in (checkpoint, test):
        if not Path(path).exists():
            raise FileNotFoundError(f'Missing Kaggle input: {path}')
    submit(checkpoint, test, output,
           batch_size=int(os.environ.get('FHF_BATCH_SIZE', '32')),
           max_length=int(os.environ.get('FHF_MAX_LENGTH', '384')))


if __name__ == '__main__':
    main()
