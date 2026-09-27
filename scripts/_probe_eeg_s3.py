"""Probe OpenNeuro ds004504 S3 structure and download participants.tsv."""
import urllib.request
import os

os.makedirs('data/raw/eeg', exist_ok=True)

base = 'https://s3.amazonaws.com/openneuro.org/ds004504'

# Download participants.tsv
urllib.request.urlretrieve(f'{base}/participants.tsv', 'data/raw/eeg/participants.tsv')
print('participants.tsv downloaded')

# Try dataset_description
try:
    urllib.request.urlretrieve(f'{base}/dataset_description.json', 'data/raw/eeg/dataset_description.json')
    print('dataset_description.json downloaded')
except Exception as e:
    print(f'dataset_description: {e}')

# Check EEG files for first subject
paths_to_check = [
    'sub-001/eeg/sub-001_task-eyesclosed_eeg.set',
    'sub-001/eeg/sub-001_task-eyesclosed_eeg.vhdr',
    'sub-001/eeg/sub-001_task-eyesclosed_eeg.edf',
    'sub-001/eeg/sub-001_task-eyesclosed_eeg.fif',
    'sub-001/eeg/sub-001_task-eyesclosed_channels.tsv',
]

for rel_path in paths_to_check:
    url = f'{base}/{rel_path}'
    try:
        req = urllib.request.Request(url, method='HEAD')
        r = urllib.request.urlopen(req, timeout=10)
        size_bytes = r.headers.get('Content-Length', 'unknown')
        print(f'FOUND: {rel_path}  size={size_bytes}')
    except Exception as e:
        print(f'MISSING: {rel_path}  ({e})')
