import tarfile
from pathlib import Path

archives = [
    Path('D:/CEREBRO_DATA/OASIS2/archives/OAS2_RAW_PART1.tar.gz'),
    Path('D:/CEREBRO_DATA/OASIS2/archives/OAS2_RAW_PART2.tar.gz')
]

for archive_path in archives:
    print(f"\\n--- Inspecting {archive_path.name} ---")
    if not archive_path.exists():
        print(f"File not found: {archive_path}")
        continue
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        count = 0
        mri_count = 0
        dirs = set()
        extensions = set()
        
        # We don't want to iterate the whole 10GB file if it takes too long, but let's look at the first 1000 items
        for member in tar:
            count += 1
            if count <= 50:
                print(member.name)
            
            # Record directories to understand structure
            parts = member.name.split('/')
            if len(parts) > 1:
                dirs.add(parts[0])
                
            if member.isfile():
                ext = Path(member.name).suffixes
                ext_str = "".join(ext)
                if ext_str:
                    extensions.add(ext_str)
                if '.nii' in member.name:
                    mri_count += 1
                    
            if count >= 2000:
                print(f"\\n... Stopping early after 2000 files ...")
                break
                
        print(f"Total files inspected: {count}")
        print(f"MRI files found in sample: {mri_count}")
        print(f"Root directories found: {dirs}")
        print(f"Extensions found: {extensions}")
