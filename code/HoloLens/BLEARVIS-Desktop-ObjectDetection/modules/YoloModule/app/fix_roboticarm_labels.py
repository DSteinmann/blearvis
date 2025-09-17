#!/usr/bin/env python3
"""
Script to check and fix roboticarm annotation class labels
"""

import os
import glob

def check_and_fix_roboticarm_labels():
    """Check all roboticarm annotation files and change class 0 to class 1"""
    dataset_dir = "detector/data/dataset"

    # Find all roboticarm annotation files
    patterns = [
        os.path.join(dataset_dir, "**", "P10*.txt")
    ]

    fixed_count = 0

    for pattern in patterns:
        for txt_file in glob.glob(pattern, recursive=True):
            print(f"Checking: {txt_file}")

            # Read the file
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            modified = False
            new_lines = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = parts[0]
                    if class_id == '0':
                        # Change class 0 to class 1
                        parts[0] = '1'
                        modified = True
                        print(f"  Fixed class 0 -> 1 in {txt_file}")
                        fixed_count += 1

                    new_lines.append(' '.join(parts))
                else:
                    new_lines.append(line.strip())

            # Write back if modified
            if modified:
                with open(txt_file, 'w') as f:
                    f.write('\n'.join(new_lines) + '\n')

    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == "__main__":
    check_and_fix_roboticarm_labels()