"""
Quick script to check latest versions of our dependencies from PyPI.
"""

import subprocess

packages = [
    'tensorflow',
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'wordcloud',
    'nltk',
    'scikit-learn',
    'tqdm',
    'jupyter',
    'ipykernel',
    'streamlit'
]

print("=" * 70)
print("CHECKING LATEST VERSIONS FROM PyPI")
print("=" * 70 + "\n")

for package in packages:
    try:
        result = subprocess.run(
            ['pip', 'index', 'versions', package],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Parse output to get latest version
        output = result.stdout
        if "Available versions:" in output:
            versions_line = output.split("Available versions:")[1].split("\n")[0]
            latest = versions_line.strip().split(",")[0].strip()
            print(f"{package:20s} -> {latest}")
        else:
            print(f"{package:20s} -> Could not determine")

    except Exception as e:
        print(f"{package:20s} -> Error: {e}")

print("\n" + "=" * 70)

