import hashlib
import datetime
import os
import fitz  # PyMuPDF

repo_url = f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'unknown/repo')}"
run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
run_number = os.environ.get("GITHUB_RUN_NUMBER", "unknown")
workflow_name = os.environ.get("GITHUB_WORKFLOW", "unknown")
commit_sha = os.environ.get("GITHUB_SHA", "unknown")

source_file = "course/intro/python_exercises.py"
test_intro_output = "intro_tests_output.txt"
test_all_output = "all_tests_output.txt"
linter_output = "flake8_output.txt"
doit_output = "doit_output.txt"


def compute_checksum(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def read_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read()
    return f"File not found: {filepath}\n"


checksum = compute_checksum(source_file)
date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
intro_tests = read_file(test_intro_output)
all_tests = read_file(test_all_output)
linter = read_file(linter_output)
doit = read_file(doit_output)

report = f"""
CI Report

Repository URL: {repo_url}
Date and Time: {date_time}

Workflow: {workflow_name}
Run Number: {run_number}
Run ID: {run_id}
Commit SHA: {commit_sha}


Checksum of {source_file}:
{checksum}

--- Intro Topic Test Results ---
{intro_tests}

--- All Topics Test Results ---
{all_tests}

--- Flake8 Linter Output ---
{linter}

---Doit pipeline Outout
{doit}
"""


# Create multi-page PDF
doc = fitz.open()
lines = report.split('\n')
lines_per_page = 40  # Adjust as needed

for i in range(0, len(lines), lines_per_page):
    page = doc.new_page()
    chunk = '\n'.join(lines[i:i + lines_per_page])
    page.insert_text((72, 72), chunk, fontsize=10)

doc.save("ci_report.pdf")
doc.close()
