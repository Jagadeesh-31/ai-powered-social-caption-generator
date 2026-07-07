"""
Deployment helper script. Stages changes, commits, pushes to GitHub,
and triggers the Render deploy hook in one command.

Usage:
    python deploy.py "your commit message"
"""

import sys
import subprocess
import requests

DEPLOY_HOOK_URL = "https://api.render.com/deploy/srv-d969r9u7r5hc73821pig?key=0LUPzjzEdvI"

def run_cmd(command):
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error executing: {command}")
        print(result.stderr)
        return False
    print(result.stdout.strip())
    return True

def main():
    commit_msg = "Auto commit & deploy"
    if len(sys.argv) > 1:
        commit_msg = sys.argv[1]

    print("Starting deployment process...")

    print("\n1. Staging changes...")
    if not run_cmd("git add ."):
        sys.exit(1)

    print("\n2. Committing changes...")
    run_cmd(f'git commit -m "{commit_msg}"')

    print("\n3. Pushing to GitHub...")
    if not run_cmd("git push origin main"):
        sys.exit(1)

    print("\n4. Triggering Render deployment...")
    response = requests.post(DEPLOY_HOOK_URL)
    if response.status_code in [200, 201, 202]:
        print("\nSuccess! Render deployment triggered successfully.")
        print(f"Render Response Status: {response.status_code}")
    else:
        print(f"\nFailed to trigger Render. Status: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
