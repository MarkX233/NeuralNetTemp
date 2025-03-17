#!/bin/bash
# Script for updating results in HPC and pushing to a specific branch
# Handles cases where the remote branch is ahead of the local branch

REPO_PATH=  # Path to the repository
BRANCH="Test"  # Target branch for pushing changes
COMMIT_MESSAGE="Auto commit $(date)"  # Auto-generated commit message

# Change to the repository directory
cd $REPO_PATH || { echo "Repository path not found: $REPO_PATH"; exit 1; }

# Ensure the SSH key is added (uncomment if necessary)
# ssh-add ~/.ssh/id_ed25519

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
  echo "Uncommitted changes detected. Preparing to commit..."
  git add .
  git commit -m "$COMMIT_MESSAGE"
else
  echo "No changes to commit."
fi

# Pull changes from the remote branch to avoid conflicts
echo "Pulling changes from the remote branch: $BRANCH..."
git fetch origin
if ! git merge origin/$BRANCH; then
  echo "Merge conflicts detected. Resolve conflicts and retry."
  exit 1
fi

# Push changes to the remote branch
echo "Pushing changes to the remote branch: $BRANCH..."
if git push origin $BRANCH; then
  echo "Changes have been pushed successfully!"
else
  echo "Failed to push changes. You might need to resolve issues manually."
  exit 1
fi
