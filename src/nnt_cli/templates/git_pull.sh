#!/bin/bash
#For updating code from github

REPO_PATH= #[You local project path]
#Example: REPO_PATH="$HOME/workspace/NN_project"

cd $REPO_PATH

Branch=Test

git checkout $Branch

if [ -n "$(git status --porcelain)" ]; then
  echo "Uncommitted changes detected."
  git status --short
  
  read -p "Push local change? (y: push, n: discard, i: ignore, s: stash): " user_choice

  if [ "$user_choice" = "y" ]; then
    echo "Saving local changes..."

    ./git_push.sh
    
  elif [ "$user_choice" = "n" ]; then
    echo "Discard local changes and force pull of remote code..."
    git reset --hard
    git clean -fd
  elif [ "$user_choice" = "i" ]; then
    echo "Ignore local changes and continue."
  elif [ "$user_choice" = "s" ]; then
    git stash
    echo "Stash local changes and continue."
  else
    echo "Invalid input, please rerun the script and select a valid option (y/n/i)"
    exit 1
  fi
else
  echo "There are no uncommitted changes locally"
fi

    git fetch origin
    git reset --hard origin/$Branch

echo "Pull completed and the current branch has been synchronized to the latest remote code!"