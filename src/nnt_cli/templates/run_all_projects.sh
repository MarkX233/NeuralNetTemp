#!/bin/bash

CLEAN_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean|-c)
            CLEAN_MODE=true
            shift
            ;;
        *)
            echo "Unknown arg!: $1"
            exit 1
            ;;
    esac
done

LOG_FILE="task_log_$(date +'%Y%m%d_%H%M%S').log"

# Define a list of Python scripts to be executed (executed in order)
# Must consist the name of your project folder
# In that folder must have a set task_script
TASKS=(
    "MLP_FashionMINIST"
    "MLP_MINIST"

)

    if [ "$CLEAN_MODE" = true ]; then
        echo -n "Warning! You are using cleaning mode, results will be deleted, continue?(y/n) " | tee -a "../$LOG_FILE"
        read -r answer

    fi

for task in "${TASKS[@]}"; do
    start_time=$(date +'%Y-%m-%d %H:%M:%S')
    echo ">>> Start executing the task: $task | Time: $start_time" | tee -a "$LOG_FILE"
    
    cd "$task" || { echo "Failed to enter directory: $task" | tee -a "../$LOG_FILE"; exit 1; }

    case $answer in
        [Yy]*)
            echo "Cleaning: Delete results/ and *.txt files..."  | tee -a "../$LOG_FILE"
            rm -rf results/ *.txt 2>&1 | tee -a "../$LOG_FILE"
            ;;
        *)
            echo "Delete skip." | tee -a "../$LOG_FILE"
            ;;
    esac

    python -u task_script.py 2>&1 | tee -a "../$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        status="Success!"
    else
        status="Failure"
    fi
    
    end_time=$(date +'%Y-%m-%d %H:%M:%S')
    echo "<<< Task completed: $task | State: $status | Time: $end_time" | tee -a "../$LOG_FILE"
    echo "----------------------------------------" | tee -a "../$LOG_FILE"

    cd ..
done

echo "All tasks completed! Logs saved to: $LOG_FILE" | tee -a "$LOG_FILE"
