#!/bin/bash

# Define the tasks array
tasks=("task_1" "task_2" "task_3" "task_4" "task_5a" "task_5b") # "task_1"
MODEL="gemini"    # Replace with your desired model
SCRIPT="eval_spa.py"
RETRY_INTERVAL=5  # Seconds to wait before retrying

# Define the visual methods array
visual_methods=("gripper_one" "gripper_all" "contact_one" "contact_all" "distance")

# Define the number of task variants for each task
declare -A task_variant_counts
task_variant_counts["task_1"]=2
task_variant_counts["task_2"]=2
task_variant_counts["task_3"]=2
task_variant_counts["task_4"]=2
task_variant_counts["task_5a"]=3
task_variant_counts["task_5b"]=3

# Map task names to numbers
declare -A task_numbers
task_numbers["task_1"]=1
task_numbers["task_2"]=2
task_numbers["task_3"]=3
task_numbers["task_4"]=4
task_numbers["task_5a"]=5
task_numbers["task_5b"]=6

# Define the data directory where the files and folders are located
data_dir="../ManiSkill2/data/mani_skill2_ycb"  # Replace with the actual path

# Initialize previous task number, wrapping around
prev_task_num=6  # Since we start with task_1, previous task is task_5b which is number 6

# Loop over each TASK
for TASK in "${tasks[@]}"
do
    echo "Processing TASK: $TASK"
    task_num=${task_numbers[$TASK]}
    echo "Current task number: $task_num"
    echo "Previous task number: $prev_task_num"

    # Before running new TASK, modify folder and file names

    # (1) Rename info_pick_v0.json to info_pick_v0_x.json where x is the previous task number
    # Wrap around at the first task
    original_file="${data_dir}/info_pick_v0.json"
    new_file="${data_dir}/info_pick_v0_${prev_task_num}.json"
    # if [ -f "$original_file" ]; then
    #     mv "$original_file" "$new_file"
    #     echo "Renamed '$original_file' to '$new_file'"
    # else
    #     echo "File '$original_file' does not exist. Skipping renaming."
    # fi

    # (2) Rename info_pick_v0_y.json to info_pick_v0.json where y is the current task number
    source_file="${data_dir}/info_pick_v0_${task_num}.json"
    dest_file="${data_dir}/info_pick_v0.json"

    if [ -f "$source_file" ]; then
        rm "$dest_file"
        cp "$source_file" "$dest_file"
        echo "Copied '$source_file' to '$dest_file'"
    else
        cp "$source_file" "$dest_file"
        echo "Copied '$source_file' to '$dest_file'"
    fi

    # # (3) Rename models to models_x where x is the previous task number
    # original_folder="${data_dir}/models"
    # new_folder="${data_dir}/models_${prev_task_num}"
    # if [ -d "$original_folder" ]; then
    #     mv "$original_folder" "$new_folder"
    #     echo "Renamed '$original_folder' to '$new_folder'"
    # else
    #     echo "Folder '$original_folder' does not exist. Skipping renaming."
    # fi

    # # (4) Rename models_y to models where y is the current task number
    # source_folder="${data_dir}/models_${task_num}"
    # dest_folder="${data_dir}/models"
    # if [ -d "$source_folder" ]; then
    #     mv "$source_folder" "$dest_folder"
    #     echo "Renamed '$source_folder' to '$dest_folder'"
    # else
    #     echo "Folder '$source_folder' does not exist. Skipping renaming."
    # fi

    # Get the number of task variants for the current TASK
    NUM_TASK_VARIANTS=${task_variant_counts[$TASK]}
    
    for (( idx=0; idx<$NUM_TASK_VARIANTS; idx++ ))
    do
        for visual_method in "${visual_methods[@]}"
        do
            while true; do
                echo "Starting the script: python $SCRIPT --task $TASK --task_idx $idx --visual_method $visual_method --model $MODEL"
                
                # Run the Python script and capture its exit status
                python $SCRIPT --task $TASK --task_idx $idx --visual_method $visual_method --model $MODEL
                EXIT_CODE=$?

                if [ $EXIT_CODE -eq 0 ]; then
                    echo "Script completed successfully for task_idx $idx and visual_method $visual_method."
                    break
                else
                    echo "Script encountered an error (exit code $EXIT_CODE). Retrying in $RETRY_INTERVAL seconds..."
                    sleep $RETRY_INTERVAL
                fi
            done
        done
    done

    # Update prev_task_num to current task number for next iteration
    prev_task_num=$task_num
done
