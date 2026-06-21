#!/bin/bash

echo "Starting prompt generation scripts..."

# Loop through numbers 1 to 5
for i in {1..5}; do
    # Loop through letters a and b
    for x in a b; do
        script_name="prompt_utils/prompt_generation_${i}${x}.py"
        
        # Check if the file actually exists before trying to run it
        if [[ -f "$script_name" ]]; then
            echo "----------------------------------------"
            echo "Running $script_name..."
            python "$script_name"
            
            # Check the exit status of the python script
            if [[ $? -ne 0 ]]; then
                echo "Error: $script_name failed to execute properly. Stopping execution."
                exit 1
            fi
        else
            echo "Warning: $script_name not found. Skipping."
        fi
    done
done

echo "----------------------------------------"
echo "All scripts executed successfully!"