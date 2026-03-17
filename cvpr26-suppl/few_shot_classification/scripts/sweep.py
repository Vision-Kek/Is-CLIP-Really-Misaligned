import subprocess
import os

def run_retrieval_experiments():
    # Define the base command components
    base_cmd = "python  main_few_shots.py"
    # more config defs here
    
    # Define the sweeps
    datasets = [
        'food-101', 'stanford_cars', 'oxford_pets', 'oxford_flowers', 'fgvc_aircraft','dtd','eurosat', 'food-101', 'caltech101', 'ucf101'
    ]
    
    models = [
        #'RN50'
        #, 'ViT-B/16', 'ViT-B/32', 'ViT-L/14',
        #'siglip-ViT-B-16', 'siglip2-ViT-B-16',
        # 'dinov2_vitb14',
        'dinov3_vitl16']
    
    # Additional parameter options
    methods = ['gda','prototype']  # False = omit, True = include
    use_text_flags = [False]
    use_project_flags = [False]
    
    # Create logs directory if it doesn't exist
    os.makedirs('../logs', exist_ok=True)
    
    # Iterate through all combinations
    for dataset in datasets:
        for model in models:
            for method in methods:
                for use_text in use_text_flags:
                    for use_project in use_project_flags:
                        # Build the command components
                        dataset_arg = f"--dataset {dataset}"
                        backbone_arg = f"--backbone {model}"
                        method_arg = f"--classifier {method}"
                        use_text_arg = f"--use_text"
                        use_project_arg = f"--project"

                        # Create query_exp_name and log filename
                        model_suffix = model.lower().replace('/', '').replace('-', '').replace('/', '')
                        log_suffix = f"{dataset}_{model_suffix}_{method}{'_use_text' if use_text else ''}{'_project' if use_project else ''}.log"

                        # Build the full command
                        cmd_parts = [
                            base_cmd,
                            dataset_arg,
                            backbone_arg,
                            method_arg
                        ]
                        if use_text: cmd_parts += [use_text_arg]
                        if use_project: cmd_parts += [use_project_arg]

                        # Remove empty strings and join
                        full_cmd = " ".join([part for part in cmd_parts if part.strip()])

                        # Add tee command for logging
                        full_cmd_with_log = f"{full_cmd} | tee logs/{log_suffix}"

                        print(f"Executing: {full_cmd_with_log}")

                        try:
                            # Execute the command
                            result = subprocess.run(full_cmd_with_log, shell=True, check=True)
                            print(f"Completed: {log_suffix}")
                        except subprocess.CalledProcessError as e:
                            print(f"Error executing command: {e}")
                            print(f"Failed command: {full_cmd_with_log}")
                        except Exception as e:
                            print(f"Unexpected error: {e}")

                        print("-" * 80)

if __name__ == "__main__":
    run_retrieval_experiments()
