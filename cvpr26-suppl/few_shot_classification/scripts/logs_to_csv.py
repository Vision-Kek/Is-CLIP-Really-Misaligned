import os
import re
import csv
import glob

def extract_metrics_simple():
    log_dir = "logs"
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    # Known datasets
    datasets = [
        'stanford_cars', 'oxford_pets', 'oxford_flowers', 'fgvc_aircraft',
        'dtd', 'eurosat', 'food-101', 'caltech101', 'ucf101', 'imagenet', 'sun397'
    ]
    
    # Store results: {method: {dataset: mAP}}
    results = {}
    
    # Regex pattern for Average results
    avg_pattern = re.compile(r'tensor\(\[\s*([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)]\)')
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        name_without_ext = filename[:-4]  # Remove .log
        
        # Find which dataset this file belongs to
        dataset = None
        for ds in datasets:
            if name_without_ext.startswith(ds):
                dataset = ds
                break
        
        if not dataset:
            print(f"Warning: Unknown dataset in {filename}")
            continue
        
        # Extract method info from remaining part of filename
        remaining = name_without_ext[len(dataset)+1:]  # +1 for the underscore
        
        # Determine method name

        method = remaining 
        # Read scores from file
        try:
            with open(log_file, 'r') as f:
                content = f.read()

            match = avg_pattern.search(content)
            if match:
                avg_scores = [float(x) for x in match.groups()] # for each shot one value
                
                if method not in results:
                    results[method] = {}
                
                results[method][dataset] = avg_scores
                print(f"Found: {method} - {dataset}: {{[s:.2f for s in avg_scores]}}")
            else:
                print(f"Warning: No Average found in {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return results

def write_csv_simple(results, output_file="metrics_summary.csv"):
    # Get all methods and datasets
    methods = sorted(results.keys())
    all_datasets = set()
    
    for method_data in results.values():
        all_datasets.update(method_data.keys())
    
    datasets_sorted = sorted(all_datasets)
    # header_row = []
    # for d in datasets_sorted:
    #     header_row.extend([d,2,4,8,16])
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Method'] + datasets_sorted)
        
        # Data rows
        for method in methods:
            row = [method]
            writer.writerow(row)
            for i,shot in enumerate([1,2,4,8,16]):
                row = [shot]
                for dataset in datasets_sorted:
                    shotlist = results[method].get(dataset, '')
                    if len(shotlist) == 0: continue
                    value = shotlist[i]
                    row.append(f"{value:.2f}" if value != '' else '')
                writer.writerow(row)
    
    print(f"CSV written to {output_file}")

if __name__ == "__main__":
    results = extract_metrics_simple()
    if results:
        write_csv_simple(results)
        
        # Print summary
        print("\nExtracted Metrics Summary:")
        for method, datasets in results.items():
            print(f"\n{method}:")
            for dataset, shotwise in datasets.items():
                for score in shotwise:
                    print(f"  {dataset}: {score:.2f}")
    else:
        print("No results found. Make sure log files exist in the logs/ directory.")
