
import os
import json
import glob
from pathlib import Path

# --- Configuration ---
OUTPUT_DIR = r"c:\Users\anand\Downloads\qbt\debug_outputs"
REPORT_FILE = r"c:\Users\anand\Downloads\qbt\model_comparison_report.md"

def load_responses(directory):
    """Load all JSON responses from txt files in directory"""
    responses = {}
    
    files = glob.glob(os.path.join(directory, "*.txt"))
    print(f"Found {len(files)} files in {directory}")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        # Assuming filename format: {qid}_{model}.txt
        # Extract model name (heuristic: remove 2024_Q32_ prefix or similar)
        # Try to find known model names
        model_name = "unknown"
        if "gemini" in filename: model_name = "gemini_2.5_pro"
        elif "claude" in filename: model_name = "claude_sonnet_4.5"
        elif "deepseek" in filename: model_name = "deepseek_r1"
        elif "gpt" in filename: model_name = "gpt_5_1"
        else:
            model_name = filename.replace(".txt", "")
            
        print(f"Loading {filename} as {model_name}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Robust JSON extraction
            # 1. Try to find the start after "FULL JSON RESPONSE:"
            marker = "FULL JSON RESPONSE:"
            start_search = content.find(marker)
            if start_search != -1:
                start_idx = content.find('{', start_search)
            else:
                start_idx = content.find('{')
                
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_candidate = content[start_idx : end_idx + 1]
                
                # It's possible there are multiple JSONs or extra text. 
                # Let's try to parse. If it fails, it might be because of multiple objects.
                # In that case, we might need a more sophisticated parser, but let's try strict first.
                try:
                    data = json.loads(json_candidate)
                    responses[model_name] = data
                    print(f"Successfully loaded JSON for {model_name}")
                except json.JSONDecodeError:
                    # Retry: The 'end_idx' might be too far if there are multiple JSONs.
                    # Let's try to find the matching closing brace for the first opening brace.
                    # Simple counter approach
                    try:
                        stack = 0
                        actual_end = -1
                        for i, char in enumerate(content[start_idx:], start=start_idx):
                            if char == '{':
                                stack += 1
                            elif char == '}':
                                stack -= 1
                                if stack == 0:
                                    actual_end = i
                                    break
                        
                        if actual_end != -1:
                            json_candidate = content[start_idx : actual_end + 1]
                            data = json.loads(json_candidate)
                            responses[model_name] = data
                            print(f"Successfully loaded JSON for {model_name} (using stack parser)")
                        else:
                            print(f"[!] Could not find matching closing brace for {filename}")
                    except Exception as e:
                        print(f"[!] JSON Parsing failed for {filename}: {e}")

            else:
                print(f"[!] No JSON structure found in {filename}")

        except Exception as e:
            print(f"[!] Error reading {filename}: {e}")
            
    return responses

def extract_field_paths(data, prefix=""):
    """Recursively extract all field paths from nested JSON"""
    paths = set()
    
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, (dict, list)):
                # Still add the container path itself if it's a list (to show full list comparison)
                # But mostly recurse
                paths.update(extract_field_paths(value, new_prefix))
                if isinstance(value, list) and not any(isinstance(x, (dict,list)) for x in value):
                     # Leaf list (list of strings/ints) - verify
                     paths.add(new_prefix)
            else:
                paths.add(new_prefix)
                
    elif isinstance(data, list):
        # For lists of objects, we want to recurse
        # For lists of primitives, we treat the list as the unit
        if data and isinstance(data[0], (dict, list)):
             for i, item in enumerate(data):
                new_prefix = f"{prefix}[{i}]"
                paths.update(extract_field_paths(item, new_prefix))
        else:
             paths.add(prefix)
             
    return paths

def get_value_at_path(data, path):
    """Get value from nested dict using dot notation"""
    try:
        keys = path.replace("[", ".").replace("]", "").split(".")
        curr = data
        for k in keys:
            if isinstance(curr, list):
                if k.isdigit() and int(k) < len(curr):
                    curr = curr[int(k)]
                else:
                    return None
            elif isinstance(curr, dict):
                if k in curr:
                    curr = curr[k]
                else:
                    return None
            else:
                return None
        return curr
    except:
        return None

def generate_markdown_report(responses, output_base_dir):
    """Generate side-by-side comparison report split by tier"""
    
    # Get all unique paths from all models
    all_paths = set()
    for data in responses.values():
        all_paths.update(extract_field_paths(data))
    
    sorted_paths = sorted(list(all_paths))
    models = sorted(responses.keys())
    
    # Group by Tiers for readability
    tiers = {
        "Meta": [],
        "Tier 0 (Classification)": [],
        "Tier 1 (Core Research)": [],
        "Tier 2 (Student Learning)": [],
        "Tier 3 (Enhanced Learning)": [],
        "Tier 4 (Metadata)": []
    }
    
    for path in sorted_paths:
        if path.startswith("tier_0"): tiers["Tier 0 (Classification)"].append(path)
        elif path.startswith("tier_1"): tiers["Tier 1 (Core Research)"].append(path)
        elif path.startswith("tier_2"): tiers["Tier 2 (Student Learning)"].append(path)
        elif path.startswith("tier_3"): tiers["Tier 3 (Enhanced Learning)"].append(path)
        elif path.startswith("tier_4"): tiers["Tier 4 (Metadata)"].append(path)
        else: tiers["Meta"].append(path)

    generated_files = []

    # Write separate files for each tier
    for tier_name, paths in tiers.items():
        if not paths: continue
        
        # Create safe filename
        safe_name = tier_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"model_comparison_{safe_name}.md"
        output_path = os.path.join(output_base_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Model Response Comparison - {tier_name}\n\n")
            
            for path in paths:
                f.write(f"### `{path}`\n\n")
                
                # Check value length to decide formatting
                values = {m: get_value_at_path(responses[m], path) for m in models}
                
                is_long_text = any(isinstance(v, str) and len(v) > 100 for v in values.values() if v)
                is_complex = any(isinstance(v, (list, dict)) for v in values.values() if v)
                
                if is_long_text or is_complex:
                    # Vertical stacked format for long text
                    for model in models:
                        val = values[model]
                        f.write(f"**{model}:**\n")
                        if isinstance(val, (dict, list)):
                                f.write(f"```json\n{json.dumps(val, indent=2)}\n```\n\n")
                        else:
                                f.write(f"> {val}\n\n")
                    f.write("---\n\n")
                else:
                    # Table format for short values
                    header = "| Model | Value |\n|---|---|\n"
                    f.write(header)
                    for model in models:
                            val = values[model]
                            # Escape pipes for markdown table
                            val_str = str(val).replace("|", "\\|").replace("\n", " ")
                            f.write(f"| **{model}** | {val_str} |\n")
                    f.write("\n")
        
        print(f"Generated: {output_path}")
        generated_files.append(output_path)

    return generated_files

# --- Main ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        print(f"Directory not found: {OUTPUT_DIR}")
    else:
        responses = load_responses(OUTPUT_DIR)
        if responses:
            generate_markdown_report(responses, OUTPUT_DIR)
        else:
            print("No valid responses found.")
