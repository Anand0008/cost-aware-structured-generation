
import os
import re

log_file = r"c:\Users\anand\Downloads\qbt\pipeline\logs\pipeline_20251217.log"

if not os.path.exists(log_file):
    print(f"Log file not found: {log_file}")
    # Try the 16th if 17th not found (though ls said it exists)
    log_file = r"c:\Users\anand\Downloads\qbt\pipeline\logs\pipeline_20251216.log"

print(f"Analyzing {log_file}...")

validation_failures = []
voting_disputes = []

try:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        # validation errors
        if "Schema validation found" in line or "Validation failed" in line:
            validation_failures.append(line.strip())
            # Look ahead for specific errors usually printed after
            for j in range(1, 20):
                if i+j < len(lines):
                    next_line = lines[i+j].strip()
                    if "Missing" in next_line or "Invalid" in next_line or "Expected" in next_line:
                        validation_failures.append(next_line)
                    elif "INFO" in next_line or "WARNING" in next_line or "ERROR" in next_line:
                        # Stop if we hit next log entry
                        if "Missing" not in next_line and "Invalid" not in next_line: 
                             break
        
        # specific validator lines from json_validator.py logging
        if "validation errors:" in line:
             validation_failures.append(line.strip())
        
        # Voting disputes
        if "Top disputed fields:" in line:
            voting_disputes.append(line.strip())
        if "Disputed:" in line and "fields" in line:
             voting_disputes.append(line.strip())

    with open(r"c:\Users\anand\Downloads\qbt\pipeline\scripts\analysis_output.txt", "w", encoding='utf-8') as out:
        out.write("\n=== JSON VALIDATION FAILURES ===\n")
        for item in validation_failures:
            out.write(item + "\n")
            
        out.write("\n=== VOTING ENGINE DISPUTES ===\n")
        for item in voting_disputes:
            out.write(item + "\n")
            
    print("Analysis complete. Written to analysis_output.txt")

except Exception as e:
    print(f"Error reading file: {e}")
