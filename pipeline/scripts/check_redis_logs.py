
import os

log_files = [
    r"c:\Users\anand\Downloads\qbt\pipeline\logs\pipeline_20251217.log",
    r"c:\Users\anand\Downloads\qbt\pipeline\logs\pipeline_20251216.log"
]

print("Scanning logs for Redis issues...")

for log_file in log_files:
    if os.path.exists(log_file):
        print(f"\n--- Checking {os.path.basename(log_file)} ---")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if "redis" in line.lower() or "connection" in line.lower():
                        # Filter out healthy connection logs unless they are followed by failure
                        if "error" in line.lower() or "failed" in line.lower() or "refused" in line.lower() or "exception" in line.lower():
                            print(line.strip())
                        elif "connecting" in line.lower():
                            # Show connection attempts to see if they hang or succeed
                            print(line.strip())
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
