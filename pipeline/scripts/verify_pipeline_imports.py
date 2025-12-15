
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"Project Root: {PROJECT_ROOT}")

modules_to_check = [
    "pipeline.scripts.init_00_initialization",
    "pipeline.scripts.init_01_question_loader",
    "pipeline.scripts.init_02_question_classifier",
    "pipeline.scripts.init_03_cache_manager",
    "pipeline.scripts.init_04_retrieval_dense",
    "pipeline.scripts.init_05_retrieval_sparse",
    "pipeline.scripts.init_06_retrieval_merger",
    "pipeline.scripts.init_07_image_consensus",
    "pipeline.scripts.init_08_model_orchestrator",
    "pipeline.scripts.init_09_voting_engine",
    "pipeline.scripts.init_10_debate_orchestrator",
    "pipeline.scripts.init_11_synthesis_engine",
    "pipeline.scripts.init_12_output_manager",
    "pipeline.scripts.init_13_checkpoint_manager",
    "pipeline.scripts.init_14_health_monitor",
    "pipeline.scripts.init_99_pipeline_runner",
]

failed = []
params_fixes = []

for module in modules_to_check:
    print(f"Checking {module}...")
    try:
        __import__(module)
        print(f"  OK")
    except ImportError as e:
        print(f"  FAIL: {e}")
        failed.append((module, str(e)))
    except Exception as e:
         print(f"  FAIL (Runtime): {e}")
         failed.append((module, str(e)))

print("\nSummary:")
if failed:
    print(f"Found {len(failed)} failures:")
    for m, e in failed:
        print(f"  {m}: {e}")
    sys.exit(1)
else:
    print("All modules imported successfully.")
    sys.exit(0)
