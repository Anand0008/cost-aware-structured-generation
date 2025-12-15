
import sys
import os
from pathlib import Path
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "pipeline"))

# Load config
config_path = PROJECT_ROOT / "config" / "prompts_config.yaml"
if not config_path.exists():
    print(f"Config not found: {config_path}")
    sys.exit(1)

with open(config_path, 'r') as f:
    prompts_config = yaml.safe_load(f)

# Initialize builder
from pipeline.utils.prompt_builder import PromptBuilder
builder = PromptBuilder(prompts_config)

print("Validating templates...")
failed = False
for template in builder.list_templates():
    validation = builder.validate_template(template)
    if not validation['valid']:
        print(f"❌ {template}: INVALID")
        for issue in validation['issues']:
            print(f"    - {issue}")
        failed = True
    else:
        print(f"✓ {template}: Valid")
        # Try a dummy build to check for format string errors
        try:
            vars = {v: "TEST" for v in validation['variables_declared']}
            builder.build_prompt(template, vars)
            print(f"    - Build check: OK")
        except Exception as e:
            print(f"    - Build check: FAIL ({e})")
            failed = True

if failed:
    sys.exit(1)
else:
    sys.exit(0)
