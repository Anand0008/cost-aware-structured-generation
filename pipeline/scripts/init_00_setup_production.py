"""
============================================================================
PRODUCTION SETUP & VALIDATION
============================================================================
Purpose: Validate production environment before running pipeline
Features:
    - Validate all environment variables
    - Test all API connections
    - Verify database access
    - Check file system permissions
    - Download required models
    - Generate health report

Usage:
    python scripts/00_setup_production.py
    
    # Or with Docker:
    docker run gate-ae-pipeline python scripts/00_setup_production.py

Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import from package
try:
    from pipeline.utils.logging_utils import setup_logger
    from pipeline.scripts.init_00_initialization import PipelineInitializer
except ImportError:
    # Fallback for direct script execution without package install
    sys.path.append(str(PROJECT_ROOT / "pipeline"))
    from utils.logging_utils import setup_logger
    from scripts.init_00_initialization import PipelineInitializer

logger = setup_logger("setup_production")


class ProductionValidator:
    """
    Validate production environment and readiness
    """
    
    def __init__(self):
        self.validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
            "overall_status": "unknown"
        }
        
        self.critical_dirs = [
            "data/questions",
            "data/images",
            "logs",
            "cache",
            "checkpoints",
            "outputs"
        ]
    
    def run_all_validations(self) -> dict:
        """
        Run all validation checks
        
        Returns:
            dict: Validation results
        """
        logger.info("=" * 80)
        logger.info("PRODUCTION ENVIRONMENT VALIDATION")
        logger.info("=" * 80)
        
        # Step 1: Environment variables
        self._validate_environment_variables()
        
        # Step 2: File system
        self._validate_filesystem()
        
        # Step 3: Initialize components
        self._validate_initialization()
        
        # Step 4: API connectivity
        self._validate_api_connections()
        
        # Step 5: Database connectivity
        self._validate_databases()
        
        # Step 6: Model availability
        self._validate_models()
        
        # Step 7: Permissions
        self._validate_permissions()
        
        # Determine overall status
        self._determine_overall_status()
        
        # Print report
        self._print_report()
        
        return self.validation_results
    
    def _validate_environment_variables(self):
        """Validate all required environment variables"""
        logger.info("\n[1/7] Validating Environment Variables...")
        
        # Check if using Bedrock or direct Anthropic
        use_bedrock = os.getenv('USE_BEDROCK_FOR_CLAUDE', 'false').lower() == 'true'
        
        # Required variables
        required = {
            "GOOGLE_API_KEY": "Google Gemini API",
            "OPENAI_API_KEY": "OpenAI GPT-5.1 API",
            "DEEPSEEK_API_KEY": "DeepSeek R1 API",
            "QDRANT_URL": "Qdrant Vector Database",
            "QDRANT_API_KEY": "Qdrant Authentication",
            "REDIS_URL": "Redis Cache"
        }
        
        # Add Claude authentication
        if use_bedrock:
            required.update({
                "AWS_ACCESS_KEY_ID": "AWS Access Key",
                "AWS_SECRET_ACCESS_KEY": "AWS Secret Key",
                "AWS_REGION_NAME": "AWS Region"
            })
        else:
            required["ANTHROPIC_API_KEY"] = "Anthropic Claude API"
        
        # Optional variables
        optional = {
            "S3_BUCKET_NAME": "AWS S3 Storage",
            "DYNAMODB_TABLE_NAME": "AWS DynamoDB Metadata",
            "BUDGET_LIMIT": "Pipeline Budget Limit",
            "LOG_LEVEL": "Logging Level"
        }
        
        missing_required = []
        present_required = []
        present_optional = []
        missing_optional = []
        
        # Check required
        for var, description in required.items():
            value = os.getenv(var)
            if value:
                present_required.append(f"✓ {var}: {description}")
                # Mask sensitive values
                if "KEY" in var or "SECRET" in var:
                    masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                    logger.info(f"  ✓ {var}: {masked}")
                else:
                    logger.info(f"  ✓ {var}: {value}")
            else:
                missing_required.append(f"✗ {var}: {description}")
                logger.error(f"  ✗ {var}: NOT SET")
        
        # Check optional
        for var, description in optional.items():
            value = os.getenv(var)
            if value:
                present_optional.append(f"✓ {var}: {description}")
                logger.info(f"  ✓ {var}: {value}")
            else:
                missing_optional.append(f"○ {var}: {description}")
                logger.warning(f"  ○ {var}: Not set (optional)")
        
        # Store results
        self.validation_results["checks"]["environment_variables"] = {
            "status": "pass" if not missing_required else "fail",
            "required_present": len(present_required),
            "required_missing": len(missing_required),
            "optional_present": len(present_optional),
            "missing_vars": missing_required
        }
        
        if missing_required:
            self.validation_results["errors"].append(
                f"Missing {len(missing_required)} required environment variables"
            )
        
        if missing_optional:
            self.validation_results["warnings"].append(
                f"Missing {len(missing_optional)} optional environment variables"
            )
    
    def _validate_filesystem(self):
        """Validate file system structure and permissions"""
        logger.info("\n[2/7] Validating File System...")
        
        missing_dirs = []
        created_dirs = []
        
        for dir_path in self.critical_dirs:
            full_path = PROJECT_ROOT / dir_path
            
            if full_path.exists():
                logger.info(f"  ✓ {dir_path}: exists")
            else:
                logger.warning(f"  ○ {dir_path}: missing, creating...")
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_path)
                    logger.info(f"    → Created {dir_path}")
                except Exception as e:
                    missing_dirs.append(dir_path)
                    logger.error(f"    ✗ Failed to create {dir_path}: {e}")
        
        self.validation_results["checks"]["filesystem"] = {
            "status": "pass" if not missing_dirs else "fail",
            "created_directories": created_dirs,
            "failed_directories": missing_dirs
        }
        
        if missing_dirs:
            self.validation_results["errors"].append(
                f"Failed to create directories: {missing_dirs}"
            )
    
    def _validate_initialization(self):
        """Test pipeline initialization"""
        logger.info("\n[3/7] Testing Pipeline Initialization...")
        
        try:
            initializer = PipelineInitializer()
            components = initializer.initialize_all()
            
            self.components = components  # Store for later tests
            
            self.validation_results["checks"]["initialization"] = {
                "status": "pass",
                "configs_loaded": len(components['configs']),
                "clients_initialized": len(components['clients']),
                "embedding_model": "loaded"
            }
            
            logger.info("  ✓ Pipeline initialization successful")
            
        except Exception as e:
            self.validation_results["checks"]["initialization"] = {
                "status": "fail",
                "error": str(e)
            }
            self.validation_results["errors"].append(f"Initialization failed: {e}")
            logger.error(f"  ✗ Initialization failed: {e}")
            
            # Store None for components
            self.components = None
    
    def _validate_api_connections(self):
        """Test API connections to all LLM providers"""
        logger.info("\n[4/7] Testing API Connections...")
        
        if not self.components:
            logger.error("  ✗ Skipping (initialization failed)")
            return
        
        clients = self.components['clients']
        api_results = {}
        
        # Test Gemini
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(
                "Test",
                generation_config={'max_output_tokens': 5}
            )
            api_results['gemini'] = 'pass'
            logger.info("  ✓ Gemini API: Connected")
        except Exception as e:
            api_results['gemini'] = 'fail'
            logger.error(f"  ✗ Gemini API: {str(e)[:50]}")
        
        # Test Claude (Bedrock or Anthropic)
        use_bedrock = self.components.get('use_bedrock_for_claude', False)
        
        if use_bedrock:
            try:
                import json
                test_body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
                })
                
                model_id = os.getenv("BEDROCK_CLAUDE_MODEL_ID",
                                    "apac.anthropic.claude-sonnet-4-20250514-v1:0")
                
                response = clients['bedrock_runtime'].invoke_model(
                    modelId=model_id,
                    body=test_body
                )
                api_results['claude_bedrock'] = 'pass'
                logger.info("  ✓ Claude (Bedrock): Connected")
            except Exception as e:
                api_results['claude_bedrock'] = 'fail'
                logger.error(f"  ✗ Claude (Bedrock): {str(e)[:50]}")
        else:
            # Skip direct Anthropic test to avoid costs
            api_results['claude_anthropic'] = 'skipped'
            logger.info("  ○ Claude (Anthropic): Skipped (client initialized)")
        
        # Test GPT-5.1
        try:
            response = clients['openai'].chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for testing
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            api_results['gpt'] = 'pass'
            logger.info("  ✓ GPT API: Connected")
        except Exception as e:
            api_results['gpt'] = 'fail'
            logger.error(f"  ✗ GPT API: {str(e)[:50]}")
        
        # Test DeepSeek
        try:
            response = clients['deepseek'].chat.completions.create(
                model="deepseek-chat",  # Use cheaper model for testing
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            api_results['deepseek'] = 'pass'
            logger.info("  ✓ DeepSeek API: Connected")
        except Exception as e:
            api_results['deepseek'] = 'fail'
            logger.error(f"  ✗ DeepSeek API: {str(e)[:50]}")
        
        # Store results
        passed = sum(1 for status in api_results.values() if status == 'pass')
        failed = sum(1 for status in api_results.values() if status == 'fail')
        
        self.validation_results["checks"]["api_connections"] = {
            "status": "pass" if failed == 0 else "fail",
            "results": api_results,
            "passed": passed,
            "failed": failed
        }
        
        if failed > 0:
            self.validation_results["errors"].append(
                f"{failed} API connection(s) failed"
            )
    
    def _validate_databases(self):
        """Test database connections"""
        logger.info("\n[5/7] Testing Database Connections...")
        
        if not self.components:
            logger.error("  ✗ Skipping (initialization failed)")
            return
        
        clients = self.components['clients']
        db_results = {}
        
        # Test Qdrant
        try:
            collections = clients['qdrant'].get_collections()
            db_results['qdrant'] = {
                'status': 'pass',
                'collections': len(collections.collections)
            }
            logger.info(f"  ✓ Qdrant: {len(collections.collections)} collections")
        except Exception as e:
            db_results['qdrant'] = {'status': 'fail', 'error': str(e)}
            logger.error(f"  ✗ Qdrant: {str(e)[:50]}")
        
        # Test Redis
        try:
            clients['redis'].ping()
            # Test read/write
            clients['redis'].set("test_production", "ok", ex=10)
            value = clients['redis'].get("test_production")
            clients['redis'].delete("test_production")
            
            db_results['redis'] = {'status': 'pass'}
            logger.info("  ✓ Redis: Connected and writable")
        except Exception as e:
            db_results['redis'] = {'status': 'fail', 'error': str(e)}
            logger.error(f"  ✗ Redis: {str(e)[:50]}")
        
        # Store results
        failed = sum(1 for r in db_results.values() if r['status'] == 'fail')
        
        self.validation_results["checks"]["databases"] = {
            "status": "pass" if failed == 0 else "fail",
            "results": db_results
        }
        
        if failed > 0:
            self.validation_results["errors"].append(
                f"{failed} database connection(s) failed"
            )
    
    def _validate_models(self):
        """Validate model availability"""
        logger.info("\n[6/7] Validating Models...")
        
        if not self.components:
            logger.error("  ✗ Skipping (initialization failed)")
            return
        
        # Test BGE-M3 embedding model
        try:
            embedding_model = self.components['embedding_model']
            test_embed = embedding_model.encode("test", show_progress_bar=False)
            
            self.validation_results["checks"]["models"] = {
                "status": "pass",
                "bge_m3": {
                    "loaded": True,
                    "dimension": len(test_embed)
                }
            }
            logger.info(f"  ✓ BGE-M3: Loaded (dim: {len(test_embed)})")
            
        except Exception as e:
            self.validation_results["checks"]["models"] = {
                "status": "fail",
                "error": str(e)
            }
            self.validation_results["errors"].append(f"Model validation failed: {e}")
            logger.error(f"  ✗ BGE-M3: {e}")
    
    def _validate_permissions(self):
        """Validate file system permissions"""
        logger.info("\n[7/7] Validating Permissions...")
        
        perm_results = {}
        
        for dir_path in self.critical_dirs:
            full_path = PROJECT_ROOT / dir_path
            
            if not full_path.exists():
                perm_results[dir_path] = "missing"
                continue
            
            # Test write permission
            test_file = full_path / ".permission_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                perm_results[dir_path] = "writable"
                logger.info(f"  ✓ {dir_path}: Writable")
            except Exception as e:
                perm_results[dir_path] = "readonly"
                logger.error(f"  ✗ {dir_path}: Not writable - {e}")
        
        readonly_count = sum(1 for status in perm_results.values() if status == "readonly")
        
        self.validation_results["checks"]["permissions"] = {
            "status": "pass" if readonly_count == 0 else "fail",
            "results": perm_results
        }
        
        if readonly_count > 0:
            self.validation_results["errors"].append(
                f"{readonly_count} directory/directories not writable"
            )
    
    def _determine_overall_status(self):
        """Determine overall validation status"""
        error_count = len(self.validation_results["errors"])
        warning_count = len(self.validation_results["warnings"])
        
        if error_count == 0:
            if warning_count == 0:
                self.validation_results["overall_status"] = "ready"
            else:
                self.validation_results["overall_status"] = "ready_with_warnings"
        else:
            self.validation_results["overall_status"] = "not_ready"
    
    def _print_report(self):
        """Print validation report"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION REPORT")
        logger.info("=" * 80)
        
        # Overall status
        status = self.validation_results["overall_status"]
        status_emoji = {
            "ready": "✅",
            "ready_with_warnings": "⚠️",
            "not_ready": "❌"
        }
        
        logger.info(f"\nOverall Status: {status_emoji.get(status, '?')} {status.upper()}")
        
        # Checks summary
        logger.info("\nChecks Summary:")
        for check_name, check_data in self.validation_results["checks"].items():
            status_mark = "✓" if check_data["status"] == "pass" else "✗"
            logger.info(f"  {status_mark} {check_name}: {check_data['status']}")
        
        # Errors
        if self.validation_results["errors"]:
            logger.error("\nErrors:")
            for error in self.validation_results["errors"]:
                logger.error(f"  ✗ {error}")
        
        # Warnings
        if self.validation_results["warnings"]:
            logger.warning("\nWarnings:")
            for warning in self.validation_results["warnings"]:
                logger.warning(f"  ⚠ {warning}")
        
        logger.info("\n" + "=" * 80)
    
    def save_report(self, output_file: str = "validation_report.json"):
        """Save validation report to file"""
        output_path = PROJECT_ROOT / output_file
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info(f"Validation report saved to: {output_file}")


def main():
    """
    Main entry point
    """
    validator = ProductionValidator()
    results = validator.run_all_validations()
    
    # Save report
    validator.save_report()
    
    # Exit with appropriate code
    if results["overall_status"] == "not_ready":
        logger.error("\n❌ Production environment NOT READY")
        sys.exit(1)
    elif results["overall_status"] == "ready_with_warnings":
        logger.warning("\n⚠️ Production environment READY (with warnings)")
        sys.exit(0)
    else:
        logger.info("\n✅ Production environment READY")
        sys.exit(0)


if __name__ == "__main__":
    main()