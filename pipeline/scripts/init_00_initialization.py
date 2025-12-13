"""
============================================================================
STAGE 0: INITIALIZATION
============================================================================
Purpose: Load all configurations, connect to external services, initialize models
Used by: 99_pipeline_runner.py (called once at startup)
Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import sys
import yaml
import redis
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from openai import OpenAI
import boto3
from pathlib import Path

# Add project root to path
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

try:
    from pipeline.utils.logging_utils import setup_logger, log_stage
except ImportError:
    sys.path.append(str(current_file.parent.parent))
    from utils.logging_utils import setup_logger, log_stage

logger = setup_logger("00_initialization")


class PipelineInitializer:
    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or os.path.join(PROJECT_ROOT, "config")
        self.configs = {}
        self.clients = {}
        self.embedding_model = None
        self.use_bedrock_for_claude = False
        
    @log_stage("Stage 0: Initialization")
    def initialize_all(self):
        logger.info("=" * 80)
        logger.info("GATE AE SOTA PIPELINE - INITIALIZATION")
        logger.info("=" * 80)
        
        self._load_configs()
        self._validate_environment()
        self._initialize_qdrant()
        self._initialize_redis()
        self._initialize_llm_clients()
        self._initialize_embeddings()
        self._test_connections()
        
        logger.info("✓ Initialization complete - all systems ready")
        
        return {
            "configs": self.configs,
            "clients": self.clients,
            "embedding_model": self.embedding_model,
            "use_bedrock_for_claude": self.use_bedrock_for_claude
        }
    
    def _load_configs(self):
        logger.info("Loading configuration files...")
        config_files = ["models_config.yaml", "weights_config.yaml", "thresholds_config.yaml", "prompts_config.yaml"]
        
        for config_file in config_files:
            config_path = os.path.join(self.config_dir, config_file)
            if not os.path.exists(config_path):
                # Fallback to absolute path check
                if os.path.exists(os.path.join(PROJECT_ROOT, "config", config_file)):
                    config_path = os.path.join(PROJECT_ROOT, "config", config_file)
                else:
                    logger.warning(f"Config file not found: {config_file}")
                    continue
            
            with open(config_path, 'r') as f:
                config_name = config_file.replace('.yaml', '')
                self.configs[config_name] = yaml.safe_load(f)
                logger.info(f"  ✓ Loaded {config_file}")
        
        logger.info(f"✓ Loaded configuration files")
    
    def _validate_environment(self):
        logger.info("Validating environment variables...")
        
        # --- LOGIC FIX: SMART DETECTION ---
        env_flag = os.getenv('USE_BEDROCK_FOR_CLAUDE', '').lower()
        has_anthropic_key = bool(os.getenv('ANTHROPIC_API_KEY'))
        has_aws_keys = bool(os.getenv('AWS_ACCESS_KEY_ID'))
        
        if env_flag == 'true':
            self.use_bedrock_for_claude = True
        elif has_aws_keys and not has_anthropic_key:
            logger.info("Auto-detecting Bedrock usage (AWS keys present, Anthropic key missing)")
            self.use_bedrock_for_claude = True
        else:
            self.use_bedrock_for_claude = False
        # ----------------------------------

        required_env_vars = [
            "GOOGLE_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "QDRANT_URL",
            "REDIS_URL",
        ]
        
        if self.use_bedrock_for_claude:
            logger.info("  Using AWS Bedrock for Claude")
            required_env_vars.extend(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"])
        else:
            logger.info("  Using direct Anthropic API for Claude")
            required_env_vars.append("ANTHROPIC_API_KEY")
        
        missing_required = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_required:
            logger.error(f"Missing required environment variables: {missing_required}")
            # Don't crash for index building, just warn
            if "build_bm25" in sys.argv[0]:
                logger.warning("Proceeding despite missing keys (Index Building Mode)")
            else:
                raise EnvironmentError(f"Please set: {', '.join(missing_required)}")
        
        logger.info("✓ Environment validated")
    
    def _initialize_qdrant(self):
        logger.info("Connecting to Qdrant vector database...")
        try:
            self.clients['qdrant'] = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                timeout=30
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
    
    def _initialize_redis(self):
        logger.info("Connecting to Redis cache...")
        try:
            self.clients['redis'] = redis.from_url(
                os.getenv("REDIS_URL"),
                decode_responses=True,
                socket_timeout=10
            )
            # Test connection
            try:
                self.clients['redis'].ping()
                logger.info("  ✓ Connected to Redis")
            except:
                logger.warning("  ⚠ Redis connection failed (is it running?)")
        except Exception as e:
            logger.error(f"Failed to init Redis client: {e}")
    

    
    def _initialize_llm_clients(self):
        logger.info("Initializing LLM API clients...")
        
        # 1. Claude
        if self.use_bedrock_for_claude:
            try:
                from botocore.config import Config
                
                # Custom config with increased timeout for large prompts
                bedrock_config = Config(
                    read_timeout=120,  # 120 seconds for large prompts
                    connect_timeout=10,
                    retries={'max_attempts': 3}
                )
                
                self.clients['bedrock_runtime'] = boto3.client(
                    service_name='bedrock-runtime',
                    region_name='us-east-1',
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    config=bedrock_config
                )
                self.clients['anthropic'] = self.clients['bedrock_runtime']
                logger.info("  ✓ AWS Bedrock Runtime initialized (aliased to 'anthropic')")
            except Exception as e:
                logger.error(f"Failed to initialize AWS Bedrock: {e}")
        else:
            # Fallback placeholder if not using bedrock
            self.clients['anthropic'] = None

        # 2. Google
        if os.getenv("GOOGLE_API_KEY"):
            try:
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.clients['google_genai'] = genai
                logger.info("  ✓ Google GenAI configured")
            except: pass

        # 3. OpenAI/DeepSeek
        if os.getenv("OPENAI_API_KEY"):
            self.clients['openai'] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if os.getenv("DEEPSEEK_API_KEY"):
             self.clients['deepseek'] = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
    
    def _initialize_embeddings(self):
        # FIX: Get model name from config or default to the one we used for indexing
        config_model_name = "BAAI/bge-large-en-v1.5"
        
        try:
            # Try to get from config if available
            if 'models_config' in self.configs:
                config_model_name = self.configs['models_config'].get('embedding_model', {}).get('name', config_model_name)
        except:
            pass

        logger.info(f"Loading embedding model: {config_model_name}...")
        logger.info("  This may take 30-60 seconds on first run...")
        
        try:
            # Load embedding model (this can take 30-60s on first run)
            self.embedding_model = SentenceTransformer(config_model_name, device='cpu')
            logger.info(f"  [OK] Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.warning("  Pipeline will continue but dense retrieval may fail")

    def _test_connections(self):
        pass

if __name__ == "__main__":
    init = PipelineInitializer()
    init.initialize_all()