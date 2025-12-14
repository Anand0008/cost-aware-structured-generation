"""
============================================================================
SETUP SCRIPT: BUILD BM25 INDEX (ROBUST IMPORT VERSION - FIXED PATHS)
============================================================================
"""

import os
import sys
import json
import time
import importlib.util
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# 1. Load Environment Variables
# Define exact path to .env file
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

print(f"🔌 Loading environment from: {ENV_PATH}")
loaded = load_dotenv(dotenv_path=ENV_PATH, override=True)
print(f"🔌 Environment loaded: {loaded}")

if not loaded:
    print("❌ ERROR: Could not load .env file.")
    sys.exit(1)

# Setup Python Path
sys.path.append(str(PROJECT_ROOT))

# Helper to import modules dynamically
def load_module_from_path(module_name, file_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Could not load spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"❌ Error loading {module_name} from {file_path}: {e}")
        sys.exit(1)

# Import necessary modules (UPDATED FILENAMES)
# Note: Using init_ prefix based on your file structure
init_path = PROJECT_ROOT / "pipeline" / "scripts" / "init_00_initialization.py"
sparse_path = PROJECT_ROOT / "pipeline" / "scripts" / "init_05_retrieval_sparse.py"
logging_path = PROJECT_ROOT / "pipeline" / "utils" / "logging_utils.py"

logging_module = load_module_from_path("logging_utils", logging_path)
init_module = load_module_from_path("pipeline_initializer", init_path)
sparse_module = load_module_from_path("sparse_retriever", sparse_path)

setup_logger = logging_module.setup_logger
PipelineInitializer = init_module.PipelineInitializer
SparseRetriever = sparse_module.SparseRetriever

logger = setup_logger("build_bm25_index")

def load_chunks(file_path: Path) -> list:
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    logger.info("="*80)
    logger.info("BUILDING BM25 INDEX IN REDIS")
    logger.info("="*80)

    # 1. Initialize
    try:
        config_dir = PROJECT_ROOT / "config"
        initializer = PipelineInitializer(config_dir=str(config_dir))
        components = initializer.initialize_all()
        redis_client = components['clients']['redis']
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return
    
    # 2. Initialize Retriever
    retriever = SparseRetriever(
        redis_client=redis_client,
        configs=components['configs']
    )
    
    # 3. Check existing
    try:
        stats = retriever.get_index_stats()
        total = stats.get('total_documents', 0) if isinstance(stats, dict) else 0
        if total > 0:
            logger.warning(f"Index contains {total} documents. Updating...")
    except:
        pass

    # 4. Load Data
    data_dir = PROJECT_ROOT / "data" / "processed"
    books_path = data_dir / "book_chunks_rich.json"
    videos_path = data_dir / "video_chunks_rich.json"

    books = load_chunks(books_path)
    videos = load_chunks(videos_path)
    all_chunks = books + videos
    
    if not all_chunks:
        logger.error(f"No chunks loaded from {data_dir}")
        return

    logger.info(f"Loaded {len(all_chunks)} chunks.")

    # 5. Index
    for chunk in tqdm(all_chunks, desc="Indexing"):
        metadata = chunk.get('metadata', {})
        concept = metadata.get('concept_name', '')
        text_content = chunk.get('text', '')
        full_text = f"{concept} {text_content}"
        
        source_name = metadata.get('book') or metadata.get('video_source') or "Unknown"
        source_type = metadata.get('source_type', 'unknown')

        retriever.index_document(
            doc_id=chunk['chunk_id'],
            text=full_text,
            source_type=source_type,
            source_name=source_name,
            metadata=metadata
        )

    # 6. Finalize
    retriever.finalize_index()
    logger.info("Indexing complete.")
    
if __name__ == "__main__":
    main()