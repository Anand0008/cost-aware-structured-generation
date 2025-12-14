"""
============================================================================
SETUP.PY - PACKAGE INSTALLATION
============================================================================
Purpose: Package installation and distribution configuration
Usage: pip install -e .  (development mode)
       pip install .     (production install)
============================================================================
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = []

# Development dependencies
dev_requirements = [
    'pytest>=8.3.4',
    'pytest-cov>=6.0.0',
    'pytest-asyncio>=0.24.0',
    'pytest-xdist>=3.5.0',
    'pytest-timeout>=2.2.0',
    'black>=24.10.0',
    'flake8>=7.1.1',
    'mypy>=1.13.0',
    'isort>=5.13.2',
    'ipython>=8.29.0',
    'jupyter>=1.1.1',
]

setup(
    # ========================================================================
    # PACKAGE METADATA
    # ========================================================================
    name="gate-ae-pipeline",
    version="1.0.0",
    description="GATE AE SOTA Question Tagging Pipeline - Multi-model consensus system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GATE AE Pipeline Team",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/gate-ae-pipeline",
    license="MIT",
    
    # ========================================================================
    # PACKAGE DISCOVERY
    # ========================================================================
    packages=find_packages(
        include=['pipeline', 'pipeline.*'],
        exclude=['tests', 'tests.*', 'docs', 'aws', 'docker']
    ),
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        'pipeline': [
            '../config/*.yaml',
            '../prompts/*.txt',
        ]
    },
    
    # ========================================================================
    # DEPENDENCIES
    # ========================================================================
    install_requires=requirements,
    
    extras_require={
        'dev': dev_requirements,
        'test': [
            'pytest>=8.3.4',
            'pytest-cov>=6.0.0',
            'pytest-asyncio>=0.24.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=2.0.0',
        ],
        'all': dev_requirements,
    },
    
    # ========================================================================
    # PYTHON VERSION
    # ========================================================================
    python_requires='>=3.11',
    
    # ========================================================================
    # CLASSIFIERS
    # ========================================================================
    classifiers=[
        # Development Status
        'Development Status :: 4 - Beta',
        
        # Intended Audience
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        
        # Topic
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Education',
        
        # License
        'License :: OSI Approved :: MIT License',
        
        # Programming Language
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        
        # Operating System
        'Operating System :: OS Independent',
        
        # Natural Language
        'Natural Language :: English',
    ],
    
    # ========================================================================
    # KEYWORDS
    # ========================================================================
    keywords=[
        'gate',
        'aerospace-engineering',
        'question-tagging',
        'multi-model',
        'consensus',
        'llm',
        'rag',
        'retrieval',
        'education',
        'ai',
        'machine-learning',
    ],
    
    # ========================================================================
    # ENTRY POINTS
    # ========================================================================
    entry_points={
        'console_scripts': [
            # Main pipeline runner
            'gate-ae-pipeline=pipeline.scripts.init_99_pipeline_runner:main',
            
            # Individual stages
            'gate-ae-init=pipeline.scripts.init_00_initialization:main',
            'gate-ae-classify=pipeline.scripts.init_02_question_classifier:main',
            'gate-ae-retrieve=pipeline.scripts.init_04_retrieval_dense:main',
            
            # Utilities
            'gate-ae-validate=pipeline.scripts.init_00_setup_production:main',
            'gate-ae-health=pipeline.utils.health_monitor:main',
        ],
    },
    
    # ========================================================================
    # PROJECT URLS
    # ========================================================================
    project_urls={
        'Documentation': 'https://github.com/yourusername/gate-ae-pipeline#readme',
        'Source': 'https://github.com/yourusername/gate-ae-pipeline',
        'Bug Reports': 'https://github.com/yourusername/gate-ae-pipeline/issues',
        'Changelog': 'https://github.com/yourusername/gate-ae-pipeline/blob/main/CHANGELOG.md',
    },
    
    # ========================================================================
    # ZIP SAFE
    # ========================================================================
    zip_safe=False,
)

# ============================================================================
# POST-INSTALL INSTRUCTIONS
# ============================================================================
"""
After installation:

1. Install package in development mode:
   pip install -e .

2. Install with all extras:
   pip install -e ".[all]"

3. Install for testing only:
   pip install -e ".[test]"

4. Verify installation:
   gate-ae-pipeline --help
   gate-ae-validate

5. Configure environment:
   cp .env.example .env
   # Edit .env with your API keys

6. Run validation:
   gate-ae-validate

7. Run pipeline:
   gate-ae-pipeline --question-id GATE_AE_2024_Q1

============================================================================
DEVELOPMENT WORKFLOW
============================================================================

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Format code
black pipeline/ tests/
isort pipeline/ tests/

# Lint code
flake8 pipeline/ tests/
mypy pipeline/

# Run tests
pytest

# Run tests with coverage
pytest --cov=pipeline --cov-report=html

# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI (requires twine)
twine upload dist/*

============================================================================
UNINSTALL
============================================================================

pip uninstall gate-ae-pipeline

============================================================================
"""