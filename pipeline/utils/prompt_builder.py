"""
============================================================================
PROMPT BUILDER UTILITY
============================================================================
Purpose: Load prompt templates and fill variables to construct complete prompts
Features:
    - Load prompt templates from files
    - Fill template variables with dynamic data
    - Validate all required variables are provided
    - Handle missing variables gracefully
    - Token estimation for prompts
    - Template caching for performance

Usage:
    from utils.prompt_builder import PromptBuilder
    
    builder = PromptBuilder(prompts_config)
    
    # Build prompt from template
    prompt = builder.build_prompt(
        prompt_name="base_system",
        variables={
            "rag_context": "...",
            "question_text": "...",
            "options": "..."
        }
    )

Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import re


# Robust import for logging
try:
    from pipeline.utils.logging_utils import setup_logger
except ImportError:
    from utils.logging_utils import setup_logger

logger = setup_logger("prompt_builder")


class PromptBuilder:
    """
    Build prompts from templates with variable substitution
    
    Template Variables:
    - Use {variable_name} syntax for variables
    - Example: "Question: {question_text}\nOptions: {options}"
    
    Templates are loaded from prompts/ directory based on prompts_config.yaml
    """
    
    def __init__(self, prompts_config: Dict[str, Any]):
        """
        Args:
            prompts_config: Prompts configuration dictionary
        """
        self.prompts_config = prompts_config
        
        # Base directory for prompts
        self.prompts_dir = Path(__file__).parent.parent.parent
        
        # Cache for loaded templates
        self.template_cache = {}
        
        # Load all templates into cache
        self._load_all_templates()
    
    def _load_all_templates(self):
        """Load all prompt templates into cache"""
        # Support both 'prompts' (from yaml) and 'templates' (legacy/test)
        templates = self.prompts_config.get('prompts') or self.prompts_config.get('templates', {})
        
        for template_name, template_info in templates.items():
            file_path = template_info.get('file')
            
            if not file_path:
                logger.warning(f"No file path for template: {template_name}")
                continue
            
            # Load template
            template = self._load_template_file(file_path)
            
            if template:
                self.template_cache[template_name] = {
                    'template': template,
                    'file_path': file_path,
                    'estimated_tokens': template_info.get('estimated_tokens', 0),
                    'variables': template_info.get('variables', [])
                }
                
                logger.debug(f"Loaded template: {template_name} ({len(template)} chars)")
            else:
                logger.error(f"Failed to load template: {template_name} from {file_path}")
    
    def _load_template_file(self, file_path: str) -> Optional[str]:
        """
        Load template from file
        
        Args:
            file_path: Relative path from prompts directory
        
        Returns:
            str: Template content or None if failed
        """
        full_path = self.prompts_dir / file_path
        
        if not full_path.exists():
            logger.error(f"Template file not found: {full_path}")
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            return template
            
        except Exception as e:
            logger.error(f"Failed to load template {file_path}: {e}")
            return None
    
    def build_prompt(
        self,
        prompt_name: str,
        variables: Dict[str, Any],
        validate: bool = True
    ) -> str:
        """
        Build prompt from template with variable substitution
        
        Args:
            prompt_name: Name of prompt template (from config)
            variables: Dictionary of variables to substitute
            validate: Validate all required variables are provided
        
        Returns:
            str: Complete prompt with variables filled
        
        Raises:
            ValueError: If template not found or missing required variables
        
        Example:
            prompt = builder.build_prompt(
                prompt_name="classification",
                variables={
                    "question_text": "What is the fuel-air ratio?",
                    "options": "A. 0.5\nB. 1.0\nC. 1.5",
                    "has_image": "false"
                }
            )
        """
        # Get template from cache
        if prompt_name not in self.template_cache:
            raise ValueError(f"Template not found: {prompt_name}")
        
        template_info = self.template_cache[prompt_name]
        template = template_info['template']
        required_variables = template_info.get('variables', [])
        
        # Validate required variables
        if validate:
            missing = [v for v in required_variables if v not in variables]
            
            if missing:
                logger.warning(
                    f"Missing variables for {prompt_name}: {missing}. "
                    f"Using empty strings."
                )
                
                # Fill missing with empty strings
                for var in missing:
                    variables[var] = ""
        
        # Escape any literal braces in variable values to prevent format errors
        # This is critical when question_text, rag_context, etc. contain { or }
        escaped_variables = {}
        for key, value in variables.items():
            if isinstance(value, str):
                # Escape { and } by doubling them
                escaped_variables[key] = value.replace('{', '{{').replace('}', '}}')
            else:
                escaped_variables[key] = value
        
        # Fill template variables
        try:
            # Python's .format() treats {{ as escaped { and }} as escaped }
            # So double braces in template will render as single braces
            prompt = template.format(**escaped_variables)
            
            logger.debug(
                f"Built prompt '{prompt_name}': {len(prompt)} chars, "
                f"~{len(prompt) // 4} tokens"
            )
            
            return prompt
            
        except (KeyError, IndexError) as e:
            # More helpful error message
            missing_var = str(e).strip("'")
            raise ValueError(
                f"Missing variable in template {prompt_name}: '{missing_var}'. "
                f"Required variables: {required_variables}, "
                f"Provided: {list(variables.keys())}"
            )
    
    def get_template_info(self, prompt_name: str) -> Dict[str, Any]:
        """
        Get information about a template
        
        Args:
            prompt_name: Name of template
        
        Returns:
            dict: Template information
        """
        if prompt_name not in self.template_cache:
            return {}
        
        info = self.template_cache[prompt_name]
        
        return {
            'file_path': info['file_path'],
            'estimated_tokens': info['estimated_tokens'],
            'required_variables': info.get('variables', []),
            'template_length': len(info['template'])
        }
    
    def list_templates(self) -> List[str]:
        """
        List all available templates
        
        Returns:
            list: Template names
        """
        return list(self.template_cache.keys())
    
    def estimate_tokens(self, prompt: str) -> int:
        """
        Estimate tokens in prompt
        
        Rule of thumb: ~4 characters per token for English
        
        Args:
            prompt: Prompt text
        
        Returns:
            int: Estimated token count
        """
        return len(prompt) // 4
    
    def extract_variables(self, template: str) -> List[str]:
        """
        Extract variable names from template
        
        Args:
            template: Template string with {variable} placeholders
        
        Returns:
            list: Variable names
        
        Example:
            template = "Question: {question_text}\nAnswer: {answer}"
            extract_variables(template)
            → ["question_text", "answer"]
        """
        # Find all {variable} patterns
        # Use lookaround to ensure we don't match {{variable}} or {{ variable }} which are escaped
        # Double braces {{ }} are escaped in Python format strings and should be ignored
        pattern = r'(?<!\{)\{([a-zA-Z0-9_]+)\}(?!\})'
        variables = re.findall(pattern, template)
        
        # Return unique variables
        return list(set(variables))
    
    def validate_template(self, prompt_name: str) -> Dict[str, Any]:
        """
        Validate template is properly formatted
        
        Args:
            prompt_name: Template name
        
        Returns:
            dict: {
                "valid": bool,
                "issues": [list of issues],
                "variables_found": [list],
                "variables_declared": [list]
            }
        """
        if prompt_name not in self.template_cache:
            return {
                "valid": False,
                "issues": [f"Template '{prompt_name}' not found"],
                "variables_found": [],
                "variables_declared": []
            }
        
        template_info = self.template_cache[prompt_name]
        template = template_info['template']
        declared_vars = template_info.get('variables', [])
        
        # Extract variables from template
        found_vars = self.extract_variables(template)
        
        issues = []
        
        # Check for undeclared variables
        undeclared = set(found_vars) - set(declared_vars)
        if undeclared:
            issues.append(f"Undeclared variables: {undeclared}")
        
        # Check for unused declared variables
        unused = set(declared_vars) - set(found_vars)
        if unused:
            issues.append(f"Declared but unused variables: {unused}")
        
        # Check for unclosed braces
        if template.count('{') != template.count('}'):
            issues.append("Unmatched braces in template")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "variables_found": found_vars,
            "variables_declared": declared_vars
        }
    
    def preview_prompt(
        self,
        prompt_name: str,
        variables: Dict[str, Any],
        max_length: int = 500
    ) -> str:
        """
        Preview prompt with truncation (for logging/debugging)
        
        Args:
            prompt_name: Template name
            variables: Variables to fill
            max_length: Max length to show
        
        Returns:
            str: Truncated prompt preview
        """
        prompt = self.build_prompt(prompt_name, variables, validate=False)
        
        if len(prompt) <= max_length:
            return prompt
        
        # Truncate with ellipsis
        half = max_length // 2
        return prompt[:half] + "\n...\n" + prompt[-half:]
    
    def reload_template(self, prompt_name: str) -> bool:
        """
        Reload template from file (useful for development)
        
        Args:
            prompt_name: Template name
        
        Returns:
            bool: Success
        """
        if prompt_name not in self.template_cache:
            logger.error(f"Template not found: {prompt_name}")
            return False
        
        file_path = self.template_cache[prompt_name]['file_path']
        
        template = self._load_template_file(file_path)
        
        if template:
            self.template_cache[prompt_name]['template'] = template
            logger.info(f"Reloaded template: {prompt_name}")
            return True
        
        return False


# Example usage
if __name__ == "__main__":
    import yaml
    
    print("Testing Prompt Builder\n")
    
    # Mock prompts config
    prompts_config = {
        "templates": {
            "base_system": {
                "file": "system_prompt_base.txt",
                "estimated_tokens": 4000,
                "variables": [
                    "rag_context",
                    "question_text",
                    "options",
                    "year",
                    "marks"
                ]
            },
            "classification": {
                "file": "classification_prompt.txt",
                "estimated_tokens": 900,
                "variables": [
                    "question_text",
                    "options",
                    "has_image"
                ]
            }
        }
    }
    
    # Create builder
    builder = PromptBuilder(prompts_config)
    
    # List templates
    print("1. Available Templates:")
    for template in builder.list_templates():
        info = builder.get_template_info(template)
        print(f"  - {template}:")
        print(f"      File: {info['file_path']}")
        print(f"      Est. tokens: {info['estimated_tokens']}")
        print(f"      Variables: {info['required_variables']}")
    
    # Build prompt (will fail if templates don't exist, which is expected in test)
    print("\n2. Build Prompt (example):")
    try:
        prompt = builder.build_prompt(
            prompt_name="classification",
            variables={
                "question_text": "What is the fuel-air ratio?",
                "options": "A. 0.5\nB. 1.0",
                "has_image": "false"
            }
        )
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Estimated tokens: {builder.estimate_tokens(prompt)}")
    except Exception as e:
        print(f"  Expected error (templates not created yet): {e}")
    
    # Validate template
    print("\n3. Validate Template:")
    for template in builder.list_templates():
        validation = builder.validate_template(template)
        print(f"  {template}: {'✓ Valid' if validation['valid'] else '✗ Invalid'}")
        if validation['issues']:
            for issue in validation['issues']:
                print(f"    - {issue}")