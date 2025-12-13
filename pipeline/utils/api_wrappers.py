"""
============================================================================
API WRAPPERS UTILITY
============================================================================
Purpose: Unified wrappers for all LLM API calls with error handling
Features:
    - Consistent interface across all models
    - Retry logic with exponential backoff
    - Token usage tracking
    - Error handling and logging
    - Rate limit handling
    - Image support for vision models

Supported Models:
    - Gemini 2.5 Pro (Google GenAI)
    - Gemini 2.0 Flash Exp (Google GenAI)
    - Claude Sonnet 4.5 (Anthropic / AWS Bedrock)
    - GPT-5.1 / GPT-4o (OpenAI)
    - DeepSeek R1 (DeepSeek API, OpenAI-compatible)

Usage:
    from pipeline.utils.api_wrappers import call_gemini, call_claude, call_gpt, call_deepseek
    
    response, usage = call_gemini(
        prompt="Explain quantum computing",
        client=google_genai_client,
        model_name="gemini-2.5-pro",
        temperature=0.3
    )

Author: GATE AE SOTA Pipeline
============================================================================
"""

import time
import json
from typing import Dict, Any, Optional, Tuple, List
import base64

# Robust import for logging
try:
    from pipeline.utils.logging_utils import setup_logger
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from pipeline.utils.logging_utils import setup_logger

logger = setup_logger("api_wrappers")


class APIError(Exception):
    """Custom exception for API errors"""
    pass


def retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    Retry function with exponential backoff
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
    
    Returns:
        Function result
    
    Raises:
        Exception: If all retries fail
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed
                raise
            
            logger.warning(
                f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            
            time.sleep(delay)
            delay *= backoff_factor


def call_gemini(
    prompt: str,
    client,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.3,
    max_tokens: int = 4000,
    image_base64: Optional[str] = None,
    images: Optional[List[str]] = None,
    max_retries: int = 3
) -> Tuple[str, Dict[str, int]]:
    """
    Call Google Gemini API
    
    Args:
        images: List of base64 encoded images (optional)
        image_base64: Legacy/Single base64 image (optional)
    """
    # Normalize images
    images_list = images or []
    if image_base64:
        images_list.append(image_base64)

    def _call():
        try:
            # Get model
            model = client.GenerativeModel(model_name)
            
            content_parts = [prompt]
            
            if images_list:
                import PIL.Image
                import io
                
                for img_b64 in images_list:
                    # Decode base64 image
                    image_bytes = base64.b64decode(img_b64)
                    image = PIL.Image.open(io.BytesIO(image_bytes))
                    content_parts.append(image)
                
            # Generate
            response = model.generate_content(
                content_parts,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )

            
            # Extract response text
            response_text = response.text
            
            # Extract token usage
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            
            return response_text, usage
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise APIError(f"Gemini call failed: {e}")
    
    return retry_with_backoff(_call, max_retries=max_retries)


def call_claude(
    prompt: str,
    client,
    model_name: str = "anthropic.claude-sonnet-4.5-v2:0",
    temperature: float = 0.3,
    max_tokens: int = 4000,
    image_base64: Optional[str] = None,
    images: Optional[List[str]] = None,
    max_retries: int = 3,
    use_bedrock: bool = None
) -> Tuple[str, Dict[str, int]]:
    """Call Claude API (Supports both AWS Bedrock and Anthropic Direct)"""
    # Normalize images
    images_list = images or []
    if image_base64:
        images_list.append(image_base64)

    def _call():
        try:
            # Auto-detect Bedrock client if not specified
            is_bedrock = use_bedrock
            if is_bedrock is None:
                is_bedrock = hasattr(client, 'invoke_model')
            
            # Prepare messages
            msgs_content = []
            
            # Add images first (usually convention)
            for img_b64 in images_list:
                if img_b64.startswith('/9j/'): 
                    media_type = "image/jpeg"
                elif img_b64.startswith('iVBORw'):
                    media_type = "image/png"
                else:
                    media_type = "image/jpeg"
                
                msgs_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64
                    }
                })
            
            # Add text prompt
            msgs_content.append({
                "type": "text",
                "text": prompt
            })

            if is_bedrock:
                # --- AWS BEDROCK PATH ---
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {"role": "user", "content": msgs_content}
                    ]
                })
                
                response = client.invoke_model(
                    body=body,
                    modelId=model_name,
                    contentType="application/json",
                    accept="application/json"
                )
                
                response_body = json.loads(response.get('body').read())
                response_text = response_body['content'][0]['text']
                usage = {
                    "input_tokens": response_body['usage']['input_tokens'],
                    "output_tokens": response_body['usage']['output_tokens']
                }
                
            else:
                # --- DIRECT ANTHROPIC PATH ---
                # Anthropic API expects prompt as string if text-only, or list if image
                # (Actually modern SDK handles list for text-only too, but let's be safe)
                if not image_base64:
                    # Some versions prefer string content for text-only
                    # But message structure is robust.
                    pass 

                response = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": msgs_content}
                    ]
                )
                
                response_text = response.content[0].text
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            
            return response_text, usage
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise APIError(f"Claude call failed: {e}")
    
    return retry_with_backoff(_call, max_retries=max_retries)


def call_gpt(
    prompt: str,
    client,
    model_name: str = "gpt-5.1",
    temperature: float = 0.3,
    max_tokens: int = 4000,
    image_base64: Optional[str] = None,
    images: Optional[List[str]] = None,
    max_retries: int = 3
) -> Tuple[str, Dict[str, int]]:
    """Call OpenAI GPT API"""
    # Normalize images
    images_list = images or []
    if image_base64:
        images_list.append(image_base64)

    def _call():
        try:
            # --- GPT-5.1 SPECIFIC PATH (Responses API) ---
            if "gpt-5.1" in model_name:
                # Responses API format
                content_list = [{"type": "input_text", "text": prompt}]
                
                for img_b64 in images_list:
                    if img_b64.startswith('/9j/'): media_type = "image/jpeg"
                    elif img_b64.startswith('iVBORw'): media_type = "image/png"
                    else: media_type = "image/jpeg"
                        
                    content_list.append({
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{img_b64}"
                    })
                
                input_payload = [{"role": "user", "content": content_list}]

                response = client.responses.create(
                    model=model_name,
                    input=input_payload,
                    temperature=temperature,
                    max_output_tokens=max_tokens 
                )
                
                response_text = response.output_text
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }

            # --- STANDARD GPT-4/3.5 PATH (Chat Completions API) ---
            else:
                content_payload = [{"type": "text", "text": prompt}]
                
                for img_b64 in images_list:
                    if img_b64.startswith('/9j/'): image_url = f"data:image/jpeg;base64,{img_b64}"
                    elif img_b64.startswith('iVBORw'): image_url = f"data:image/png;base64,{img_b64}"
                    else: image_url = f"data:image/jpeg;base64,{img_b64}"
                    
                    content_payload.append({
                        "type": "image_url", 
                        "image_url": {"url": image_url}
                    })

                if isinstance(content_payload, str):
                    messages_payload = [{"role": "user", "content": content_payload}]
                else:
                    messages_payload = [{"role": "user", "content": content_payload}]

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages_payload,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                response_text = response.choices[0].message.content
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                }
            
            return response_text, usage
            
        except Exception as e:
            logger.error(f"GPT API error ({model_name}): {e}")
            raise APIError(f"GPT call failed: {e}")
    
    return retry_with_backoff(_call, max_retries=max_retries)


def call_deepseek(
    prompt: str,
    client,
    model_name: str = "deepseek-reasoner",
    temperature: float = 0.3,
    max_tokens: int = 4000,
    max_retries: int = 3
) -> Tuple[str, Dict[str, int]]:
    """
    Call DeepSeek API (OpenAI-compatible)
    
    Note: DeepSeek R1 does NOT support image input
    Image descriptions should be included in the prompt text
    
    Args:
        prompt: Text prompt (include image description if needed)
        client: DeepSeek client (OpenAI-compatible)
        model_name: Model name (deepseek-reasoner, deepseek-chat)
        temperature: Temperature (0.0-1.0)
        max_tokens: Max output tokens
        max_retries: Number of retries on failure
    
    Returns:
        tuple: (response_text, usage_dict)
    
    Raises:
        APIError: If API call fails after retries
    """
    def _call():
        try:
            # DeepSeek uses OpenAI-compatible API
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Extract token usage
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
            
            return response_text, usage
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise APIError(f"DeepSeek call failed: {e}")
    
    return retry_with_backoff(_call, max_retries=max_retries)


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of tokens in text
    
    Rule of thumb: ~4 characters per token for English
    This is approximate and varies by model
    
    Args:
        text: Input text
    
    Returns:
        int: Estimated token count
    """
    return len(text) // 4


def validate_response_format(
    response: str,
    expected_format: str = "json"
) -> bool:
    """
    Validate response format
    
    Args:
        response: Response text
        expected_format: Expected format ("json", "text")
    
    Returns:
        bool: True if valid, False otherwise
    """
    if expected_format == "json":
        import json
        try:
            # Try to parse as JSON
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return False
    
    # For text, just check it's not empty
    return len(response.strip()) > 0


# Example usage
if __name__ == "__main__":
    import os
    import google.generativeai as genai
    from anthropic import Anthropic
    from openai import OpenAI
    
    print("Testing API Wrappers\n")
    
    # Test Gemini (if API key available)
    if os.getenv("GOOGLE_API_KEY"):
        print("Testing Gemini...")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        try:
            response, usage = call_gemini(
                prompt="What is 2+2? Respond with just the number.",
                client=genai,
                model_name="gemini-2.0-flash-exp",
                max_tokens=10
            )
            print(f"✓ Gemini Response: {response}")
            print(f"  Usage: {usage}")
        except Exception as e:
            print(f"✗ Gemini Error: {e}")
    
    # Test Claude (if API key available)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\nTesting Claude...")
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        try:
            response, usage = call_claude(
                prompt="What is 2+2? Respond with just the number.",
                client=anthropic_client,
                model_name="claude-sonnet-4-20250514",
                max_tokens=10
            )
            print(f"✓ Claude Response: {response}")
            print(f"  Usage: {usage}")
        except Exception as e:
            print(f"✗ Claude Error: {e}")
    
    # Test GPT (if API key available)
    if os.getenv("OPENAI_API_KEY"):
        print("\nTesting GPT...")
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        try:
            response, usage = call_gpt(
                prompt="What is 2+2? Respond with just the number.",
                client=openai_client,
                model_name="gpt-4o-mini",  # Use cheaper model for testing
                max_tokens=10
            )
            print(f"✓ GPT Response: {response}")
            print(f"  Usage: {usage}")
        except Exception as e:
            print(f"✗ GPT Error: {e}")
    
    # Test DeepSeek (if API key available)
    if os.getenv("DEEPSEEK_API_KEY"):
        print("\nTesting DeepSeek...")
        deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        try:
            response, usage = call_deepseek(
                prompt="What is 2+2? Respond with just the number.",
                client=deepseek_client,
                model_name="deepseek-chat",
                max_tokens=10
            )
            print(f"✓ DeepSeek Response: {response}")
            print(f"  Usage: {usage}")
        except Exception as e:
            print(f"✗ DeepSeek Error: {e}")