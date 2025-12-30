#!/usr/bin/env python3
"""
OpenAI Agents Module for LocalDeployer
Provides two specialized agents for different tasks
"""
import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from openai import RateLimitError, APIConnectionError, APIError
from typing import Optional, Dict, Any, List, Tuple

# Path to the secret file
SECRET_FILE = Path(__file__).parent / ".openai_secret"
CONFIG_FILE = Path(__file__).parent / "config.json"

# Configure logging
logger = logging.getLogger(__name__)

# Load configuration
_config = None

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    global _config
    if _config is not None:
        return _config
    
    default_config = {
        "openai": {
            "debug": False,
            "log_requests": False,
            "log_responses": False,
            "log_errors": True
        },
        "backend": {
            "log_level": "INFO"
        }
    }
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                _config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in _config:
                        _config[key] = value
                    elif isinstance(value, dict):
                        _config[key] = {**value, **_config.get(key, {})}
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}. Using defaults.")
            _config = default_config
    else:
        _config = default_config
    
    return _config

def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    return load_config()

def is_debug_enabled() -> bool:
    """Check if debug logging is enabled"""
    config = load_config()
    return config.get("openai", {}).get("debug", False)

def should_log_requests() -> bool:
    """Check if request logging is enabled"""
    config = load_config()
    return config.get("openai", {}).get("log_requests", False) or is_debug_enabled()

def should_log_responses() -> bool:
    """Check if response logging is enabled"""
    config = load_config()
    return config.get("openai", {}).get("log_responses", False) or is_debug_enabled()

def should_log_errors() -> bool:
    """Check if error logging is enabled"""
    config = load_config()
    return config.get("openai", {}).get("log_errors", True)

def get_openai_model() -> str:
    """
    Get the OpenAI model to use from config, with fallback.
    
    Current recommended models (as of December 2025):
    - gpt-5.2: Latest flagship model, optimized for coding and agentic tasks (default)
    - gpt-5.1-codex-max: Specialized for agentic coding tasks
    - gpt-5-mini: Faster, cost-effective GPT-5 variant
    - gpt-4o-mini: Fast, cost-effective GPT-4 variant
    - gpt-4o: More capable GPT-4 model
    - gpt-3.5-turbo: Legacy model, still available but older
    
    Note: GPT-5 models may require specific API access tiers. If you get a model_not_found
    error, try a fallback model from the config.
    """
    config = load_config()
    model = config.get("openai", {}).get("model", "gpt-5.2")
    
    if is_debug_enabled():
        logger.debug(f"Using OpenAI model: {model}")
    
    return model

def validate_pipeline_format(parsed_response: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate that the parsed response has the correct pipeline format.
    
    Args:
        parsed_response: The parsed JSON response from OpenAI
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(parsed_response, dict):
        return False, "Response is not a JSON object"
    
    # Check for required top-level keys
    if "pipeline" not in parsed_response:
        return False, "Missing 'pipeline' key in response"
    
    pipeline = parsed_response.get("pipeline", {})
    if not isinstance(pipeline, dict):
        return False, "'pipeline' is not a JSON object"
    
    # Check for steps array
    if "steps" not in pipeline:
        return False, "Missing 'steps' array in pipeline"
    
    steps = pipeline.get("steps", [])
    if not isinstance(steps, list):
        return False, "'steps' is not an array"
    
    if len(steps) == 0:
        return False, "'steps' array is empty"
    
    # Validate each step has required fields
    required_step_fields = ["step_number", "script_name", "script_id"]
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return False, f"Step {i+1} is not a JSON object"
        
        for field in required_step_fields:
            if field not in step:
                return False, f"Step {i+1} is missing required field: {field}"
    
    return True, None

def save_openai_response(response_text: str, project_name: str, agent_name: str, 
                         response_data: Optional[Dict[str, Any]] = None,
                         request_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Save OpenAI API response to a text file, including request details.
    
    Args:
        response_text: The raw response text from OpenAI
        project_name: Name of the project
        agent_name: Name of the agent (e.g., "ScriptAnalysisAgent", "DeploymentAgent")
        response_data: Optional parsed response data to include
        request_data: Optional request data (messages, model, parameters) to include
        
    Returns:
        Path to saved file, or None if save failed
    """
    try:
        responses_dir = Path("responses")
        responses_dir.mkdir(exist_ok=True)
        
        # Create filename: project_agent_timestamp.txt
        safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_project_name = safe_project_name.replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_file = responses_dir / f"{safe_project_name}_{agent_name}_{timestamp}.txt"
        
        # Format the response
        file_content = f"""OpenAI API Request/Response Log
================================
Project: {project_name}
Agent: {agent_name}
Timestamp: {datetime.now().isoformat()}

"""
        
        # Add request information if provided
        if request_data:
            file_content += f"""API REQUEST
{'-' * 60}
"""
            if request_data.get("model"):
                file_content += f"Model: {request_data.get('model')}\n"
            if request_data.get("temperature") is not None:
                file_content += f"Temperature: {request_data.get('temperature')}\n"
            if request_data.get("max_completion_tokens"):
                file_content += f"Max Completion Tokens: {request_data.get('max_completion_tokens')}\n"
            if request_data.get("max_tokens"):
                file_content += f"Max Tokens: {request_data.get('max_tokens')}\n"
            if request_data.get("response_format"):
                file_content += f"Response Format: {json.dumps(request_data.get('response_format'), indent=2)}\n"
            
            file_content += f"\nMessages:\n"
            messages = request_data.get("messages", [])
            for i, msg in enumerate(messages, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate very long content for readability
                content_preview = content
                if len(content) > 5000:
                    content_preview = content[:5000] + f"\n... (truncated, {len(content)} total characters)"
                
                file_content += f"\n  Message {i} ({role}):\n"
                file_content += f"  {'-' * 58}\n"
                # Indent content
                for line in content_preview.split('\n'):
                    file_content += f"  {line}\n"
                file_content += f"  {'-' * 58}\n"
            
            file_content += f"\n"
        
        file_content += f"""API RESPONSE
{'-' * 60}
{response_text}
{'-' * 60}
"""
        
        # Add parsed data if provided
        if response_data:
            file_content += f"\nParsed Response Data:\n{'-' * 60}\n"
            file_content += json.dumps(response_data, indent=2, ensure_ascii=False)
            file_content += "\n"
        
        # Save to file
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        if is_debug_enabled():
            logger.debug(f"OpenAI response saved to: {response_file}")
        
        return str(response_file)
    except Exception as e:
        if should_log_errors():
            logger.error(f"Failed to save OpenAI response: {e}", exc_info=True)
        return None

def load_api_key() -> Optional[str]:
    """Load OpenAI API key from secret file"""
    if not SECRET_FILE.exists():
        raise FileNotFoundError(
            f"OpenAI API key file not found: {SECRET_FILE}\n"
            f"Please create this file and add your OpenAI API key."
        )
    
    with open(SECRET_FILE, 'r') as f:
        key = f.read().strip()
    
    if not key:
        raise ValueError("OpenAI API key file is empty")
    
    return key

def get_client() -> OpenAI:
    """Get OpenAI client instance"""
    api_key = load_api_key()
    
    if is_debug_enabled():
        logger.info("Initializing OpenAI client...")
        logger.debug(f"API key loaded: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    
    try:
        client = OpenAI(api_key=api_key)
        if is_debug_enabled():
            logger.info("OpenAI client initialized successfully")
        return client
    except Exception as e:
        if should_log_errors():
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        raise

class ScriptAnalysisAgent:
    """
    Agent specialized in analyzing Python scripts and their execution results
    """
    
    def __init__(self):
        self.client = get_client()
        self.system_prompt = """You are a Python script analysis agent specialized in:
- Analyzing Python script execution results
- Identifying errors and issues in script output
- Providing recommendations for fixing problems
- Understanding AWS deployment scripts and CloudFormation operations
- Interpreting console output and stack status messages

Provide clear, actionable insights based on the script execution data."""
    
    def analyze_execution(self, script_name: str, stdout: str, stderr: str, 
                         exit_code: int, error: Optional[str] = None,
                         script_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze script execution results and provide insights.
        Returns file requests in a structured format for easy parsing.
        
        Args:
            script_name: Name of the executed script
            stdout: Standard output from script
            stderr: Standard error output
            exit_code: Exit code from script execution
            error: Optional error message
            script_path: Optional path to the script file
            
        Returns:
            Dictionary with analysis results and file requests:
            {
                "success": bool,
                "analysis": str,
                "files_to_investigate": [
                    {"path": str, "reason": str},
                    ...
                ],
                "script_name": str,
                "exit_code": int
            }
        """
        try:
            # Build context for analysis
            context = f"""
Script: {script_name}
Exit Code: {exit_code}
"""
            
            if script_path:
                context += f"Script Path: {script_path}\n"
            
            if error:
                context += f"Error: {error}\n"
            
            if stdout:
                context += f"\nStandard Output:\n{stdout}\n"
            
            if stderr:
                context += f"\nStandard Error:\n{stderr}\n"
            
            if not stdout and not stderr and not error:
                context += "\nScript completed with no output.\n"
            
            user_prompt = f"""
Analyze the following script execution results:

{context}

Provide your response as a JSON object with the following structure:
{{
    "analysis": "Your detailed analysis text here",
    "files_to_investigate": [
        {{
            "path": "relative/path/to/file.py",
            "reason": "Why this file is relevant (e.g., 'Contains error handling logic', 'Referenced in error message')"
        }}
    ]
}}

The "files_to_investigate" array should contain any files you want to examine to better understand the issue.
Use relative paths from the workspace root (D:/Workspace) or relative to the script's directory.
Common files to investigate:
- The script itself (if not already provided)
- Configuration files (yaml, json, ini, etc.)
- Related Python modules imported by the script
- Log files mentioned in output
- Template files referenced in errors

If no files are needed, return an empty array.
"""
            
            # Get model from config
            model = get_openai_model()
            
            # Log request details
            if should_log_requests():
                logger.info(f"[OpenAI Request] analyze_execution for script: {script_name}")
                logger.debug(f"[OpenAI Request] Model: {model}")
                logger.debug(f"[OpenAI Request] Context length: {len(context)} chars")
                logger.debug(f"[OpenAI Request] Prompt length: {len(user_prompt)} chars")
                logger.debug(f"[OpenAI Request] System prompt length: {len(self.system_prompt)} chars")
            
            try:
                if is_debug_enabled():
                    logger.info(f"Sending request to OpenAI API using model: {model}...")
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.system_prompt + "\n\nAlways respond with valid JSON in the requested format."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_completion_tokens=2000,
                    response_format={"type": "json_object"}
                )
                
                if should_log_responses():
                    logger.info(f"[OpenAI Response] Received response successfully")
                    logger.debug(f"[OpenAI Response] Model used: {response.model}")
                    logger.debug(f"[OpenAI Response] Finish reason: {response.choices[0].finish_reason}")
                    logger.debug(f"[OpenAI Response] Tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
                
                response_text = response.choices[0].message.content
                
                # Save response to file
                script_project = script_name.replace(" Project Analysis", "") if " Project Analysis" in script_name else "Unknown"
                save_openai_response(
                    response_text, 
                    script_project, 
                    "ScriptAnalysisAgent",
                    {"method": "analyze_execution", "script_name": script_name},
                    {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt + "\n\nAlways respond with valid JSON in the requested format."},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.3,
                        "max_completion_tokens": 2000,
                        "response_format": {"type": "json_object"}
                    }
                )
                
                # Parse JSON response
                try:
                    parsed_response = json.loads(response_text)
                    analysis_text = parsed_response.get("analysis", "")
                    files_to_investigate = parsed_response.get("files_to_investigate", [])
                    
                    if is_debug_enabled():
                        logger.debug(f"[OpenAI Response] Parsed JSON successfully")
                        logger.debug(f"[OpenAI Response] Files to investigate: {len(files_to_investigate)}")
                        
                except json.JSONDecodeError as json_error:
                    if should_log_errors():
                        logger.warning(f"[OpenAI Warning] Failed to parse JSON response: {json_error}")
                        logger.debug(f"[OpenAI Warning] Response text (first 500 chars): {response_text[:500]}")
                    # Fallback if JSON parsing fails
                    analysis_text = response_text
                    files_to_investigate = []
                
                if is_debug_enabled():
                    logger.info(f"[OpenAI Success] Analysis completed for {script_name}")
                
                return {
                    "success": True,
                    "analysis": analysis_text,
                    "files_to_investigate": files_to_investigate,
                    "script_name": script_name,
                    "exit_code": exit_code
                }
                
            except Exception as api_error:
                if should_log_errors():
                    logger.error(f"[OpenAI Error] Failed to get response from OpenAI API")
                    logger.error(f"[OpenAI Error] Error type: {type(api_error).__name__}")
                    logger.error(f"[OpenAI Error] Error message: {str(api_error)}")
                    logger.error(f"[OpenAI Error] Full traceback:", exc_info=True)
                    
                    # Check for specific error types
                    if "Connection" in str(api_error) or "connect" in str(api_error).lower():
                        logger.error("[OpenAI Error] Connection issue detected. Possible causes:")
                        logger.error("  - Network connectivity problems")
                        logger.error("  - OpenAI API service is down")
                        logger.error("  - Firewall/proxy blocking the connection")
                        logger.error("  - DNS resolution issues")
                    elif "API key" in str(api_error) or "authentication" in str(api_error).lower():
                        logger.error("[OpenAI Error] Authentication issue detected. Possible causes:")
                        logger.error("  - Invalid API key")
                        logger.error("  - API key expired or revoked")
                        logger.error("  - API key format incorrect")
                    elif "rate limit" in str(api_error).lower() or "429" in str(api_error):
                        logger.error("[OpenAI Error] Rate limit issue detected.")
                
                return {
                    "success": False,
                    "error": str(api_error),
                    "error_type": type(api_error).__name__,
                    "analysis": None,
                    "files_to_investigate": []
                }
            
        except Exception as e:
            if should_log_errors():
                logger.error(f"[OpenAI Error] analyze_execution failed for {script_name}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "analysis": None,
                "files_to_investigate": []
            }
    
    def analyze_with_files(self, script_name: str, stdout: str, stderr: str,
                          exit_code: int, files_content: Dict[str, str],
                          error: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze script execution results with additional file contents provided.
        
        Args:
            script_name: Name of the executed script
            stdout: Standard output from script
            stderr: Standard error output
            exit_code: Exit code from script execution
            files_content: Dictionary mapping file paths to their contents
                         e.g., {"path/to/file.py": "file contents here"}
            error: Optional error message
            
        Returns:
            Dictionary with enhanced analysis results
        """
        try:
            # Build context for analysis
            context = f"""
Script: {script_name}
Exit Code: {exit_code}
"""
            
            if error:
                context += f"Error: {error}\n"
            
            if stdout:
                context += f"\nStandard Output:\n{stdout}\n"
            
            if stderr:
                context += f"\nStandard Error:\n{stderr}\n"
            
            # Add file contents
            if files_content:
                context += "\n\nAdditional Files Provided for Analysis:\n"
                context += "=" * 60 + "\n"
                for file_path, content in files_content.items():
                    context += f"\nFile: {file_path}\n"
                    context += "-" * 60 + "\n"
                    # Limit file content to avoid token limits
                    content_preview = content[:5000] if len(content) > 5000 else content
                    context += content_preview
                    if len(content) > 5000:
                        context += f"\n... (truncated, {len(content)} total characters)\n"
                    context += "\n" + "=" * 60 + "\n"
            
            user_prompt = f"""
Analyze the following project files and provide a comprehensive analysis document.

{context}

You are analyzing a project with scripts and YAML files for AWS deployment. Based on the files provided (including any README files), create a detailed analysis document that can be used as context for the DeploymentAgent (debug agent) to help troubleshoot and understand deployment issues.

Provide your analysis as a clear, well-structured document covering:
- Project structure and purpose
- Scripts and their functions
- YAML/CloudFormation templates and their purpose
- Deployment workflow and dependencies
- Common issues and troubleshooting approaches
- Key information that would help debug deployment failures

Write in a clear, structured format that can be easily read and used as context for debugging.
"""
            
            # Get model from config
            model = get_openai_model()
            
            # Log request details
            if should_log_requests():
                logger.info(f"[OpenAI Request] analyze_with_files for script: {script_name}")
                logger.debug(f"[OpenAI Request] Model: {model}")
                logger.debug(f"[OpenAI Request] Files provided: {len(files_content)}")
                logger.debug(f"[OpenAI Request] Total context length: {len(context)} chars")
                for file_path in files_content.keys():
                    file_size = len(files_content[file_path])
                    logger.debug(f"[OpenAI Request]   - {file_path}: {file_size} chars")
            
            # Retry logic for format validation
            max_retries = 3
            retry_count = 0
            last_error = None
            last_response_text = None
            
            while retry_count <= max_retries:
                try:
                    if is_debug_enabled():
                        if retry_count > 0:
                            logger.info(f"Retrying OpenAI API request (attempt {retry_count + 1}/{max_retries + 1})...")
                        else:
                            logger.info(f"Sending request to OpenAI API with file contents using model: {model}...")
                    
                    # Build messages
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.3,
                        max_completion_tokens=4000
                    )
                    
                    if should_log_responses():
                        logger.info(f"[OpenAI Response] Received response successfully (attempt {retry_count + 1})")
                        logger.debug(f"[OpenAI Response] Model used: {response.model}")
                        logger.debug(f"[OpenAI Response] Tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
                    
                    response_text = response.choices[0].message.content
                    
                    # Save response to file
                    script_project = script_name.replace(" Project Analysis", "") if " Project Analysis" in script_name else "Unknown"
                    save_openai_response(
                        response_text, 
                        script_project, 
                        "ScriptAnalysisAgent",
                        {"method": "analyze_with_files", "script_name": script_name},
                        {
                            "model": model,
                            "messages": messages,
                            "temperature": 0.3,
                            "max_completion_tokens": 4000
                        }
                    )
                    
                    # Return plain text document
                    return {
                        "success": True,
                        "analysis": response_text,
                        "document": response_text,
                        "script_name": script_name,
                        "exit_code": exit_code,
                        "files_analyzed": list(files_content.keys())
                    }
                
                except RateLimitError as rate_limit_error:
                    # Handle rate limit errors with retry and exponential backoff
                    retry_count += 1
                    wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60 seconds
                    
                    logger.warning(f"[OpenAI Rate Limit] Rate limit error detected: {str(rate_limit_error)}")
                    logger.warning(f"[OpenAI Rate Limit] Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                    
                    if retry_count <= max_retries:
                        logger.info(f"[OpenAI Retry] Will retry after {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                        logger.info(f"[OpenAI Retry] Retrying request now (attempt {retry_count}/{max_retries})...")
                        continue  # Retry the request
                    else:
                        # Max retries reached for rate limit
                        logger.error(f"[OpenAI Error] Max retries ({max_retries}) reached for rate limit error")
                        return {
                            "success": False,
                            "error": f"Rate limit error after {max_retries} retries: {str(rate_limit_error)}",
                            "error_type": "RateLimitError",
                            "analysis": None
                        }
                
                except APIConnectionError as conn_error:
                    # Handle connection errors with retry
                    retry_count += 1
                    wait_time = min(2 ** retry_count, 30)  # Exponential backoff, max 30 seconds
                    
                    logger.warning(f"[OpenAI Connection Error] Connection failed: {str(conn_error)}")
                    logger.warning(f"[OpenAI Connection Error] Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                    
                    if retry_count <= max_retries:
                        logger.info(f"[OpenAI Retry] Will retry after {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                        logger.info(f"[OpenAI Retry] Retrying request now (attempt {retry_count}/{max_retries})...")
                        continue  # Retry the request
                    else:
                        # Max retries reached for connection error
                        logger.error(f"[OpenAI Error] Max retries ({max_retries}) reached for connection error")
                        return {
                            "success": False,
                            "error": f"Connection error after {max_retries} retries: {str(conn_error)}",
                            "error_type": "APIConnectionError",
                            "analysis": None
                        }
                
                except APIError as api_error:
                    # Check if it's a rate limit error (sometimes wrapped in APIError)
                    error_str = str(api_error).lower()
                    if "rate limit" in error_str or "429" in error_str:
                        retry_count += 1
                        wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60 seconds
                        
                        logger.warning(f"[OpenAI Rate Limit] Rate limit detected in APIError: {str(api_error)}")
                        logger.warning(f"[OpenAI Rate Limit] Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                        
                        if retry_count <= max_retries:
                            logger.info(f"[OpenAI Retry] Will retry after {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                            time.sleep(wait_time)
                            logger.info(f"[OpenAI Retry] Retrying request now (attempt {retry_count}/{max_retries})...")
                            continue  # Retry the request
                        else:
                            logger.error(f"[OpenAI Error] Max retries ({max_retries}) reached for rate limit")
                            return {
                                "success": False,
                                "error": f"Rate limit error after {max_retries} retries: {str(api_error)}",
                                "error_type": "APIError",
                                "analysis": None
                            }
                    
                    # Other API errors (auth, etc.) - don't retry
                    if should_log_errors():
                        logger.error(f"[OpenAI Error] Failed to get response from OpenAI API")
                        logger.error(f"[OpenAI Error] Error type: {type(api_error).__name__}")
                        logger.error(f"[OpenAI Error] Error message: {str(api_error)}")
                        logger.error(f"[OpenAI Error] Full traceback:", exc_info=True)
                        
                        # Provide more detailed error information
                        if "API key" in error_str or "authentication" in error_str:
                            logger.error("[OpenAI Error] Authentication issue detected. Check API key in .openai_secret file.")
                    
                    return {
                        "success": False,
                        "error": str(api_error),
                        "error_type": type(api_error).__name__,
                        "analysis": None
                    }
                
                except Exception as api_error:
                    # Check for rate limit in generic exceptions
                    error_str = str(api_error).lower()
                    if "rate limit" in error_str or "429" in error_str:
                        retry_count += 1
                        wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60 seconds
                        
                        logger.warning(f"[OpenAI Rate Limit] Rate limit detected in exception: {str(api_error)}")
                        logger.warning(f"[OpenAI Rate Limit] Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                        
                        if retry_count <= max_retries:
                            logger.info(f"[OpenAI Retry] Will retry after {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                            time.sleep(wait_time)
                            logger.info(f"[OpenAI Retry] Retrying request now (attempt {retry_count}/{max_retries})...")
                            continue  # Retry the request
                        else:
                            logger.error(f"[OpenAI Error] Max retries ({max_retries}) reached for rate limit")
                            return {
                                "success": False,
                                "error": f"Rate limit error after {max_retries} retries: {str(api_error)}",
                                "error_type": type(api_error).__name__,
                                "analysis": None
                            }
                    
                    # Other errors - don't retry
                    if should_log_errors():
                        logger.error(f"[OpenAI Error] Failed to get response from OpenAI API")
                        logger.error(f"[OpenAI Error] Error type: {type(api_error).__name__}")
                        logger.error(f"[OpenAI Error] Error message: {str(api_error)}")
                        logger.error(f"[OpenAI Error] Full traceback:", exc_info=True)
                        
                        # Provide more detailed error information
                        if "Connection" in str(api_error) or "connect" in error_str:
                            logger.error("[OpenAI Error] Connection issue detected. Check network connectivity and OpenAI service status.")
                        elif "API key" in error_str or "authentication" in error_str:
                            logger.error("[OpenAI Error] Authentication issue detected. Check API key in .openai_secret file.")
                    
                    return {
                        "success": False,
                        "error": str(api_error),
                        "error_type": type(api_error).__name__,
                        "analysis": None
                    }
            
            # Should not reach here, but handle just in case
            if should_log_errors():
                logger.error(f"[OpenAI Error] Unexpected end of retry loop")
            return {
                "success": False,
                "error": "Failed to get valid response after retries",
                "analysis": None
            }
            
            
        except Exception as e:
            if should_log_errors():
                logger.error(f"[OpenAI Error] analyze_with_files failed for {script_name}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "analysis": None
            }
    
    def suggest_fix(self, script_name: str, error_message: str, 
                    context: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest fixes for script errors
        
        Args:
            script_name: Name of the script with the error
            error_message: The error message
            context: Additional context (e.g., script output)
            
        Returns:
            Dictionary with suggested fixes
        """
        try:
            user_prompt = f"""
Script: {script_name}
Error: {error_message}
"""
            
            if context:
                user_prompt += f"\nAdditional Context:\n{context}\n"
            
            user_prompt += "\nPlease provide specific, actionable steps to fix this error."
            
            model = get_openai_model()
            if should_log_requests():
                logger.info(f"[OpenAI Request] suggest_fix for script: {script_name}")
                logger.debug(f"[OpenAI Request] Model: {model}")
            
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_completion_tokens=800
                )
                
                if should_log_responses():
                    logger.info(f"[OpenAI Response] Received suggestion successfully")
                
                suggestion = response.choices[0].message.content
                
                # Save response to file
                save_openai_response(
                    suggestion, 
                    script_name, 
                    "ScriptAnalysisAgent",
                    {"method": "suggest_fix", "script_name": script_name},
                    {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.3,
                        "max_completion_tokens": 800
                    }
                )
                
                return {
                    "success": True,
                    "suggestion": suggestion,
                    "script_name": script_name
                }
            except Exception as e:
                if should_log_errors():
                    logger.error(f"[OpenAI Error] suggest_fix failed: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "suggestion": None
                }
            
        except Exception as e:
            if should_log_errors():
                logger.error(f"[OpenAI Error] suggest_fix outer exception: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "suggestion": None
            }

class DeploymentAgent:
    """
    Agent specialized in AWS deployment operations and infrastructure management
    """
    
    def __init__(self):
        self.client = get_client()
        self.system_prompt = """You are an AWS deployment and infrastructure management agent specialized in:
- AWS CloudFormation stack deployments
- Infrastructure as Code (IaC) best practices
- Deployment strategies and rollback procedures
- AWS service configuration and troubleshooting
- Stack status interpretation and recommendations
- Multi-stack deployment orchestration

Provide expert guidance on AWS deployments and infrastructure management."""
    
    def analyze_deployment_status(self, stack_name: str = "", status: str = "", 
                                  script_output: str = "", context_document: str = "") -> Dict[str, Any]:
        """
        Analyze script execution output and provide debugging guidance
        
        Args:
            stack_name: Optional stack name if available
            status: Optional stack status if available
            script_output: The script's stdout/stderr output
            context_document: Optional analysis document from ScriptAnalysisAgent
            
        Returns:
            Dictionary with deployment analysis
        """
        try:
            context = ""
            
            if context_document:
                context += f"Project Analysis Document:\n{context_document}\n\n"
            
            if stack_name:
                context += f"Stack Name: {stack_name}\n"
            if status:
                context += f"Status: {status}\n"
            
            context += f"\nScript Output:\n{script_output}\n"
            
            user_prompt = f"""
Analyze the following script execution output and provide debugging guidance:

{context}

Provide a clear analysis of what happened, what might have gone wrong, and recommendations for fixing any issues. Write in a clear, structured format.
"""
            
            model = get_openai_model()
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_completion_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            
            # Save response to file
            save_openai_response(
                response_text, 
                stack_name or "script_output", 
                "DeploymentAgent",
                {"method": "analyze_deployment_status", "stack_name": stack_name, "status": status},
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,
                    "max_completion_tokens": 2000
                }
            )
            
            return {
                "success": True,
                "analysis": response_text,
                "stack_name": stack_name,
                "status": status
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }
    
    def recommend_deployment_strategy(self, project_name: str, 
                                     environment: str,
                                     stacks: list) -> Dict[str, Any]:
        """
        Recommend deployment strategy for a multi-stack project
        
        Args:
            project_name: Name of the project
            environment: Deployment environment
            stacks: List of stack definitions
            
        Returns:
            Dictionary with deployment recommendations
        """
        try:
            context = f"""
Project: {project_name}
Environment: {environment}
Stacks: {', '.join([s.get('name', 'N/A') for s in stacks])}
"""
            
            user_prompt = f"""
For the following deployment scenario:

{context}

Please provide:
1. Recommended deployment order
2. Dependencies between stacks
3. Best practices for this deployment
4. Rollback strategy recommendations
5. Monitoring and validation steps
"""
            
            model = get_openai_model()
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_completion_tokens=1200
            )
            
            recommendation = response.choices[0].message.content
            
            # Save response to file
            save_openai_response(
                recommendation, 
                project_name, 
                "DeploymentAgent",
                {"method": "recommend_deployment_strategy", "project_name": project_name, "environment": environment},
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.4,
                    "max_completion_tokens": 1200
                }
            )
            
            return {
                "success": True,
                "recommendation": recommendation,
                "project_name": project_name,
                "environment": environment
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recommendation": None
            }

def read_files_for_analysis(file_requests: List[Dict[str, str]], 
                           base_path: Optional[Path] = None,
                           workspace_root: Optional[Path] = None) -> Dict[str, str]:
    """
    Read files requested by the agent for analysis.
    
    Args:
        file_requests: List of file request dicts with "path" and "reason" keys
                     e.g., [{"path": "path/to/file.py", "reason": "..."}]
        base_path: Base path to resolve relative paths (e.g., script directory)
        workspace_root: Workspace root path (default: D:/Workspace)
        
    Returns:
        Dictionary mapping file paths to their contents
        e.g., {"path/to/file.py": "file contents"}
    
    Example:
        # After getting file requests from analyze_execution()
        result = agent.analyze_execution(...)
        file_requests = result.get("files_to_investigate", [])
        
        # Read the requested files
        files_content = read_files_for_analysis(
            file_requests,
            base_path=Path("D:/Workspace/DiamondDrip/aws")
        )
        
        # Analyze with file contents
        enhanced_result = agent.analyze_with_files(
            script_name="...",
            stdout="...",
            stderr="...",
            exit_code=0,
            files_content=files_content
        )
    """
    if workspace_root is None:
        workspace_root = Path("D:/Workspace")
    
    files_content = {}
    
    for file_request in file_requests:
        file_path_str = file_request.get("path", "")
        if not file_path_str:
            continue
        
        # Try multiple path resolution strategies
        resolved_path = None
        
        # Strategy 1: Absolute path
        if Path(file_path_str).is_absolute():
            resolved_path = Path(file_path_str)
        
        # Strategy 2: Relative to workspace root
        elif workspace_root.exists():
            resolved_path = workspace_root / file_path_str
            if not resolved_path.exists():
                # Try with aws folder
                resolved_path = workspace_root / file_path_str.replace("aws/", "").replace("aws\\", "")
        
        # Strategy 3: Relative to base_path (script directory)
        if (not resolved_path or not resolved_path.exists()) and base_path:
            resolved_path = base_path / file_path_str
            if not resolved_path.exists():
                # Try parent directories
                for parent in base_path.parents:
                    candidate = parent / file_path_str
                    if candidate.exists():
                        resolved_path = candidate
                        break
        
        # Strategy 4: Try as-is relative to current directory
        if not resolved_path or not resolved_path.exists():
            resolved_path = Path(file_path_str)
        
        # Read file if it exists
        if resolved_path and resolved_path.exists() and resolved_path.is_file():
            try:
                with open(resolved_path, 'r', encoding='utf-8', errors='replace') as f:
                    files_content[str(resolved_path)] = f.read()
            except Exception as e:
                # Store error message instead of content
                files_content[str(resolved_path)] = f"[Error reading file: {str(e)}]"
        else:
            # File not found - store placeholder
            files_content[file_path_str] = f"[File not found: {file_path_str}]"
    
    return files_content

# Convenience functions to get agent instances
def get_script_analysis_agent() -> ScriptAnalysisAgent:
    """Get an instance of the ScriptAnalysisAgent"""
    return ScriptAnalysisAgent()

def get_deployment_agent() -> DeploymentAgent:
    """Get an instance of the DeploymentAgent"""
    return DeploymentAgent()

