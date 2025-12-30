#!/usr/bin/env python3
"""
Backend server for LocalDeployer web app
Discovers and runs Python scripts from AWS folders in D:\Workspace
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from datetime import datetime
import traceback
import logging
import re
import threading
import queue
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import OpenAI agents
OPENAI_AVAILABLE = False
OPENAI_ERROR = None
try:
    from openai_agents import get_script_analysis_agent, read_files_for_analysis, get_deployment_agent
    OPENAI_AVAILABLE = True
except ImportError as e:
    OPENAI_ERROR = f"Import error: {str(e)}"
    print(f"Warning: OpenAI agents module not available. AI analysis features will be disabled.")
    print(f"Error details: {OPENAI_ERROR}")
    print("Make sure 'openai' package is installed: pip install openai")
except Exception as e:
    OPENAI_ERROR = f"Error loading OpenAI module: {str(e)}"
    print(f"Warning: OpenAI agents module not available. AI analysis features will be disabled.")
    print(f"Error details: {OPENAI_ERROR}")

# Import Docker deployment manager
DOCKER_AVAILABLE = False
docker_manager = None
try:
    from docker_deployment import get_docker_manager
    DOCKER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Docker deployment module not available: {e}")
except Exception as e:
    print(f"Warning: Error loading Docker deployment module: {e}")

# Initialize docker_manager after WORKSPACE_ROOT is defined
if DOCKER_AVAILABLE:
    try:
        docker_manager = get_docker_manager(WORKSPACE_ROOT)
    except Exception as e:
        print(f"Warning: Failed to initialize Docker manager: {e}")
        DOCKER_AVAILABLE = False

app = Flask(__name__, static_folder='frontend')
CORS(app)  # Enable CORS for frontend

WORKSPACE_ROOT = Path("D:/Workspace")

# Store running script processes for interactive input
running_scripts = {}
script_lock = threading.Lock()

# Store running script processes for interactive input
running_scripts = {}
script_lock = threading.Lock()

def load_script_arguments(aws_folder):
    """Load script-arguments.json (or script_arguments.json for backward compatibility) from the aws folder if it exists"""
    # Try hyphen version first (preferred)
    script_args_file = aws_folder / "script-arguments.json"
    if not script_args_file.exists():
        # Fall back to underscore version for backward compatibility
        script_args_file = aws_folder / "script_arguments.json"
    
    if script_args_file.exists():
        try:
            with open(script_args_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded {script_args_file.name} from {aws_folder}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {script_args_file.name} at {aws_folder}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load {script_args_file.name} from {aws_folder}: {e}")
    else:
        logger.debug(f"No script-arguments.json or script_arguments.json found in {aws_folder}")
    return None

def get_script_info(script_name, script_args_data):
    """Get script information from script-arguments.json (or script_arguments.json)
    Supports two formats:
    1. Array format: {"scripts": [{"name": "script.py", ...}]}
    2. Object format: {"scripts": {"script.py": {...}}}
    """
    if not script_args_data:
        return None
    
    if not isinstance(script_args_data, dict):
        logger.warning(f"script-arguments.json data is not a dictionary: {type(script_args_data)}")
        return None
    
    if "scripts" not in script_args_data:
        return None
    
    scripts_data = script_args_data.get("scripts")
    
    # Handle object format (DiamondDrip style): {"scripts": {"script.py": {...}}}
    if isinstance(scripts_data, dict):
        if script_name in scripts_data:
            script_info = scripts_data[script_name]
            if isinstance(script_info, dict):
                # Convert arguments from object to array format for frontend compatibility
                if "arguments" in script_info and isinstance(script_info["arguments"], dict):
                    args_dict = script_info["arguments"]
                    args_array = []
                    for arg_name, arg_data in args_dict.items():
                        if isinstance(arg_data, dict):
                            arg_entry = {
                                "name": arg_name,
                                "type": arg_data.get("type", "string"),
                                "required": arg_data.get("required", False),
                                "description": arg_data.get("description", ""),
                                "default": arg_data.get("default")
                            }
                            # Add example if available
                            if "example" in arg_data:
                                arg_entry["example"] = arg_data["example"]
                            elif "examples" in arg_data and isinstance(arg_data["examples"], list) and len(arg_data["examples"]) > 0:
                                arg_entry["example"] = arg_data["examples"][0]
                            args_array.append(arg_entry)
                    script_info = script_info.copy()  # Don't modify original
                    script_info["arguments"] = args_array
                return script_info
        return None
    
    # Handle array format: {"scripts": [{"name": "script.py", ...}]}
    if isinstance(scripts_data, list):
        for script_info in scripts_data:
            if not isinstance(script_info, dict):
                logger.warning(f"Invalid script entry in script-arguments.json (not a dict): {type(script_info)}")
                continue
            if script_info.get("name") == script_name:
                return script_info
        return None
    
    logger.warning(f"scripts field in script-arguments.json is neither a list nor a dict: {type(scripts_data)}")
    return None

def discover_aws_scripts():
    """Discover all Python scripts in aws folders within the workspace"""
    scripts = []
    
    if not WORKSPACE_ROOT.exists():
        return scripts
    
    # Find all aws folders
    for aws_folder in WORKSPACE_ROOT.rglob("aws"):
        if aws_folder.is_dir():
            project_name = aws_folder.parent.name
            
            # Load script-arguments.json for this project
            script_args_data = load_script_arguments(aws_folder)
            
            # Find all Python scripts in this aws folder (including subdirectories)
            for script_file in aws_folder.rglob("*.py"):
                # Skip __init__ and other special files
                if script_file.name.startswith("__"):
                    continue
                
                # Calculate relative path from aws folder
                rel_path = script_file.relative_to(aws_folder)
                rel_path_str = str(rel_path).replace("\\", "/")
                
                script_info = {
                    "id": f"{project_name}_{script_file.stem}",
                    "project": project_name,
                    "name": script_file.name,
                    "path": str(script_file),
                    "relative_path": f"{project_name}/aws/{rel_path_str}",
                    "type": "script"
                }
                
                # Add script arguments info if available
                if script_args_data:
                    script_args_info = get_script_info(script_file.name, script_args_data)
                    if script_args_info:
                        script_info["arguments_info"] = script_args_info
                        logger.debug(f"Added arguments_info for {script_file.name}")
                    else:
                        logger.debug(f"No matching entry in script-arguments.json for {script_file.name}")
                
                scripts.append(script_info)
    
    return sorted(scripts, key=lambda x: (x["project"], x["name"]))

def discover_aws_yaml_files():
    """Discover all YAML files in aws folders within the workspace"""
    yaml_files = []
    
    if not WORKSPACE_ROOT.exists():
        return yaml_files
    
    # Find all aws folders
    for aws_folder in WORKSPACE_ROOT.rglob("aws"):
        if aws_folder.is_dir():
            project_name = aws_folder.parent.name
            
            # Find all YAML files in this aws folder (including subdirectories)
            for yaml_file in aws_folder.rglob("*.yaml"):
                # Calculate relative path from aws folder
                rel_path = yaml_file.relative_to(aws_folder)
                rel_path_str = str(rel_path).replace("\\", "/")
                
                yaml_files.append({
                    "id": f"{project_name}_yaml_{yaml_file.stem}",
                    "project": project_name,
                    "name": yaml_file.name,
                    "path": str(yaml_file),
                    "relative_path": f"{project_name}/aws/{rel_path_str}",
                    "type": "yaml"
                })
            
            # Also check for .yml files
            for yaml_file in aws_folder.rglob("*.yml"):
                rel_path = yaml_file.relative_to(aws_folder)
                rel_path_str = str(rel_path).replace("\\", "/")
                
                yaml_files.append({
                    "id": f"{project_name}_yaml_{yaml_file.stem}",
                    "project": project_name,
                    "name": yaml_file.name,
                    "path": str(yaml_file),
                    "relative_path": f"{project_name}/aws/{rel_path_str}",
                    "type": "yaml"
                })
    
    return sorted(yaml_files, key=lambda x: (x["project"], x["name"]))

def discover_build_instruction_files():
    """Discover build instruction files (Lambda_build_instructions, Dockerfile, etc.) in aws folders"""
    instruction_files = []
    
    if not WORKSPACE_ROOT.exists():
        return instruction_files
    
    # Patterns to look for
    patterns = [
        "*Lambda_build_instructions*",
        "*build_instructions*",
        "*docker*instructions*",
        "*deployment*instructions*",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml"
    ]
    
    # Find all aws folders
    for aws_folder in WORKSPACE_ROOT.rglob("aws"):
        if aws_folder.is_dir():
            project_name = aws_folder.parent.name
            
            # Search for instruction files
            for pattern in patterns:
                for instruction_file in aws_folder.rglob(pattern):
                    if instruction_file.is_file():
                        # Calculate relative path from aws folder
                        rel_path = instruction_file.relative_to(aws_folder)
                        rel_path_str = str(rel_path).replace("\\", "/")
                        
                        # Detect file type
                        file_type = "instructions"
                        if "dockerfile" in instruction_file.name.lower():
                            file_type = "dockerfile"
                        elif "docker-compose" in instruction_file.name.lower():
                            file_type = "docker_compose"
                        elif "lambda_build" in instruction_file.name.lower() or "build_instructions" in instruction_file.name.lower():
                            file_type = "build_instructions"
                        
                        instruction_files.append({
                            "id": f"{project_name}_instruction_{instruction_file.stem}",
                            "project": project_name,
                            "name": instruction_file.name,
                            "path": str(instruction_file),
                            "relative_path": f"{project_name}/aws/{rel_path_str}",
                            "type": file_type
                        })
    
    return sorted(instruction_files, key=lambda x: (x["project"], x["name"]))

def run_script(script_path, args=None, cwd=None):
    """Run a Python script and capture its output"""
    if not Path(script_path).exists():
        return {
            "success": False,
            "error": f"Script not found: {script_path}",
            "stdout": "",
            "stderr": "",
            "exit_code": -1
        }
    
    # Use the script's directory as working directory if not specified
    if cwd is None:
        cwd = Path(script_path).parent
    
    # Prepare command
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    command_str = " ".join(cmd)
    start_time = time.time()
    
    try:
        # Set environment to use UTF-8 encoding for Python scripts
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        # Also set console encoding on Windows
        if sys.platform == 'win32':
            env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
        
        # Run the script and capture output
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            encoding='utf-8',
            errors='replace',  # Handle encoding errors gracefully
            env=env
        )
        
        duration = time.time() - start_time
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "error": None if result.returncode == 0 else f"Script exited with code {result.returncode}",
            "command": command_str,
            "duration_seconds": round(duration, 2),
            "working_directory": str(cwd),
            "script_path": str(script_path)
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            "success": False,
            "error": "Script execution timed out after 5 minutes",
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "command": command_str,
            "duration_seconds": round(duration, 2),
            "working_directory": str(cwd),
            "script_path": str(script_path)
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "error": f"Error running script: {str(e)}",
            "stdout": "",
            "stderr": traceback.format_exc(),
            "exit_code": -1,
            "command": command_str,
            "duration_seconds": round(duration, 2),
            "working_directory": str(cwd),
            "script_path": str(script_path)
        }

import time
import re

def manage_pipeline_versions(project_name: str, pipelines_dir: Path) -> dict:
    """
    Manage pipeline versioning by deprecating old pipelines.
    
    When a new pipeline is saved:
    - The current active pipeline becomes deprecated_1
    - All existing deprecated pipelines have their numbers incremented
    
    Args:
        project_name: Sanitized project name (safe for filesystem)
        pipelines_dir: Path to pipelines directory
    
    Returns:
        Dictionary with information about versioning actions taken
    """
    result = {
        "versions_managed": False,
        "active_pipeline_deprecated": False,
        "deprecated_pipelines_renamed": [],
        "errors": []
    }
    
    try:
        active_pipeline_file = pipelines_dir / f"{project_name}.json"
        
        # Step 1: If active pipeline exists, rename it to deprecated_1
        if active_pipeline_file.exists():
            deprecated_1_file = pipelines_dir / f"{project_name}_deprecated_1.json"
            
            # If deprecated_1 already exists, we need to handle it
            if deprecated_1_file.exists():
                # We'll increment all deprecated versions first, then rename active
                # Find all deprecated files and sort by number (descending)
                deprecated_files = []
                pattern = re.compile(rf"^{re.escape(project_name)}_deprecated_(\d+)\.json$")
                
                for file in pipelines_dir.glob(f"{project_name}_deprecated_*.json"):
                    match = pattern.match(file.name)
                    if match:
                        num = int(match.group(1))
                        deprecated_files.append((num, file))
                
                # Sort by number descending (highest first)
                deprecated_files.sort(key=lambda x: x[0], reverse=True)
                
                # Increment each deprecated file's number
                for num, file in deprecated_files:
                    new_num = num + 1
                    new_file = pipelines_dir / f"{project_name}_deprecated_{new_num}.json"
                    try:
                        file.rename(new_file)
                        result["deprecated_pipelines_renamed"].append({
                            "old": file.name,
                            "new": new_file.name,
                            "old_version": num,
                            "new_version": new_num
                        })
                        logger.info(f"Renamed deprecated pipeline: {file.name} -> {new_file.name}")
                    except Exception as e:
                        error_msg = f"Failed to rename {file.name}: {str(e)}"
                        result["errors"].append(error_msg)
                        logger.error(error_msg, exc_info=True)
            
            # Now rename active pipeline to deprecated_1
            try:
                active_pipeline_file.rename(deprecated_1_file)
                result["active_pipeline_deprecated"] = True
                result["versions_managed"] = True
                logger.info(f"Deprecated active pipeline: {active_pipeline_file.name} -> {deprecated_1_file.name}")
            except Exception as e:
                error_msg = f"Failed to deprecate active pipeline: {str(e)}"
                result["errors"].append(error_msg)
                logger.error(error_msg, exc_info=True)
        else:
            # No active pipeline exists, but check for deprecated ones to increment
            # Find all deprecated files and sort by number (descending)
            deprecated_files = []
            pattern = re.compile(rf"^{re.escape(project_name)}_deprecated_(\d+)\.json$")
            
            for file in pipelines_dir.glob(f"{project_name}_deprecated_*.json"):
                match = pattern.match(file.name)
                if match:
                    num = int(match.group(1))
                    deprecated_files.append((num, file))
            
            # Sort by number descending (highest first) and increment
            deprecated_files.sort(key=lambda x: x[0], reverse=True)
            
            for num, file in deprecated_files:
                new_num = num + 1
                new_file = pipelines_dir / f"{project_name}_deprecated_{new_num}.json"
                try:
                    file.rename(new_file)
                    result["deprecated_pipelines_renamed"].append({
                        "old": file.name,
                        "new": new_file.name,
                        "old_version": num,
                        "new_version": new_num
                    })
                    result["versions_managed"] = True
                    logger.info(f"Renamed deprecated pipeline: {file.name} -> {new_file.name}")
                except Exception as e:
                    error_msg = f"Failed to rename {file.name}: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg, exc_info=True)
        
    except Exception as e:
        error_msg = f"Error managing pipeline versions: {str(e)}"
        result["errors"].append(error_msg)
        logger.error(error_msg, exc_info=True)
    
    return result

def extract_value_from_output(var_name: str, execution_context: dict) -> Optional[str]:
    """
    Extract values from stdout/stderr based on variable name patterns.
    
    Args:
        var_name: Variable name like "failed_stack_name_from_stdout" or "stack_status_from_output"
        execution_context: Context dictionary with stdout, stderr, etc.
    
    Returns:
        Extracted value or None if not found
    """
    stdout = execution_context.get("stdout", "")
    stderr = execution_context.get("stderr", "")
    combined_output = stdout + "\n" + stderr
    
    var_lower = var_name.lower()
    
    # Pattern: *_from_stdout or *_from_output
    if "_from_stdout" in var_lower or "_from_output" in var_lower:
        base_name = var_lower.replace("_from_stdout", "").replace("_from_output", "")
        
        # Try to extract stack name
        if "stack_name" in base_name or ("stack" in base_name and "name" in base_name):
            # Look for stack name patterns in output
            patterns = [
                r'stack[_\s]+name[:\s]+([a-zA-Z0-9\-_]+)',
                r'Stack[:\s]+([a-zA-Z0-9\-_]+)',
                r'([a-zA-Z0-9\-]+-[a-zA-Z0-9\-]+-[a-zA-Z0-9\-]+)',  # Common stack name pattern: project-env-stack
                r'CREATE_FAILED.*?([a-zA-Z0-9\-]+-[a-zA-Z0-9\-]+-[a-zA-Z0-9\-]+)',
                r'ROLLBACK.*?([a-zA-Z0-9\-]+-[a-zA-Z0-9\-]+-[a-zA-Z0-9\-]+)',
                r'arn:aws:cloudformation:[^:]+:stack/([a-zA-Z0-9\-]+)/',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, combined_output, re.IGNORECASE)
                if matches:
                    # Return the first non-empty match
                    for match in matches:
                        if match and len(match) > 3:  # Reasonable minimum length
                            return match
        
        # Try to extract stack status
        if "status" in base_name and "stack" in base_name:
            # Look for CloudFormation status patterns
            status_patterns = [
                r'status[:\s]+([A-Z_]+)',
                r'Status[:\s]+([A-Z_]+)',
                r'(CREATE_COMPLETE|CREATE_FAILED|CREATE_IN_PROGRESS|UPDATE_COMPLETE|UPDATE_FAILED|UPDATE_IN_PROGRESS|ROLLBACK_COMPLETE|ROLLBACK_FAILED|ROLLBACK_IN_PROGRESS|DELETE_COMPLETE|DELETE_FAILED|DELETE_IN_PROGRESS)',
            ]
            for pattern in status_patterns:
                matches = re.findall(pattern, combined_output, re.IGNORECASE)
                if matches:
                    return matches[0]
    
    return None

def execute_action(action: dict, scripts: list, execution_context: dict = None) -> dict:
    """
    Execute a single action from a pipeline step.
    
    Args:
        action: Action dictionary with 'type' and action-specific parameters
        scripts: List of available scripts
        execution_context: Context from previous steps (stdout, stderr, etc.)
    
    Returns:
        Dictionary with action execution results
    """
    if execution_context is None:
        execution_context = {}
    
    action_type = action.get("type", "").lower()
    result = {
        "action_type": action_type,
        "success": False,
        "output": "",
        "error": None,
        "data": {}
    }
    
    try:
        if action_type == "runscript":
            script_id = action.get("script_id", "")
            # Case-insensitive matching for script IDs
            script = next((s for s in scripts if s["id"].lower() == script_id.lower()), None)
            
            if not script:
                result["error"] = f"Script not found: {script_id}"
                return result
            
            # Get arguments, potentially substituting from context
            args = action.get("arguments", [])
            # Simple variable substitution from context
            processed_args = []
            for arg in args:
                if isinstance(arg, str) and "{{" in arg and "}}" in arg:
                    # Try to extract variable name and substitute
                    var_name = arg.replace("{{", "").replace("}}", "").strip()
                    if var_name in execution_context:
                        processed_args.append(str(execution_context[var_name]))
                    else:
                        processed_args.append(arg)
                else:
                    processed_args.append(arg)
            
            logger.info(f"Executing action: RunScript {script_id} with args: {processed_args}")
            script_result = run_script(
                script["path"],
                args=processed_args,
                cwd=Path(script["path"]).parent
            )
            
            result["success"] = script_result["success"]
            result["output"] = script_result.get("stdout", "")
            result["error"] = script_result.get("error")
            result["data"] = {
                "script_id": script_id,
                "script_name": script["name"],
                "stdout": script_result.get("stdout", ""),
                "stderr": script_result.get("stderr", ""),
                "exit_code": script_result.get("exit_code", -1)
            }
            
        elif action_type == "sendagent":
            if not OPENAI_AVAILABLE:
                result["error"] = "OpenAI agents not available"
                return result
            
            agent_name = action.get("agent", "")
            method = action.get("method", "")
            parameters = action.get("parameters", {})
            
            # Substitute parameters from context
            processed_params = {}
            for key, value in parameters.items():
                if isinstance(value, str):
                    # Support both {{variable}} and {variable} formats
                    if "{{" in value and "}}" in value:
                        # Double braces format: {{variable}}
                        var_name = value.replace("{{", "").replace("}}", "").strip()
                        if var_name in execution_context:
                            processed_params[key] = execution_context[var_name]
                        else:
                            # Try to extract from stdout/stderr if it's a pattern like "failed_stack_name_from_stdout"
                            extracted_value = extract_value_from_output(var_name, execution_context)
                            if extracted_value:
                                processed_params[key] = extracted_value
                            else:
                                # Try to extract from stdout/stderr if it's a pattern like "failed_stack_name_from_stdout"
                                extracted_value = extract_value_from_output(var_name, execution_context)
                                if extracted_value:
                                    processed_params[key] = extracted_value
                                    logger.info(f"Extracted {var_name} from output: {extracted_value}")
                                else:
                                    logger.warning(f"Variable {var_name} not found in context and could not be extracted from output, using literal value")
                                    processed_params[key] = value
                    elif value.startswith("{") and value.endswith("}") and value.count("{") == 1 and value.count("}") == 1:
                        # Single brace format: {variable}
                        var_name = value.replace("{", "").replace("}", "").strip()
                        if var_name in execution_context:
                            processed_params[key] = execution_context[var_name]
                        else:
                            # Try to extract from stdout/stderr
                            extracted_value = extract_value_from_output(var_name, execution_context)
                            if extracted_value:
                                processed_params[key] = extracted_value
                                logger.info(f"Extracted {var_name} from output: {extracted_value}")
                            else:
                                logger.warning(f"Variable {var_name} not found in context and could not be extracted from output, using literal value")
                                processed_params[key] = value
                    else:
                        processed_params[key] = value
                else:
                    processed_params[key] = value
            
            logger.info(f"Executing action: SendAgent {agent_name}.{method} with params: {processed_params}")
            
            try:
                if agent_name == "DeploymentAgent":
                    agent = get_deployment_agent()
                    if method == "analyze_deployment_status":
                        agent_result = agent.analyze_deployment_status(
                            stack_name=processed_params.get("stack_name", ""),
                            status=processed_params.get("status", ""),
                            events=processed_params.get("events")
                        )
                        result["success"] = agent_result.get("success", False)
                        result["output"] = agent_result.get("analysis", "")
                        result["data"] = agent_result
                    elif method == "recommend_deployment_strategy":
                        agent_result = agent.recommend_deployment_strategy(
                            project_name=processed_params.get("project_name", ""),
                            environment=processed_params.get("environment", ""),
                            stacks=processed_params.get("stacks", [])
                        )
                        result["success"] = agent_result.get("success", False)
                        result["output"] = agent_result.get("recommendation", "")
                        result["data"] = agent_result
                    elif method == "suggest_fix":
                        agent_result = agent.suggest_fix(
                            error_message=processed_params.get("error_message", ""),
                            context=processed_params.get("context")
                        )
                        result["success"] = agent_result.get("success", False)
                        result["output"] = agent_result.get("suggestion", "")
                        result["data"] = agent_result
                    else:
                        result["error"] = f"Unknown method: {method}"
                elif agent_name == "ScriptAnalysisAgent":
                    agent = get_script_analysis_agent()
                    if method == "analyze_execution":
                        agent_result = agent.analyze_execution(
                            script_name=processed_params.get("script_name", ""),
                            stdout=processed_params.get("stdout", ""),
                            stderr=processed_params.get("stderr", ""),
                            exit_code=processed_params.get("exit_code", -1),
                            error=processed_params.get("error")
                        )
                        result["success"] = agent_result.get("success", False)
                        result["output"] = agent_result.get("analysis", "")
                        result["data"] = agent_result
                    else:
                        result["error"] = f"Unknown method: {method}"
                else:
                    result["error"] = f"Unknown agent: {agent_name}"
                
                # If agent returned a next_action, store it for potential follow-up
                if result.get("data", {}).get("next_action"):
                    result["next_action"] = result["data"]["next_action"]
                    result["files_to_inspect"] = result["data"].get("files_to_inspect", [])
            except Exception as e:
                logger.error(f"Error calling agent {agent_name}.{method}: {e}", exc_info=True)
                result["error"] = str(e)
                
        elif action_type == "inspect":
            # Inspect action: Read a file and return its contents
            file_path = action.get("file_path", "")
            reason = action.get("reason", "File inspection requested")
            
            if not file_path:
                result["error"] = "No file_path specified for Inspect action"
                return result
            
            logger.info(f"Executing action: Inspect {file_path} - {reason}")
            
            try:
                # Try to resolve the file path
                # Could be relative to workspace, absolute, or relative to project
                file_path_obj = Path(file_path)
                
                # If not absolute, try relative to workspace root
                if not file_path_obj.is_absolute():
                    workspace_root = Path("D:/Workspace")
                    # Try workspace root first
                    potential_path = workspace_root / file_path
                    if not potential_path.exists():
                        # Try relative to current execution context if available
                        if "working_directory" in execution_context:
                            potential_path = Path(execution_context["working_directory"]) / file_path
                        if not potential_path.exists():
                            # Try as absolute path
                            potential_path = file_path_obj.resolve()
                    file_path_obj = potential_path
                
                if file_path_obj.exists() and file_path_obj.is_file():
                    # Read the file
                    try:
                        with open(file_path_obj, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        
                        result["success"] = True
                        result["output"] = f"File read successfully: {file_path}"
                        result["data"] = {
                            "file_path": str(file_path_obj),
                            "file_content": file_content,
                            "file_size": len(file_content),
                            "reason": reason
                        }
                    except UnicodeDecodeError:
                        # Try binary read for non-text files
                        with open(file_path_obj, 'rb') as f:
                            file_content = f.read()
                        result["success"] = True
                        result["output"] = f"File read as binary (size: {len(file_content)} bytes)"
                        result["data"] = {
                            "file_path": str(file_path_obj),
                            "file_content": file_content.decode('utf-8', errors='replace'),
                            "file_size": len(file_content),
                            "is_binary": True,
                            "reason": reason
                        }
                else:
                    result["error"] = f"File not found: {file_path_obj}"
                    result["data"] = {
                        "file_path": str(file_path_obj),
                        "reason": reason
                    }
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
                result["error"] = f"Error reading file: {str(e)}"
                result["data"] = {
                    "file_path": file_path,
                    "reason": reason
                }
                
        elif action_type == "continue":
            next_step = action.get("next_step")
            result["success"] = True
            result["data"] = {"next_step": next_step}
            
        elif action_type == "retry":
            max_attempts = action.get("max_attempts", 1)
            delay_seconds = action.get("delay_seconds", 0)
            result["success"] = True
            result["data"] = {
                "max_attempts": max_attempts,
                "delay_seconds": delay_seconds
            }
            
        elif action_type == "stop":
            reason = action.get("reason", "Pipeline stopped")
            result["success"] = True
            result["data"] = {"reason": reason, "should_stop": True}
            
        elif action_type == "wait":
            duration_seconds = action.get("duration_seconds", 0)
            logger.info(f"Waiting {duration_seconds} seconds...")
            time.sleep(duration_seconds)
            result["success"] = True
            result["data"] = {"waited_seconds": duration_seconds}
            
        elif action_type == "validate":
            script_id = action.get("script_id", "")
            # Case-insensitive matching for script IDs
            script = next((s for s in scripts if s["id"].lower() == script_id.lower()), None)
            
            if not script:
                result["error"] = f"Validation script not found: {script_id}"
                return result
            
            args = action.get("arguments", [])
            logger.info(f"Executing validation: {script_id}")
            script_result = run_script(
                script["path"],
                args=args,
                cwd=Path(script["path"]).parent
            )
            
            # Check expected result
            expected_result = action.get("expected_result", "exit_code == 0")
            validation_passed = False
            
            if "exit_code == 0" in expected_result:
                validation_passed = script_result.get("exit_code", -1) == 0
            
            if "'CREATE_COMPLETE' in stdout" in expected_result:
                validation_passed = validation_passed and "CREATE_COMPLETE" in script_result.get("stdout", "")
            
            result["success"] = validation_passed
            result["output"] = script_result.get("stdout", "")
            result["data"] = {
                "script_id": script_id,
                "validation_passed": validation_passed,
                "expected_result": expected_result,
                "stdout": script_result.get("stdout", ""),
                "stderr": script_result.get("stderr", ""),
                "exit_code": script_result.get("exit_code", -1)
            }
            
        else:
            result["error"] = f"Unknown action type: {action_type}"
            
    except Exception as e:
        logger.error(f"Error executing action {action_type}: {e}", exc_info=True)
        result["error"] = str(e)
        result["success"] = False
    
    return result

@app.route('/api/scripts', methods=['GET'])
def get_scripts():
    """Get list of all discovered Python scripts"""
    try:
        scripts = discover_aws_scripts()
        return jsonify({
            "success": True,
            "scripts": scripts,
            "count": len(scripts)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "scripts": [],
            "count": 0
        }), 500

@app.route('/api/yaml-files', methods=['GET'])
def get_yaml_files():
    """Get list of all discovered YAML files"""
    try:
        yaml_files = discover_aws_yaml_files()
        return jsonify({
            "success": True,
            "yaml_files": yaml_files,
            "count": len(yaml_files)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "yaml_files": [],
            "count": 0
        }), 500

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """Get list of all projects with their scripts and YAML files"""
    try:
        scripts = discover_aws_scripts()
        yaml_files = discover_aws_yaml_files()
        instruction_files = discover_build_instruction_files()
        
        # Group by project
        projects = {}
        
        for script in scripts:
            project = script["project"]
            if project not in projects:
                projects[project] = {
                    "name": project,
                    "scripts": [],
                    "yaml_files": [],
                    "instruction_files": []
                }
            projects[project]["scripts"].append(script)
        
        for yaml_file in yaml_files:
            project = yaml_file["project"]
            if project not in projects:
                projects[project] = {
                    "name": project,
                    "scripts": [],
                    "yaml_files": [],
                    "instruction_files": []
                }
            projects[project]["yaml_files"].append(yaml_file)
        
        for instruction_file in instruction_files:
            project = instruction_file["project"]
            if project not in projects:
                projects[project] = {
                    "name": project,
                    "scripts": [],
                    "yaml_files": [],
                    "instruction_files": []
                }
            projects[project]["instruction_files"].append(instruction_file)
        
        # Convert to list and sort
        projects_list = sorted(projects.values(), key=lambda x: x["name"])
        
        return jsonify({
            "success": True,
            "projects": projects_list,
            "count": len(projects_list)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "projects": [],
            "count": 0
        }), 500

@app.route('/api/scripts/<script_id>/run', methods=['POST'])
def run_script_endpoint(script_id):
    """Run a specific script with live streaming output"""
    try:
        # Find the script
        scripts = discover_aws_scripts()
        # Case-insensitive matching for script IDs
        script = next((s for s in scripts if s["id"].lower() == script_id.lower()), None)
        
        if not script:
            return jsonify({
                "success": False,
                "error": f"Script not found: {script_id}"
            }), 404
        
        # Get arguments from request body
        data = request.get_json() or {}
        args = data.get("args", [])
        if isinstance(args, str):
            args = args.split() if args else []
        
        # Check if streaming is requested (default to True)
        stream = data.get("stream", True)
        
        if stream:
            return run_script_streaming(script["path"], args=args, script_id=script_id, script_name=script["name"])
        else:
            # Fallback to non-streaming for compatibility
            result = run_script(script["path"], args=args)
            result["script_id"] = script_id
            result["script_name"] = script["name"]
            result["timestamp"] = datetime.now().isoformat()
            return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

def run_script_streaming(script_path, args=None, script_id=None, script_name=None):
    """Run a script with live streaming output using Server-Sent Events"""
    def generate():
        process = None
        try:
            # Prepare command
            cmd = [sys.executable, script_path]
            if args:
                cmd.extend(args)
            
            command_str = " ".join(cmd)
            start_time = time.time()
            
            # Set environment to use UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            if sys.platform == 'win32':
                env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
            
            # Start the process with live output
            process = subprocess.Popen(
                cmd,
                cwd=str(Path(script_path).parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                encoding='utf-8',
                errors='replace',
                env=env
            )
            
            # Store process for input handling
            if script_id:
                with script_lock:
                    running_scripts[script_id] = {
                        'process': process,
                        'start_time': start_time,
                        'command': command_str
                    }
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'start', 'command': command_str})}\n\n"
            
            # Read stdout and stderr in real-time using threads
            stdout_queue = queue.Queue()
            stderr_queue = queue.Queue()
            
            def read_stdout():
                try:
                    for line in iter(process.stdout.readline, ''):
                        if not line:
                            break
                        stdout_queue.put(('stdout', line))
                except Exception:
                    pass
                finally:
                    stdout_queue.put(('stdout', None))
            
            def read_stderr():
                try:
                    for line in iter(process.stderr.readline, ''):
                        if not line:
                            break
                        stderr_queue.put(('stderr', line))
                except Exception:
                    pass
                finally:
                    stderr_queue.put(('stderr', None))
            
            # Start reader threads
            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            stdout_lines = []
            stderr_lines = []
            stdout_done = False
            stderr_done = False
            
            # Read from queues and yield output
            while process.poll() is None or not stdout_done or not stderr_done:
                # Check stdout
                try:
                    output_type, line = stdout_queue.get(timeout=0.1)
                    if line is None:
                        stdout_done = True
                    else:
                        stdout_lines.append(line)
                        yield f"data: {json.dumps({'type': 'stdout', 'line': line.rstrip()})}\n\n"
                except queue.Empty:
                    pass
                
                # Check stderr
                try:
                    output_type, line = stderr_queue.get(timeout=0.1)
                    if line is None:
                        stderr_done = True
                    else:
                        stderr_lines.append(line)
                        yield f"data: {json.dumps({'type': 'stderr', 'line': line.rstrip()})}\n\n"
                except queue.Empty:
                    pass
                
                # Small delay to prevent busy waiting
                time.sleep(0.01)
            
            # Wait for threads to finish
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            
            # Wait for process to complete
            exit_code = process.wait()
            duration = time.time() - start_time
            
            # Clean up
            if script_id:
                with script_lock:
                    running_scripts.pop(script_id, None)
            
            # Send final status
            yield f"data: {json.dumps({'type': 'complete', 'exit_code': exit_code, 'duration': round(duration, 2), 'success': exit_code == 0, 'stdout': ''.join(stdout_lines), 'stderr': ''.join(stderr_lines)})}\n\n"
            
        except subprocess.TimeoutExpired:
            if process:
                process.kill()
            duration = time.time() - start_time if 'start_time' in locals() else 0
            yield f"data: {json.dumps({'type': 'error', 'message': 'Script execution timed out', 'duration': round(duration, 2)})}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming script: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if script_id:
                with script_lock:
                    running_scripts.pop(script_id, None)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/scripts/<script_id>/stop', methods=['POST'])
def stop_script(script_id):
    """Stop a running script"""
    try:
        with script_lock:
            if script_id not in running_scripts:
                return jsonify({
                    "success": False,
                    "error": f"No running script found for {script_id}"
                }), 404
            
            script_info = running_scripts[script_id]
            process = script_info['process']
            start_time = script_info.get('start_time', time.time())
            
            if process.poll() is not None:
                # Process has already finished
                running_scripts.pop(script_id, None)
                return jsonify({
                    "success": False,
                    "error": "Script has already finished"
                }), 400
            
            # Terminate the process
            try:
                process.terminate()
                # Wait a bit for graceful termination
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    process.kill()
                    process.wait()
                
                duration = time.time() - start_time
                
                # Clean up
                running_scripts.pop(script_id, None)
                
                return jsonify({
                    "success": True,
                    "message": "Script stopped successfully",
                    "duration": round(duration, 2)
                })
            except Exception as e:
                logger.error(f"Error stopping script: {e}", exc_info=True)
                # Try to kill it anyway
                try:
                    process.kill()
                    running_scripts.pop(script_id, None)
                except:
                    pass
                
                return jsonify({
                    "success": False,
                    "error": f"Failed to stop script: {str(e)}"
                }), 500
                
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/payload', methods=['POST'])
def receive_payload():
    """Receive and log payload from frontend"""
    try:
        payload = request.get_json()
        
        # Log to console
        print("\n" + "=" * 80)
        print(f"[{datetime.now().isoformat()}] Received Payload")
        print("=" * 80)
        print(json.dumps(payload, indent=2))
        print("=" * 80 + "\n")
        
        return jsonify({
            "success": True,
            "message": "Payload received and logged",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"\n[ERROR] Failed to process payload: {e}\n")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "workspace_root": str(WORKSPACE_ROOT),
        "workspace_exists": WORKSPACE_ROOT.exists()
    })

@app.route('/api/pipelines', methods=['GET'])
def list_pipelines():
    """List all saved pipeline files"""
    try:
        pipelines_dir = Path("pipelines")
        if not pipelines_dir.exists():
            return jsonify({
                "success": True,
                "pipelines": [],
                "count": 0
            })
        
        pipelines = []
        for pipeline_file in pipelines_dir.glob("*.json"):
            try:
                # Read metadata without loading full file
                stat = pipeline_file.stat()
                pipelines.append({
                    "name": pipeline_file.stem,
                    "filename": pipeline_file.name,
                    "path": str(pipeline_file),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                logger.warning(f"Error reading pipeline {pipeline_file}: {e}")
        
        return jsonify({
            "success": True,
            "pipelines": sorted(pipelines, key=lambda x: x["modified"], reverse=True),
            "count": len(pipelines)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/pipelines/<project_name>', methods=['GET'])
def get_pipeline(project_name):
    """Get the active pipeline by project name"""
    try:
        pipelines_dir = Path("pipelines")
        safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_project_name = safe_project_name.replace(' ', '_')
        pipeline_file = pipelines_dir / f"{safe_project_name}.json"
        
        if not pipeline_file.exists():
            return jsonify({
                "success": False,
                "error": f"Pipeline not found for project: {project_name}"
            }), 404
        
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)
        
        return jsonify({
            "success": True,
            "pipeline": pipeline_data,
            "is_deprecated": False
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/pipelines/<project_name>/versions', methods=['GET'])
def get_pipeline_versions(project_name):
    """Get all pipeline versions (active and deprecated) for a project"""
    try:
        pipelines_dir = Path("pipelines")
        safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_project_name = safe_project_name.replace(' ', '_')
        
        pipelines = []
        
        # Get active pipeline
        active_file = pipelines_dir / f"{safe_project_name}.json"
        if active_file.exists():
            try:
                with open(active_file, 'r', encoding='utf-8') as f:
                    pipeline_data = json.load(f)
                pipelines.append({
                    "filename": active_file.name,
                    "version": "active",
                    "is_deprecated": False,
                    "created_at": pipeline_data.get("timestamp", pipeline_data.get("pipeline", {}).get("created_at", "unknown")),
                    "pipeline": pipeline_data
                })
            except Exception as e:
                logger.error(f"Error reading active pipeline: {e}", exc_info=True)
        
        # Get deprecated pipelines
        pattern = re.compile(rf"^{re.escape(safe_project_name)}_deprecated_(\d+)\.json$")
        deprecated_files = []
        
        for file in pipelines_dir.glob(f"{safe_project_name}_deprecated_*.json"):
            match = pattern.match(file.name)
            if match:
                num = int(match.group(1))
                deprecated_files.append((num, file))
        
        # Sort by version number (ascending: 1, 2, 3...)
        deprecated_files.sort(key=lambda x: x[0])
        
        for num, file in deprecated_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    pipeline_data = json.load(f)
                pipelines.append({
                    "filename": file.name,
                    "version": f"deprecated_{num}",
                    "deprecated_version": num,
                    "is_deprecated": True,
                    "created_at": pipeline_data.get("timestamp", pipeline_data.get("pipeline", {}).get("created_at", "unknown")),
                    "pipeline": pipeline_data
                })
            except Exception as e:
                logger.error(f"Error reading deprecated pipeline {file.name}: {e}", exc_info=True)
        
        return jsonify({
            "success": True,
            "project_name": project_name,
            "pipelines": pipelines,
            "total_versions": len(pipelines),
            "active_count": sum(1 for p in pipelines if not p.get("is_deprecated", False)),
            "deprecated_count": sum(1 for p in pipelines if p.get("is_deprecated", False))
        })
    except Exception as e:
        logger.error(f"Error getting pipeline versions: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/pipelines/<project_name>/execute', methods=['POST'])
def execute_pipeline(project_name):
    """Execute a pipeline step by step"""
    try:
        # Get pipeline
        pipelines_dir = Path("pipelines")
        safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_project_name = safe_project_name.replace(' ', '_')
        pipeline_file = pipelines_dir / f"{safe_project_name}.json"
        
        if not pipeline_file.exists():
            return jsonify({
                "success": False,
                "error": f"Pipeline not found for project: {project_name}"
            }), 404
        
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)
        
        pipeline = pipeline_data.get("pipeline", {})
        steps = pipeline.get("steps", [])
        
        if not steps:
            return jsonify({
                "success": False,
                "error": "Pipeline has no steps"
            }), 400
        
        # Get request data
        data = request.get_json() or {}
        start_step = data.get("start_step", 1)
        stop_on_failure = data.get("stop_on_failure", True)
        
        # Find scripts
        scripts = discover_aws_scripts()
        
        # Execute steps
        execution_results = []
        current_step = start_step
        should_stop = False
        step_retry_counts = {}  # Track retry attempts per step
        
        for step in steps:
            step_num = step.get("step_number", 0)
            if step_num < start_step:
                continue
            
            if should_stop:
                break
            
            script_id = step.get("script_id", "")
            # Case-insensitive matching for script IDs
            script = next((s for s in scripts if s["id"].lower() == script_id.lower()), None)
            
            if not script:
                execution_results.append({
                    "step_number": step_num,
                    "script_name": step.get("script_name", "Unknown"),
                    "status": "error",
                    "error": f"Script not found: {script_id}",
                    "stdout": "",
                    "stderr": "",
                    "exit_code": -1,
                    "actions_executed": []
                })
                if stop_on_failure:
                    break
                continue
            
            # Run the script
            logger.info(f"Executing pipeline step {step_num}: {script['name']}")
            result = run_script(
                script["path"],
                args=step.get("arguments", []),
                cwd=Path(script["path"]).parent
            )
            
            # Build execution context for actions
            execution_context = {
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "exit_code": result.get("exit_code", -1),
                "script_name": script["name"],
                "script_id": script_id
            }
            
            execution_result = {
                "step_number": step_num,
                "script_name": step.get("script_name", script["name"]),
                "script_id": script_id,
                "description": step.get("description", ""),
                "status": "success" if result["success"] else "failed",
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "exit_code": result["exit_code"],
                "error": result.get("error"),
                "timestamp": datetime.now().isoformat(),
                "actions_executed": [],
                "command": result.get("command", ""),
                "duration_seconds": result.get("duration_seconds", 0),
                "working_directory": result.get("working_directory", ""),
                "script_path": result.get("script_path", ""),
                "stdout_lines": len(result.get("stdout", "").splitlines()) if result.get("stdout") else 0,
                "stderr_lines": len(result.get("stderr", "").splitlines()) if result.get("stderr") else 0,
                "script_path": result.get("script_path", ""),
                "arguments": step.get("arguments", []),
                "stdout_lines": len(result.get("stdout", "").splitlines()) if result.get("stdout") else 0,
                "stderr_lines": len(result.get("stderr", "").splitlines()) if result.get("stderr") else 0
            }
            
            # Execute actions based on success/failure
            actions_to_execute = []
            if result["success"]:
                on_success = step.get("on_success", {})
                actions_to_execute = on_success.get("actions", [])
                # Legacy support: if no actions but has next_step
                if not actions_to_execute and on_success.get("next_step"):
                    actions_to_execute = [{"type": "Continue", "next_step": on_success.get("next_step")}]
            else:
                on_failure = step.get("on_failure", {})
                actions_to_execute = on_failure.get("actions", [])
                # Legacy support: check for deployment_agent_call
                if not actions_to_execute:
                    deployment_agent_call = on_failure.get("deployment_agent_call", {})
                    if deployment_agent_call.get("enabled"):
                        actions_to_execute.append({
                            "type": "SendAgent",
                            "agent": "DeploymentAgent",
                            "method": "analyze_deployment_status",
                            "parameters": deployment_agent_call.get("context", {})
                        })
            
            # Execute actions
            for action in actions_to_execute:
                action_result = execute_action(action, scripts, execution_context)
                execution_result["actions_executed"].append({
                    "action": action,
                    "result": action_result
                })
                
                # Handle agent-returned next_action (for iterative agent analysis)
                # Check both direct next_action and nested in data
                next_action_from_result = action_result.get("next_action") or action_result.get("data", {}).get("next_action")
                if action.get("type") == "SendAgent" and next_action_from_result:
                    next_action = next_action_from_result
                    agent_name = action.get("agent", "")
                    method = action.get("method", "")
                    processed_params = action.get("_processed_params", {})
                    
                    # Check if agent wants to stop
                    if next_action.get("type") == "Stop":
                        logger.info(f"Agent {agent_name}.{method} requested to stop")
                        should_stop = True
                        execution_result["stopped_by_action"] = "Agent requested Stop"
                        execution_result["actions_executed"].append({
                            "action": {"type": "Stop", "reason": next_action.get("parameters", {}).get("reason", "Agent requested stop")},
                            "result": {"success": True, "output": "Agent requested stop"},
                            "requested_by_agent": True
                        })
                        break
                    
                    logger.info(f"Agent {agent_name}.{method} requested follow-up action: {next_action.get('type')}")
                    
                    # Execute the next_action requested by the agent
                    # The next_action from agent has parameters nested, so we need to flatten it
                    action_to_execute = {
                        "type": next_action.get("type"),
                        **next_action.get("parameters", {})
                    }
                    follow_up_result = execute_action(action_to_execute, scripts, execution_context)
                    execution_result["actions_executed"].append({
                        "action": next_action,
                        "result": follow_up_result,
                        "requested_by_agent": True
                    })
                    
                    # Update execution context with follow-up results
                    if follow_up_result.get("success"):
                        if next_action.get("type") == "Inspect":
                            # Add file content to context for agent
                            file_content = follow_up_result.get("data", {}).get("file_content", "")
                            file_path = follow_up_result.get("data", {}).get("file_path", "")
                            execution_context[f"inspected_file_{Path(file_path).name}"] = file_content
                        elif next_action.get("type") == "RunScript":
                            # Add script output to context
                            execution_context.update({
                                "follow_up_stdout": follow_up_result.get("output", ""),
                                "follow_up_stderr": follow_up_result.get("error", ""),
                                "follow_up_exit_code": follow_up_result.get("data", {}).get("exit_code", -1)
                            })
                    
                    # Send follow-up results back to the agent for continued analysis
                    if OPENAI_AVAILABLE and agent_name == "DeploymentAgent":
                        try:
                            from openai_agents import get_deployment_agent
                            agent = get_deployment_agent()
                            
                            # Build follow-up context
                            follow_up_context = f"""
Previous analysis context:
Stack Name: {processed_params.get('stack_name', '')}
Status: {processed_params.get('status', '')}

Follow-up action executed: {next_action.get('type')}
"""
                            
                            if next_action.get("type") == "Inspect":
                                follow_up_context += f"""
File inspected: {follow_up_result.get('data', {}).get('file_path', '')}
Reason: {action_to_execute.get('reason', '')}
File content:
{follow_up_result.get('data', {}).get('file_content', '')[:5000]}
"""
                            elif next_action.get("type") == "RunScript":
                                follow_up_context += f"""
Script executed: {action_to_execute.get('script_id', '')}
Output:
{follow_up_result.get('output', '')}
Error: {follow_up_result.get('error', 'None')}
"""
                            
                            # Call agent again with follow-up information
                            if method == "analyze_deployment_status":
                                # Create a new analysis with the follow-up information
                                follow_up_analysis = agent.analyze_deployment_status(
                                    stack_name=processed_params.get("stack_name", ""),
                                    status=processed_params.get("status", ""),
                                    events=None,
                                    follow_up_context=follow_up_context
                                )
                                # Update the action result with the follow-up analysis
                                if follow_up_analysis.get("success"):
                                    action_result["data"]["follow_up_analysis"] = follow_up_analysis.get("analysis", "")
                                    action_result["output"] += f"\n\n=== Follow-up Analysis (after {next_action.get('type')}) ===\n{follow_up_analysis.get('analysis', '')}"
                                    # Check if agent wants another action
                                    follow_up_next_action = follow_up_analysis.get("next_action")
                                    if follow_up_next_action:
                                        # Check if agent wants to stop
                                        if follow_up_next_action.get("type") == "Stop":
                                            logger.info(f"Agent {agent_name}.{method} requested to stop after follow-up")
                                            should_stop = True
                                            execution_result["stopped_by_action"] = "Agent requested Stop after follow-up"
                                            execution_result["actions_executed"].append({
                                                "action": {"type": "Stop", "reason": follow_up_next_action.get("parameters", {}).get("reason", "Agent requested stop")},
                                                "result": {"success": True, "output": "Agent requested stop"},
                                                "requested_by_agent": True
                                            })
                                            break
                                        action_result["next_action"] = follow_up_next_action
                                        action_result["data"]["next_action"] = follow_up_next_action
                        except Exception as e:
                            logger.error(f"Error sending follow-up to agent: {e}", exc_info=True)
                
                # Handle action-specific logic
                if action_result.get("data", {}).get("should_stop"):
                    should_stop = True
                    execution_result["stopped_by_action"] = action.get("type")
                    break
                
                # Handle Retry action
                if action.get("type") == "Retry" and not result["success"]:
                    retry_key = f"step_{step_num}"
                    retry_count = step_retry_counts.get(retry_key, 0)
                    max_attempts = action.get("max_attempts", 1)
                    delay_seconds = action.get("delay_seconds", 0)
                    
                    if retry_count < max_attempts:
                        step_retry_counts[retry_key] = retry_count + 1
                        logger.info(f"Retrying step {step_num} (attempt {retry_count + 1}/{max_attempts}) after {delay_seconds}s")
                        if delay_seconds > 0:
                            time.sleep(delay_seconds)
                        # Re-execute the step
                        result = run_script(
                            script["path"],
                            args=step.get("arguments", []),
                            cwd=Path(script["path"]).parent
                        )
                        execution_result["status"] = "success" if result["success"] else "failed"
                        execution_result["stdout"] = result["stdout"]
                        execution_result["stderr"] = result["stderr"]
                        execution_result["exit_code"] = result["exit_code"]
                        execution_result["retry_attempt"] = retry_count + 1
                        # Update context for next actions
                        execution_context.update({
                            "stdout": result.get("stdout", ""),
                            "stderr": result.get("stderr", ""),
                            "exit_code": result.get("exit_code", -1)
                        })
                
                # Handle Continue action
                if action.get("type") == "Continue":
                    next_step = action.get("next_step")
                    if next_step:
                        current_step = next_step
                        # Skip to the specified step
                        break
            
            execution_results.append(execution_result)
            
            # Stop on failure if configured
            if not result["success"] and stop_on_failure and not should_stop:
                break
        
        # Determine overall status
        all_success = all(r["status"] == "success" for r in execution_results)
        completed_steps = len(execution_results)
        total_steps = len(steps)
        
        return jsonify({
            "success": all_success,
            "project_name": project_name,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "execution_results": execution_results,
            "pipeline_complete": completed_steps == total_steps and all_success
        })
        
    except Exception as e:
        logger.error(f"Error executing pipeline: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/projects/<project_name>/analyze', methods=['POST'])
def analyze_project(project_name):
    """Analyze all files in a project using the ScriptAnalysisAgent"""
    if not OPENAI_AVAILABLE:
        error_msg = "OpenAI agents module not available."
        if OPENAI_ERROR:
            error_msg += f" {OPENAI_ERROR}"
        else:
            error_msg += " Please check that the 'openai' package is installed (pip install openai) and the API key is configured in .openai_secret"
        return jsonify({
            "success": False,
            "error": error_msg
        }), 503
    
    try:
        # Get project data
        scripts = discover_aws_scripts()
        yaml_files = discover_aws_yaml_files()
        
        # Filter by project
        project_scripts = [s for s in scripts if s["project"] == project_name]
        project_yaml_files = [y for y in yaml_files if y["project"] == project_name]
        
        if not project_scripts and not project_yaml_files:
            return jsonify({
                "success": False,
                "error": f"Project '{project_name}' not found or has no files"
            }), 404
        
        # Get file list from request (optional - can analyze all files or specific ones)
        data = request.get_json() or {}
        file_list = data.get("files", [])
        
        # If no specific files requested, analyze all project files
        if not file_list:
            file_list = [
                {"path": s["path"], "type": "script", "name": s["name"]}
                for s in project_scripts
            ] + [
                {"path": y["path"], "type": "yaml", "name": y["name"]}
                for y in project_yaml_files
            ]
        
        # Find and include README files for the project
        project_root = WORKSPACE_ROOT / project_name
        readme_files = []
        if project_root.exists():
            # Look for README files in project root and aws folder
            for readme_pattern in ["README.md", "README.txt", "readme.md", "readme.txt"]:
                readme_path = project_root / readme_pattern
                if readme_path.exists() and readme_path.is_file():
                    readme_files.append({
                        "path": str(readme_path),
                        "type": "readme",
                        "name": readme_pattern
                    })
                    break  # Use first found README
            
            # Also check aws folder
            aws_folder = project_root / "aws"
            if aws_folder.exists():
                for readme_pattern in ["README.md", "README.txt", "readme.md", "readme.txt"]:
                    readme_path = aws_folder / readme_pattern
                    if readme_path.exists() and readme_path.is_file():
                        readme_files.append({
                            "path": str(readme_path),
                            "type": "readme",
                            "name": f"aws/{readme_pattern}"
                        })
                        break
        
        # Add README files to file list
        file_list.extend(readme_files)
        
        # Read file contents
        files_content = {}
        for file_info in file_list:
            file_path = Path(file_info.get("path", ""))
            if file_path.exists() and file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        files_content[str(file_path)] = f.read()
                except Exception as e:
                    files_content[str(file_path)] = f"[Error reading file: {str(e)}]"
            else:
                files_content[str(file_path)] = f"[File not found: {file_path}]"
        
        # Use the analysis agent
        logger.info(f"Starting AI analysis for project: {project_name}")
        logger.info(f"Total files to analyze: {len(file_list)}")
        
        try:
            agent = get_script_analysis_agent()
            logger.info("ScriptAnalysisAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ScriptAnalysisAgent: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": f"Failed to initialize AI agent: {str(e)}",
                "traceback": traceback.format_exc()
            }), 500
        
        # Create a summary context for analysis
        summary = f"""
Project: {project_name}
Total Files: {len(file_list)}
Scripts: {len([f for f in file_list if f.get('type') == 'script'])}
YAML Files: {len([f for f in file_list if f.get('type') == 'yaml'])}

Files to analyze:
"""
        for file_info in file_list:
            summary += f"- {file_info.get('name', 'Unknown')} ({file_info.get('type', 'unknown')})\n"
        
        # Analyze with file contents
        logger.info("Sending analysis request to OpenAI...")
        try:
            result = agent.analyze_with_files(
                script_name=f"{project_name} Project Analysis",
                stdout=summary,
                stderr="",
                exit_code=0,
                files_content=files_content
            )
            logger.info(f"Analysis completed. Success: {result.get('success', False)}")
                
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": f"OpenAI API error: {str(e)}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }), 500
        
        # Add metadata
        result["project_name"] = project_name
        result["files_analyzed"] = list(files_content.keys())
        result["total_files"] = len(file_list)
        result["timestamp"] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/debug-agent', methods=['POST'])
def debug_agent():
    """Call the DeploymentAgent to analyze script output"""
    if not OPENAI_AVAILABLE:
        error_msg = "OpenAI agents module not available."
        if OPENAI_ERROR:
            error_msg += f" {OPENAI_ERROR}"
        else:
            error_msg += " Please check that the 'openai' package is installed (pip install openai) and the API key is configured in .openai_secret"
        return jsonify({
            "success": False,
            "error": error_msg
        }), 503
    
    try:
        data = request.get_json() or {}
        script_name = data.get("script_name", "")
        stdout = data.get("stdout", "")
        stderr = data.get("stderr", "")
        exit_code = data.get("exit_code", -1)
        error = data.get("error")
        
        # Get project analysis document if available
        # Try to find the most recent analysis document for the project
        context_document = ""
        try:
            project_name = script_name.split("_")[0] if "_" in script_name else ""
            if project_name:
                responses_dir = Path("responses")
                if responses_dir.exists():
                    # Find most recent analysis document
                    pattern = f"{project_name}_ScriptAnalysisAgent_*.txt"
                    analysis_files = list(responses_dir.glob(pattern))
                    if analysis_files:
                        # Sort by modification time, get most recent
                        most_recent = max(analysis_files, key=lambda p: p.stat().st_mtime)
                        with open(most_recent, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Extract the analysis/document part
                            if "Analysis:" in content:
                                context_document = content.split("Analysis:")[-1].strip()
                            else:
                                context_document = content
        except Exception as e:
            logger.debug(f"Could not load context document: {e}")
        
        # Combine stdout and stderr
        script_output = stdout
        if stderr:
            script_output += f"\n\nErrors:\n{stderr}"
        if error:
            script_output += f"\n\nError Message:\n{error}"
        
        # Call the DeploymentAgent
        agent = get_deployment_agent()
        result = agent.analyze_deployment_status(
            stack_name="",
            status="",
            script_output=script_output,
            context_document=context_document
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error calling debug agent: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/docker/check', methods=['GET'])
def check_docker():
    """Check if Docker is available"""
    global docker_manager, DOCKER_AVAILABLE
    if not DOCKER_AVAILABLE or docker_manager is None:
        # Try to initialize if not available
        try:
            from docker_deployment import get_docker_manager
            docker_manager = get_docker_manager(WORKSPACE_ROOT)
            DOCKER_AVAILABLE = True
        except Exception as e:
            logger.error(f"Failed to initialize Docker manager: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "available": False,
                "error": f"Docker deployment module not available: {str(e)}"
            }), 503
    
    try:
        result = docker_manager.check_docker_available()
        return jsonify({
            "success": True,
            **result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "available": False,
            "error": str(e)
        }), 500

@app.route('/api/projects/<project_name>/docker/instructions', methods=['GET'])
def get_docker_instructions(project_name):
    """Get Docker build instructions for a project"""
    global docker_manager, DOCKER_AVAILABLE
    if not DOCKER_AVAILABLE or docker_manager is None:
        # Try to initialize if not available
        try:
            from docker_deployment import get_docker_manager
            docker_manager = get_docker_manager(WORKSPACE_ROOT)
            DOCKER_AVAILABLE = True
        except Exception as e:
            logger.error(f"Failed to initialize Docker manager: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": f"Docker deployment module not available: {str(e)}"
            }), 503
    
    try:
        instructions = docker_manager.discover_build_instructions(project_name)
        if instructions:
            parsed = docker_manager.parse_build_instructions(instructions)
            return jsonify({
                "success": True,
                "instructions": instructions,
                "parsed_config": parsed
            })
        else:
            return jsonify({
                "success": False,
                "error": f"No build instructions found for project: {project_name}",
                "instructions": None
            }), 404
    except Exception as e:
        logger.error(f"Error getting Docker instructions: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/projects/<project_name>/docker/build', methods=['POST'])
def build_docker_image(project_name):
    """Build a Docker image for a project"""
    if not DOCKER_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Docker deployment module not available"
        }), 503
    
    try:
        # Get build instructions
        instructions = docker_manager.discover_build_instructions(project_name)
        if not instructions:
            return jsonify({
                "success": False,
                "error": f"No build instructions found for project: {project_name}"
            }), 404
        
        # Parse instructions
        config = docker_manager.parse_build_instructions(instructions)
        
        # Get build args from request
        data = request.get_json() or {}
        build_args = data.get("build_args", {})
        tag = data.get("tag")
        
        # Build the image (this will also extract the lambda package automatically)
        result = docker_manager.build_docker_image(config, build_args=build_args, tag=tag)
        
        # Add additional info about the extracted package
        if result.get("success") and result.get("extracted_package_path"):
            package_path = Path(result["extracted_package_path"])
            if package_path.exists():
                result["package_info"] = {
                    "path": str(package_path),
                    "size_mb": round(package_path.stat().st_size / (1024 * 1024), 2),
                    "ready_for_deployment": True
                }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error building Docker image: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/docker/deployments', methods=['GET'])
def list_deployments():
    """List all Docker deployments"""
    if not DOCKER_AVAILABLE:
        return jsonify({
            "success": False,
            "deployments": [],
            "error": "Docker deployment module not available"
        }), 503
    
    try:
        deployments = docker_manager.list_deployments()
        # Convert datetime objects to strings for JSON serialization
        for deployment in deployments:
            if "start_time" in deployment and isinstance(deployment["start_time"], datetime):
                deployment["start_time"] = deployment["start_time"].isoformat()
            if "end_time" in deployment and isinstance(deployment["end_time"], datetime):
                deployment["end_time"] = deployment["end_time"].isoformat()
            # Remove process object if present (not JSON serializable)
            if "process" in deployment:
                deployment["process"] = "running" if deployment.get("process") and hasattr(deployment["process"], "poll") and deployment["process"].poll() is None else "completed"
        
        return jsonify({
            "success": True,
            "deployments": deployments,
            "count": len(deployments)
        })
    except Exception as e:
        logger.error(f"Error listing deployments: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "deployments": [],
            "error": str(e)
        }), 500

@app.route('/api/docker/deployments/<deployment_id>', methods=['GET'])
def get_deployment_status(deployment_id):
    """Get status of a specific deployment"""
    if not DOCKER_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Docker deployment module not available"
        }), 503
    
    try:
        deployment = docker_manager.get_deployment_status(deployment_id)
        if deployment:
            # Convert datetime objects to strings
            if "start_time" in deployment and isinstance(deployment["start_time"], datetime):
                deployment["start_time"] = deployment["start_time"].isoformat()
            if "end_time" in deployment and isinstance(deployment["end_time"], datetime):
                deployment["end_time"] = deployment["end_time"].isoformat()
            # Remove process object
            if "process" in deployment:
                deployment["process"] = "running" if deployment.get("process") and hasattr(deployment["process"], "poll") and deployment["process"].poll() is None else "completed"
            
            return jsonify({
                "success": True,
                "deployment": deployment
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Deployment {deployment_id} not found"
            }), 404
    except Exception as e:
        logger.error(f"Error getting deployment status: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/docker/deployments/<deployment_id>/stop', methods=['POST'])
def stop_deployment(deployment_id):
    """Stop a running deployment"""
    if not DOCKER_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Docker deployment module not available"
        }), 503
    
    try:
        result = docker_manager.stop_deployment(deployment_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error stopping deployment: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/docker/containers/<container_name>/logs', methods=['GET'])
def get_container_logs(container_name):
    """Get logs from a Docker container"""
    if not DOCKER_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Docker deployment module not available"
        }), 503
    
    try:
        tail = request.args.get('tail', 100, type=int)
        result = docker_manager.get_container_logs(container_name, tail=tail)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting container logs: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/projects/<project_name>/docker/run', methods=['POST'])
def run_docker_container(project_name):
    """Run a Docker container for a project"""
    if not DOCKER_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Docker deployment module not available"
        }), 503
    
    try:
        data = request.get_json() or {}
        image_name = data.get("image_name")
        
        if not image_name:
            return jsonify({
                "success": False,
                "error": "image_name is required"
            }), 400
        
        container_name = data.get("container_name")
        ports = data.get("ports", {})
        environment_vars = data.get("environment_vars", {})
        volumes = data.get("volumes", {})
        detach = data.get("detach", True)
        
        result = docker_manager.run_docker_container(
            image_name=image_name,
            container_name=container_name,
            ports=ports,
            environment_vars=environment_vars,
            volumes=volumes,
            detach=detach
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error running Docker container: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/')
def index():
    """Serve the frontend index page"""
    return send_from_directory('frontend', 'index.html')

if __name__ == '__main__':
    print("=" * 80)
    print("LocalDeployer Backend Server")
    print("=" * 80)
    print(f"Workspace Root: {WORKSPACE_ROOT}")
    print(f"Workspace Exists: {WORKSPACE_ROOT.exists()}")
    print("=" * 80)
    print("\nStarting server on http://localhost:5000")
    print("Frontend available at: http://localhost:5000/")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

