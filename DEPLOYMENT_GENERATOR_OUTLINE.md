# Deployment Generator Outline

This document provides a comprehensive outline for generating deployment-ready scripts and configuration files for the LocalDeployer application. This outline is designed to be consumed by AI models (like Cursor instances) to generate the necessary files for a project.

## Overview

The LocalDeployer is a web-based tool that discovers and executes Python scripts from AWS deployment folders across multiple projects. For a project to be fully compatible with LocalDeployer, it must meet specific requirements and include certain configuration files.

## Minimum Requirements

### 1. Script Requirements

All deployment scripts **MUST** meet these criteria:

- **No Interactive Input**: Scripts must accept all required information via command-line arguments. They must NOT prompt for user input using `input()`, `raw_input()`, or similar functions.
- **Non-Blocking Execution**: Scripts should not require manual intervention or waiting for user confirmation during execution.
- **Clear Exit Codes**: Scripts should return appropriate exit codes (0 for success, non-zero for failure).
- **UTF-8 Output**: Scripts should output text in UTF-8 encoding to avoid encoding errors on Windows.
- **Error Handling**: Scripts should handle errors gracefully and provide clear error messages in stderr.

### 2. Required File Structure

Each project that wants to be deployable via LocalDeployer should have:

```
ProjectName/
├── aws/
│   ├── script_arguments.json    # REQUIRED: Script documentation
│   ├── deploy-stacks.py         # Example: Deployment script
│   ├── check-stacks-status.py   # Example: Status checking script
│   ├── requirements.txt         # Optional: Python dependencies
│   └── *.yaml                   # Optional: CloudFormation templates
└── README.md                    # Optional: Project documentation
```

## Required Files

### 1. script_arguments.json

**Location**: `ProjectName/aws/script_arguments.json`

**Purpose**: Documents all scripts in the `aws/` folder, their arguments, descriptions, and requirements. This file is consumed by LocalDeployer to display script information to users.

**Structure**:

```json
{
  "scripts": [
    {
      "name": "script-name.py",
      "description": "A clear, concise description of what this script does and its purpose in the deployment workflow.",
      "arguments": [
        {
          "name": "--argument-name",
          "type": "string|integer|boolean",
          "required": true|false,
          "description": "Clear description of what this argument does and when it should be used.",
          "default": "default-value (only if required is false)",
          "example": "--argument-name example-value"
        }
      ],
      "requirements": [
        "List of prerequisites or requirements",
        "e.g., 'AWS CLI must be configured'",
        "e.g., 'Python 3.8+'",
        "e.g., 'boto3 library installed'"
      ],
      "output": "Description of what the script outputs (stdout/stderr) and what users should expect to see."
    }
  ]
}
```

**Key Points**:
- The `name` field must exactly match the Python script filename (e.g., `deploy-stacks.py`)
- All arguments that the script accepts should be documented
- Required arguments must be marked with `"required": true`
- Provide clear examples for each argument
- List all external dependencies and prerequisites

### 2. Python Scripts

**Location**: `ProjectName/aws/*.py`

**Requirements**:
- Must use command-line argument parsing (e.g., `argparse`, `click`, `sys.argv`)
- Must NOT use interactive input functions
- Should use `argparse` or similar for robust argument handling
- Must handle errors gracefully
- Should provide helpful error messages
- Must return appropriate exit codes

**Example Structure**:

```python
#!/usr/bin/env python3
"""
Script description here.
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--env', required=True, help='Environment name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    
    args = parser.parse_args()
    
    try:
        # Script logic here
        # NO input() calls!
        # NO raw_input() calls!
        # NO interactive prompts!
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

### 3. requirements.txt (Optional but Recommended)

**Location**: `ProjectName/aws/requirements.txt`

**Purpose**: Lists Python dependencies needed to run the scripts.

**Example**:
```
boto3>=1.26.0
botocore>=1.29.0
```

## Minimum Script Set

For a project to be deployable, it should have at minimum:

1. **Deployment Script**: A script that deploys infrastructure (e.g., `deploy-stacks.py`)
   - Should accept environment and region arguments
   - Should handle stack creation and updates
   - Should provide clear output about deployment status

2. **Status/Verification Script**: A script that checks deployment status (e.g., `check-stacks-status.py`)
   - Should verify that deployments completed successfully
   - Should display current state of deployed resources
   - Should help with troubleshooting

3. **Cleanup Script** (Optional but Recommended): A script that can tear down deployments
   - Should safely remove deployed resources
   - Should have confirmation mechanisms via arguments (not interactive prompts)

## Script Execution Flow

When LocalDeployer executes a script:

1. User selects a script from the UI
2. User provides arguments (based on `script_arguments.json` documentation)
3. LocalDeployer executes: `python script.py --arg1 value1 --arg2 value2`
4. Script output is streamed live to the UI
5. Exit code determines success/failure
6. Output is captured for debugging/analysis

## Common Patterns to Avoid

### ❌ DO NOT:
- Use `input()` or `raw_input()` for user input
- Use `getpass.getpass()` for passwords (use environment variables or config files)
- Require manual file editing during execution
- Use blocking prompts that wait for user confirmation
- Print prompts like "Press Enter to continue..."
- Use interactive menus or choice prompts

### ✅ DO:
- Use command-line arguments for all configuration
- Use environment variables for sensitive data
- Use configuration files for complex settings
- Provide clear error messages
- Return appropriate exit codes
- Document all arguments in `script_arguments.json`

## Example: Complete Project Setup

```
MyProject/
├── aws/
│   ├── script_arguments.json
│   ├── deploy-stacks.py
│   ├── check-stacks-status.py
│   ├── destroy-stacks.py
│   ├── requirements.txt
│   └── templates/
│       ├── stack1.yaml
│       └── stack2.yaml
└── README.md
```

### script_arguments.json Example:

```json
{
  "scripts": [
    {
      "name": "deploy-stacks.py",
      "description": "Deploys all CloudFormation stacks for the specified environment. Creates stacks if they don't exist, updates them if they do.",
      "arguments": [
        {
          "name": "--env",
          "type": "string",
          "required": true,
          "description": "Target environment (production, staging, development)",
          "example": "--env production"
        },
        {
          "name": "--region",
          "type": "string",
          "required": false,
          "description": "AWS region for deployment",
          "default": "us-east-1",
          "example": "--region us-west-2"
        }
      ],
      "requirements": [
        "AWS CLI configured with appropriate credentials",
        "Python 3.8+",
        "boto3>=1.26.0"
      ],
      "output": "Prints stack deployment progress, ARNs, and completion status. Errors are printed to stderr."
    },
    {
      "name": "check-stacks-status.py",
      "description": "Checks the current status of all deployed stacks and displays recent events.",
      "arguments": [
        {
          "name": "--env",
          "type": "string",
          "required": true,
          "description": "Environment to check",
          "example": "--env production"
        }
      ],
      "requirements": [
        "AWS CLI configured",
        "Python 3.8+",
        "boto3>=1.26.0"
      ],
      "output": "Displays stack status table with creation time, last update, and status. Shows recent stack events for troubleshooting."
    }
  ]
}
```

## Integration Checklist

When generating files for a project, ensure:

- [ ] All scripts use command-line arguments (no interactive input)
- [ ] `script_arguments.json` exists in `aws/` folder
- [ ] All scripts are documented in `script_arguments.json`
- [ ] All required arguments are marked as `"required": true`
- [ ] All optional arguments have default values or are clearly optional
- [ ] Examples are provided for each argument
- [ ] Requirements/dependencies are listed
- [ ] Scripts handle errors gracefully
- [ ] Scripts return appropriate exit codes
- [ ] No `input()`, `raw_input()`, or similar functions are used
- [ ] Scripts output UTF-8 encoded text
- [ ] `requirements.txt` lists all Python dependencies (if any)

## Notes for AI Models

When generating deployment scripts and configuration:

1. **Always check for interactive input**: Search the script for `input(`, `raw_input(`, `getpass`, and similar functions. Remove or replace them with command-line arguments.

2. **Use argparse**: Prefer `argparse` over manual `sys.argv` parsing for better error handling and help text.

3. **Document thoroughly**: The `script_arguments.json` file is the primary way users understand how to use scripts. Be comprehensive and clear.

4. **Test assumptions**: If generating scripts, ensure they can run non-interactively. Test with various argument combinations.

5. **Error handling**: Scripts should fail gracefully with clear error messages, not crash with stack traces.

6. **Exit codes**: Use `sys.exit(0)` for success and `sys.exit(1)` (or other non-zero) for failures.

7. **Output format**: Prefer structured output when possible, but plain text is acceptable. Avoid emoji or special characters that might cause encoding issues (though UTF-8 is supported).

## Summary

The LocalDeployer requires:
- **Scripts with no interactive input** - all configuration via arguments
- **script_arguments.json file** - comprehensive documentation of all scripts
- **Clear error handling** - graceful failures with helpful messages
- **Appropriate exit codes** - 0 for success, non-zero for failure
- **UTF-8 output** - for proper display in the web UI

By following this outline, AI models can generate deployment-ready projects that integrate seamlessly with LocalDeployer.

