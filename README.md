# LocalDeployer

A web application to discover and run Python scripts from AWS folders across your workspace, capture console output, and send it as payloads to a Python backend.

## Features

- 🔍 **Auto-discovery**: Automatically finds all Python scripts in `aws/` folders within `D:\Workspace`
- ▶️ **Script Execution**: Run scripts with custom arguments and capture console output
- 📤 **Payload Management**: Send script execution results as structured payloads to the backend
- 🎨 **Modern UI**: Clean, responsive web interface
- 📊 **Real-time Output**: See script output in real-time as it executes

## Setup

### Prerequisites

- Python 3.7 or higher
- Python scripts in AWS folders within `D:\Workspace`

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key (optional, for AI agents):
   - Create a file named `.openai_secret` in the project root
   - Add your OpenAI API key to this file (one line, no quotes)
   - Example: `sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - This file is already in `.gitignore` for security
   - Configure the model in `config.json` (default: `gpt-4o-mini`)
     - Available models: `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`, `gpt-4-turbo`
     - Newer models like GPT-5 may require specific API access tiers

2. Start the backend server:
```bash
python backend.py
```

The server will start on `http://localhost:5000`

3. Open the frontend:
   - Open `frontend/index.html` in your web browser
   - Or serve it using a local web server (e.g., `python -m http.server 8000` in the frontend directory)

## Usage

1. **Discover Scripts**: The app automatically scans `D:\Workspace` for Python scripts in `aws/` folders
2. **Select a Script**: Click on a script from the sidebar to view its details
3. **Add Arguments** (optional): Enter command-line arguments in the input field
4. **Run Script**: Click "Run Script" to execute it and see the output
5. **Send Payload**: Click "Send Payload" to send the execution results to the backend (logged to console)

## API Endpoints

- `GET /api/scripts` - Get list of all discovered scripts
- `POST /api/scripts/<script_id>/run` - Run a specific script
- `POST /api/payload` - Receive and log payload from frontend
- `GET /api/health` - Health check endpoint

## Project Structure

```
LocalDeployer/
├── backend.py          # Flask backend server
├── openai_agents.py    # OpenAI agents module (ScriptAnalysisAgent, DeploymentAgent)
├── .openai_secret      # OpenAI API key (create this file, not in git)
├── frontend/
│   └── index.html      # Web interface
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Notes

- Scripts are executed in their own directory context
- Script execution has a 5-minute timeout
- Output includes both stdout and stderr
- Payloads sent to the backend are logged to the console with timestamps

