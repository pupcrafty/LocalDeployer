#!/usr/bin/env python3
"""
Docker Deployment Module for LocalDeployer
Abstract architecture for Docker container deployment across projects
"""
import os
import json
import subprocess
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import threading
import queue

logger = logging.getLogger(__name__)

class DockerDeploymentManager:
    """
    Abstract Docker deployment manager that can work with any project
    """
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.deployments = {}  # Track active deployments
        self.deployment_lock = threading.Lock()
        
    def check_docker_available(self) -> Dict[str, Any]:
        """Check if Docker is installed and available"""
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                docker_version = result.stdout.strip()
                return {
                    "available": True,
                    "version": docker_version,
                    "error": None
                }
            else:
                return {
                    "available": False,
                    "version": None,
                    "error": "Docker command failed"
                }
        except FileNotFoundError:
            return {
                "available": False,
                "version": None,
                "error": "Docker not found. Please install Docker Desktop."
            }
        except Exception as e:
            return {
                "available": False,
                "version": None,
                "error": f"Error checking Docker: {str(e)}"
            }
    
    def discover_build_instructions(self, project_name: str) -> Optional[Dict[str, Any]]:
        """
        Discover Lambda_build_instructions or similar build instruction files
        in the project's aws folder
        """
        project_path = self.workspace_root / project_name
        aws_folder = project_path / "aws"
        
        if not aws_folder.exists():
            return None
        
        # Look for various build instruction file patterns
        instruction_patterns = [
            "Lambda_build_instructions*",
            "*build_instructions*",
            "*docker*instructions*",
            "*deployment*instructions*",
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml"
        ]
        
        instructions = {}
        
        for pattern in instruction_patterns:
            for file_path in aws_folder.rglob(pattern):
                if file_path.is_file():
                    file_type = self._detect_file_type(file_path)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        instructions[str(file_path.relative_to(aws_folder))] = {
                            "path": str(file_path),
                            "relative_path": str(file_path.relative_to(aws_folder)),
                            "type": file_type,
                            "content": content,
                            "size": len(content)
                        }
                    except Exception as e:
                        logger.warning(f"Failed to read {file_path}: {e}")
        
        if instructions:
            return {
                "project": project_name,
                "project_path": str(project_path),
                "aws_folder": str(aws_folder),
                "instructions": instructions,
                "found_at": datetime.now().isoformat()
            }
        
        return None
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect the type of build instruction file"""
        name_lower = file_path.name.lower()
        
        if "dockerfile" in name_lower:
            return "dockerfile"
        elif "docker-compose" in name_lower:
            return "docker_compose"
        elif "lambda_build" in name_lower or "build_instructions" in name_lower:
            return "build_instructions"
        elif file_path.suffix in ['.yml', '.yaml']:
            return "yaml_config"
        elif file_path.suffix == '.json':
            return "json_config"
        else:
            return "text_instructions"
    
    def parse_build_instructions(self, instructions_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse build instructions to extract Docker deployment configuration
        Returns a standardized deployment configuration
        """
        parsed = {
            "project_name": instructions_data.get("project"),
            "dockerfile_path": None,
            "docker_compose_path": None,
            "build_context": None,
            "image_name": None,
            "build_args": {},
            "environment_vars": {},
            "ports": {},
            "volumes": {},
            "commands": [],
            "instructions_text": "",
            "metadata": {}
        }
        
        instructions = instructions_data.get("instructions", {})
        
        for rel_path, file_data in instructions.items():
            file_type = file_data.get("type")
            content = file_data.get("content", "")
            
            if file_type == "dockerfile":
                parsed["dockerfile_path"] = file_data.get("path")
                parsed["build_context"] = str(Path(file_data.get("path")).parent)
                # Extract image name from Dockerfile if present
                for line in content.split('\n'):
                    if line.strip().startswith('# IMAGE_NAME:'):
                        parsed["image_name"] = line.split(':', 1)[1].strip()
                    elif line.strip().startswith('# BUILD_ARGS:'):
                        args_str = line.split(':', 1)[1].strip()
                        parsed["build_args"] = json.loads(args_str) if args_str else {}
            
            elif file_type == "docker_compose":
                parsed["docker_compose_path"] = file_data.get("path")
                try:
                    try:
                        import yaml
                    except ImportError:
                        logger.warning("PyYAML not installed. Cannot parse docker-compose file. Install with: pip install PyYAML")
                        parsed["instructions_text"] = content
                        continue
                    
                    compose_data = yaml.safe_load(content)
                    if compose_data:
                        services = compose_data.get("services", {})
                        for service_name, service_config in services.items():
                            if not parsed["image_name"]:
                                parsed["image_name"] = service_config.get("image", f"{parsed['project_name']}:latest")
                            parsed["ports"] = service_config.get("ports", {})
                            parsed["environment_vars"] = service_config.get("environment", {})
                            parsed["volumes"] = service_config.get("volumes", {})
                except Exception as e:
                    logger.warning(f"Failed to parse docker-compose file: {e}")
                    parsed["instructions_text"] = content
            
            elif file_type == "build_instructions":
                parsed["instructions_text"] = content
                # Try to extract structured info from text
                parsed["metadata"]["raw_instructions"] = content
        
        # Generate default image name if not found
        if not parsed["image_name"]:
            project_name_safe = instructions_data.get("project", "project").lower().replace(" ", "-")
            parsed["image_name"] = f"{project_name_safe}:latest"
        
        return parsed
    
    def build_docker_image(
        self,
        deployment_config: Dict[str, Any],
        build_args: Optional[Dict[str, str]] = None,
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a Docker image from the deployment configuration
        """
        deployment_id = f"{deployment_config['project_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        dockerfile_path = deployment_config.get("dockerfile_path")
        build_context = deployment_config.get("build_context")
        image_name = tag or deployment_config.get("image_name", "project:latest")
        
        if not dockerfile_path or not Path(dockerfile_path).exists():
            return {
                "success": False,
                "error": "Dockerfile not found",
                "deployment_id": deployment_id
            }
        
        if not build_context:
            build_context = str(Path(dockerfile_path).parent)
        
        # Prepare build command
        cmd = ["docker", "build"]
        
        # Add build args
        if build_args:
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])
        
        # Add tag
        cmd.extend(["-t", image_name])
        
        # Add dockerfile and context
        cmd.extend(["-f", dockerfile_path, build_context])
        
        logger.info(f"Building Docker image: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )
            
            # Store process for tracking
            with self.deployment_lock:
                self.deployments[deployment_id] = {
                    "id": deployment_id,
                    "type": "build",
                    "status": "building",
                    "process": process,
                    "start_time": datetime.now(),
                    "image_name": image_name,
                    "config": deployment_config
                }
            
            # Read output in real-time
            stdout_lines = []
            stderr_lines = []
            
            def read_output():
                try:
                    for line in iter(process.stdout.readline, ''):
                        if not line:
                            break
                        stdout_lines.append(line)
                        logger.debug(f"Docker build stdout: {line.strip()}")
                except (UnicodeDecodeError, UnicodeError) as e:
                    logger.warning(f"Unicode decode error in stdout: {e}")
                except Exception as e:
                    logger.warning(f"Error reading stdout: {e}")
            
            def read_errors():
                try:
                    for line in iter(process.stderr.readline, ''):
                        if not line:
                            break
                        stderr_lines.append(line)
                        logger.debug(f"Docker build stderr: {line.strip()}")
                except (UnicodeDecodeError, UnicodeError) as e:
                    logger.warning(f"Unicode decode error in stderr: {e}")
                except Exception as e:
                    logger.warning(f"Error reading stderr: {e}")
            
            stdout_thread = threading.Thread(target=read_output, daemon=True)
            stderr_thread = threading.Thread(target=read_errors, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            exit_code = process.wait()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)
            
            # Extract lambda package if build was successful
            extracted_package_path = None
            if exit_code == 0:
                extracted_package_path = self._extract_lambda_package(
                    image_name, 
                    deployment_config.get("project_name"),
                    build_context
                )
            
            # Update deployment status
            with self.deployment_lock:
                if deployment_id in self.deployments:
                    self.deployments[deployment_id]["status"] = "completed" if exit_code == 0 else "failed"
                    self.deployments[deployment_id]["exit_code"] = exit_code
                    self.deployments[deployment_id]["stdout"] = stdout
                    self.deployments[deployment_id]["stderr"] = stderr
                    self.deployments[deployment_id]["end_time"] = datetime.now()
                    if extracted_package_path:
                        self.deployments[deployment_id]["extracted_package"] = extracted_package_path
            
            return {
                "success": exit_code == 0,
                "deployment_id": deployment_id,
                "image_name": image_name,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "extracted_package_path": extracted_package_path,
                "error": None if exit_code == 0 else f"Docker build failed with exit code {exit_code}"
            }
            
        except Exception as e:
            logger.error(f"Error building Docker image: {e}", exc_info=True)
            with self.deployment_lock:
                if deployment_id in self.deployments:
                    self.deployments[deployment_id]["status"] = "failed"
                    self.deployments[deployment_id]["error"] = str(e)
            
            return {
                "success": False,
                "deployment_id": deployment_id,
                "error": str(e),
                "stdout": "",
                "stderr": ""
            }
    
    def run_docker_container(
        self,
        image_name: str,
        container_name: Optional[str] = None,
        ports: Optional[Dict[str, str]] = None,
        environment_vars: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        detach: bool = True
    ) -> Dict[str, Any]:
        """
        Run a Docker container from an image
        """
        deployment_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not container_name:
            container_name = f"{image_name.split(':')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cmd = ["docker", "run"]
        
        if detach:
            cmd.append("-d")
        
        # Add container name
        cmd.extend(["--name", container_name])
        
        # Add port mappings
        if ports:
            for host_port, container_port in ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])
        
        # Add environment variables
        if environment_vars:
            for key, value in environment_vars.items():
                cmd.extend(["-e", f"{key}={value}"])
        
        # Add volumes
        if volumes:
            for host_path, container_path in volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        # Add image name
        cmd.append(image_name)
        
        logger.info(f"Running Docker container: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            container_id = result.stdout.strip() if result.returncode == 0 else None
            
            with self.deployment_lock:
                self.deployments[deployment_id] = {
                    "id": deployment_id,
                    "type": "run",
                    "status": "running" if result.returncode == 0 else "failed",
                    "container_name": container_name,
                    "container_id": container_id,
                    "image_name": image_name,
                    "start_time": datetime.now(),
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            
            return {
                "success": result.returncode == 0,
                "deployment_id": deployment_id,
                "container_name": container_name,
                "container_id": container_id,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": None if result.returncode == 0 else result.stderr
            }
            
        except Exception as e:
            logger.error(f"Error running Docker container: {e}", exc_info=True)
            return {
                "success": False,
                "deployment_id": deployment_id,
                "error": str(e),
                "stdout": "",
                "stderr": ""
            }
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a deployment"""
        with self.deployment_lock:
            return self.deployments.get(deployment_id)
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all tracked deployments"""
        with self.deployment_lock:
            return list(self.deployments.values())
    
    def stop_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Stop a running deployment"""
        with self.deployment_lock:
            deployment = self.deployments.get(deployment_id)
            if not deployment:
                return {
                    "success": False,
                    "error": f"Deployment {deployment_id} not found"
                }
            
            if deployment.get("type") == "build":
                process = deployment.get("process")
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    deployment["status"] = "stopped"
                    deployment["end_time"] = datetime.now()
            
            elif deployment.get("type") == "run":
                container_name = deployment.get("container_name")
                if container_name:
                    try:
                        subprocess.run(
                            ["docker", "stop", container_name],
                            capture_output=True,
                            timeout=10
                        )
                        deployment["status"] = "stopped"
                        deployment["end_time"] = datetime.now()
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Failed to stop container: {str(e)}"
                        }
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "message": "Deployment stopped"
            }
    
    def get_container_logs(self, container_name: str, tail: int = 100) -> Dict[str, Any]:
        """Get logs from a running container"""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(tail), container_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "success": result.returncode == 0,
                "logs": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "logs": ""
            }
    
    def _extract_lambda_package(
        self, 
        image_name: str, 
        project_name: str,
        build_context: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract lambda-package.zip from the built Docker image.
        Saves it to the project's aws folder.
        
        Returns:
            Path to the extracted package file, or None if extraction failed
        """
        try:
            # Determine where to save the package
            if build_context:
                aws_folder = Path(build_context)
            else:
                aws_folder = self.workspace_root / project_name / "aws"
            
            if not aws_folder.exists():
                logger.warning(f"AWS folder not found: {aws_folder}")
                return None
            
            # Create a temporary container to extract from
            logger.info(f"Extracting lambda-package.zip from image {image_name}...")
            create_result = subprocess.run(
                ["docker", "create", image_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if create_result.returncode != 0:
                logger.error(f"Failed to create container: {create_result.stderr}")
                return None
            
            container_id = create_result.stdout.strip()
            
            try:
                # Copy the package from container
                package_path_in_container = "/build/lambda-package.zip"
                output_path = aws_folder / "lambda-package-docker.zip"
                
                copy_result = subprocess.run(
                    ["docker", "cp", f"{container_id}:{package_path_in_container}", str(output_path)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if copy_result.returncode == 0 and output_path.exists():
                    package_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
                    logger.info(f"Successfully extracted package to {output_path} ({package_size:.2f} MB)")
                    return str(output_path)
                else:
                    logger.error(f"Failed to copy package: {copy_result.stderr}")
                    # Try alternative location
                    alt_path = aws_folder / "lambda-package.zip"
                    copy_result2 = subprocess.run(
                        ["docker", "cp", f"{container_id}:{package_path_in_container}", str(alt_path)],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if copy_result2.returncode == 0 and alt_path.exists():
                        logger.info(f"Extracted to alternative location: {alt_path}")
                        return str(alt_path)
                    return None
                    
            finally:
                # Always remove the temporary container
                subprocess.run(
                    ["docker", "rm", container_id],
                    capture_output=True,
                    timeout=10
                )
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout while extracting lambda package")
            return None
        except Exception as e:
            logger.error(f"Error extracting lambda package: {e}", exc_info=True)
            return None

# Global instance
_docker_manager = None

def get_docker_manager(workspace_root: Path = None) -> DockerDeploymentManager:
    """Get or create the global Docker deployment manager"""
    global _docker_manager
    if _docker_manager is None:
        if workspace_root is None:
            workspace_root = Path("D:/Workspace")
        _docker_manager = DockerDeploymentManager(workspace_root)
    return _docker_manager

