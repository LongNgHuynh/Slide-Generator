import os
import json
import logging
import asyncio
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import uuid

logger = logging.getLogger(__name__)

class MCPServerConfig(BaseModel):
    """Configuration for an MCP server"""
    name: str = Field(description="Name of the MCP server")
    command: str = Field(description="Command to start the MCP server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")
    cwd: Optional[str] = Field(default=None, description="Working directory")

class MCPTool(BaseModel):
    """Represents an MCP tool"""
    name: str
    description: str
    inputSchema: Dict[str, Any]

class MCPClient:
    """Client for connecting to MCP servers"""
    
    def __init__(self):
        self.server_processes: Dict[str, subprocess.Popen] = {}
        self.available_tools: List[MCPTool] = []
        self.servers: Dict[str, MCPServerConfig] = {}
        
    def discover_servers(self) -> Dict[str, MCPServerConfig]:
        """Discover available MCP servers from environment and configuration"""
        servers = {}
        
        # Check for official MCP servers that can be installed
        official_servers = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "description": "File system operations"
            },
            "everything": {
                "command": "npx", 
                "args": ["-y", "@modelcontextprotocol/server-everything"],
                "description": "Test server with all MCP features"
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "description": "GitHub API integration",
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")}
            }
        }
        
        # Add official servers if they have required environment variables or are always available
        for name, config in official_servers.items():
            if name == "github":
                # GitHub server requires token
                if os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
                    servers[name] = MCPServerConfig(
                        name=name,
                        command=config["command"],
                        args=config["args"],
                        env=config.get("env", {})
                    )
            else:
                # Filesystem and everything servers don't require special env vars
                servers[name] = MCPServerConfig(
                    name=name,
                    command=config["command"],
                    args=config["args"],
                    env=config.get("env", {})
                )
        
        # Check for Firecrawl API key and add a community server
        if os.getenv("FIRECRAWL_API_KEY"):
            servers["firecrawl"] = MCPServerConfig(
                name="firecrawl",
                command="npx",
                args=["-y", "@mendable/firecrawl-mcp"],
                env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")}
            )
        
        # Check for other common API keys and add relevant servers
        api_key_mappings = {
            "ANTHROPIC_API_KEY": {
                "name": "anthropic",
                "command": "npx", 
                "args": ["-y", "@anthropic-ai/mcp-server"],
                "env": {"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")}
            },
            "OPENAI_API_KEY": {
                "name": "openai",
                "command": "npx",
                "args": ["-y", "@openai/mcp-server"], 
                "env": {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
            }
        }
        
        # Note: The above servers may not exist, they're examples of how the system would work
        
        # Check for batch configuration in MCP_SERVERS environment variable
        batch_config = os.getenv("MCP_SERVERS")
        if batch_config:
            try:
                config_data = json.loads(batch_config)
                for name, server_config in config_data.items():
                    servers[name] = MCPServerConfig(
                        name=name,
                        command=server_config.get("command", ""),
                        args=server_config.get("args", []),
                        env=server_config.get("env", {}),
                        cwd=server_config.get("cwd")
                    )
            except json.JSONDecodeError:
                logger.error("Invalid JSON in MCP_SERVERS environment variable")
        
        self.servers = servers
        return servers
    
    def start_server(self, server_name: str) -> bool:
        """Start an MCP server process"""
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not found in configuration")
            return False
            
        if server_name in self.server_processes:
            logger.info(f"Server {server_name} already running")
            return True
            
        config = self.servers[server_name]
        
        # Check if the command exists before trying to start the server
        if config.command == "npx":
            if not self._check_command_exists("npx"):
                logger.warning(f"NPX not found. Server {server_name} requires Node.js to be installed. Skipping.")
                return False
        elif config.command == "node":
            if not self._check_command_exists("node"):
                logger.warning(f"Node.js not found. Server {server_name} requires Node.js to be installed. Skipping.")
                return False
        
        try:
            env = os.environ.copy()
            if config.env:
                env.update(config.env)
                
            # Filter out None values from env
            env = {k: v for k, v in env.items() if v is not None}
            
            # Use cmd /c on Windows for better PATH resolution
            import platform
            if platform.system() == "Windows" and config.command in ["npx", "node", "npm"]:
                cmd_args = ["cmd", "/c", config.command] + config.args
            else:
                cmd_args = [config.command] + config.args
            
            process = subprocess.Popen(
                cmd_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=config.cwd,
                text=True
            )
            
            self.server_processes[server_name] = process
            logger.info(f"Started MCP server: {server_name}")
            return True
            
        except FileNotFoundError as e:
            logger.warning(f"Command not found for MCP server {server_name}: {config.command}. Please install required dependencies.")
            return False
        except Exception as e:
            logger.error(f"Failed to start MCP server {server_name}: {str(e)}")
            return False
    
    def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists in the system PATH"""
        import platform
        
        try:
            if platform.system() == "Windows":
                # Use cmd /c on Windows to ensure proper PATH resolution
                result = subprocess.run(["cmd", "/c", command, "--version"], 
                                      capture_output=True, 
                                      timeout=5)
            else:
                result = subprocess.run([command, "--version"], 
                                      capture_output=True, 
                                      timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def stop_server(self, server_name: str):
        """Stop an MCP server process"""
        if server_name in self.server_processes:
            process = self.server_processes[server_name]
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            del self.server_processes[server_name]
            logger.info(f"Stopped MCP server: {server_name}")
    
    def disconnect_all(self):
        """Disconnect from all MCP servers"""
        for server_name in list(self.server_processes.keys()):
            self.stop_server(server_name)
    
    def load_tools_from_servers(self) -> List[MCPTool]:
        """Load tools from all running MCP servers"""
        tools = []
        
        # For now, we'll create mock tools based on known server capabilities
        # In a full implementation, this would query each server's capabilities
        
        for server_name, config in self.servers.items():
            if server_name == "filesystem":
                tools.extend([
                    MCPTool(
                        name="read_file",
                        description="Read contents of a file",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Path to the file"}
                            },
                            "required": ["path"]
                        }
                    ),
                    MCPTool(
                        name="write_file", 
                        description="Write content to a file",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Path to the file"},
                                "content": {"type": "string", "description": "Content to write"}
                            },
                            "required": ["path", "content"]
                        }
                    )
                ])
            elif server_name == "github":
                tools.extend([
                    MCPTool(
                        name="search_repositories",
                        description="Search GitHub repositories",
                        inputSchema={
                            "type": "object", 
                            "properties": {
                                "query": {"type": "string", "description": "Search query"}
                            },
                            "required": ["query"]
                        }
                    ),
                    MCPTool(
                        name="get_file_contents",
                        description="Get file contents from GitHub repository",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "owner": {"type": "string", "description": "Repository owner"},
                                "repo": {"type": "string", "description": "Repository name"},
                                "path": {"type": "string", "description": "File path"}
                            },
                            "required": ["owner", "repo", "path"]
                        }
                    )
                ])
            elif server_name == "firecrawl":
                tools.extend([
                    MCPTool(
                        name="scrape_url",
                        description="Scrape content from a URL using Firecrawl",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "URL to scrape"}
                            },
                            "required": ["url"]
                        }
                    )
                ])
        
        self.available_tools = tools
        return tools

# Global MCP client instance
mcp_client = MCPClient()

def get_available_mcp_tools():
    """Get available MCP tools as LangChain tools"""
    langchain_tools = []
    
    try:
        # Discover and start servers
        servers = mcp_client.discover_servers()
        
        if not servers:
            logger.info("No MCP servers found in environment configuration")
            return []
        
        logger.info(f"Discovered {len(servers)} potential MCP servers: {list(servers.keys())}")
        
        # Start servers (in a real implementation, this might be async)
        started_servers = []
        for server_name in servers.keys():
            success = mcp_client.start_server(server_name)
            if success:
                started_servers.append(server_name)
            else:
                logger.info(f"Could not start MCP server: {server_name}")
        
        if not started_servers:
            logger.info("No MCP servers could be started. This is normal if Node.js is not installed or API keys are not configured.")
            return []
        
        logger.info(f"Successfully started {len(started_servers)} MCP servers: {started_servers}")
        
        # Load tools from servers
        mcp_tools = mcp_client.load_tools_from_servers()
        
        if not mcp_tools:
            logger.info("No tools available from MCP servers")
            return []
        
        # Convert MCP tools to LangChain tools
        for mcp_tool in mcp_tools:
            def create_tool_wrapper(tool_name, tool_description):
                @tool
                def mcp_tool_wrapper(**kwargs):
                    """Execute MCP tool"""
                    # In a real implementation, this would communicate with the MCP server
                    # For now, return a mock response
                    return json.dumps({
                        "tool": tool_name,
                        "input": kwargs,
                        "result": f"Mock result from {tool_name}",
                        "status": "success"
                    })
                
                # Set the tool name and description manually
                mcp_tool_wrapper.name = tool_name
                mcp_tool_wrapper.description = tool_description
                mcp_tool_wrapper.__name__ = tool_name
                return mcp_tool_wrapper
            
            tool_wrapper = create_tool_wrapper(mcp_tool.name, mcp_tool.description)
            langchain_tools.append(tool_wrapper)
        
        logger.info(f"Successfully loaded {len(langchain_tools)} MCP tools from {len(started_servers)} servers")
        return langchain_tools
        
    except Exception as e:
        logger.error(f"Error initializing MCP tools: {str(e)}")
        logger.info("Continuing without MCP tools...")
        return []

def get_mcp_status():
    """Get status of MCP client and servers"""
    status = {
        "servers_discovered": len(mcp_client.servers),
        "servers_running": len(mcp_client.server_processes),
        "tools_available": len(mcp_client.available_tools),
        "servers": {}
    }
    
    for name, config in mcp_client.servers.items():
        status["servers"][name] = {
            "configured": True,
            "running": name in mcp_client.server_processes,
            "command": config.command,
            "description": getattr(config, 'description', 'No description')
        }
    
    return status 