"""
SQL Server MCP - AZURE WEB APP PRODUCTION VERSION
Hardcoded configuration for easy deployment
"""

import os
import sys
import logging
import json
from typing import Any, Dict, List, Optional
import pyodbc
from datetime import datetime, date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# ============================================================================
# HARDCODED CONFIGURATION FOR AZURE DEPLOYMENT
# ============================================================================

# Database Configuration - Hardcoded
DB_SERVER = "sqltpt.database.windows.net"
DB_DATABASE = "NorthWinds Updated"
DB_USERNAME = "sqladmin"
DB_PASSWORD = "Wind0wsazure@123"

# Azure Web App Configuration
# Use Azure's PORT environment variable if available, otherwise default to 8000
PORT = int(os.environ.get('PORT', 8000))
HOST = "0.0.0.0"

# MCP Server Configuration
MCP_SERVER_NAME = "SQL Server MCP"
MCP_SERVER_VERSION = "1.0.0"
MCP_DESCRIPTION = "Model Context Protocol server for MS SQL Server database operations"

# Security Settings
QUERY_TIMEOUT = 30
MAX_ROWS_RETURNED = 1000
ENABLE_WRITE_OPERATIONS = False

# Logging Configuration
LOGGING_LEVEL = "INFO"
LOG_SQL_QUERIES = True

# CORS Settings - Open for all origins (adjust for production security)
CORS_ORIGINS = ["*"]
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Database connection and operations manager with hardcoded Azure SQL config"""
    
    def __init__(self):
        self.connection_string = self._build_connection_string()
        logger.info("Database manager initialized with Azure SQL Database")
    
    def _build_connection_string(self) -> str:
        """Build SQL Server connection string for Azure SQL Database"""
        # Use ODBC Driver 18 for Azure Web Apps (Linux)
        driver = "{ODBC Driver 18 for SQL Server}"
        
        conn_str = (
            f"Driver={driver};"
            f"Server=tcp:{DB_SERVER},1433;"
            f"Database={DB_DATABASE};"
            f"Uid={DB_USERNAME};"
            f"Pwd={DB_PASSWORD};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout={QUERY_TIMEOUT};"
        )
        
        logger.info(f"Connection string built for server: {DB_SERVER}")
        return conn_str
    
    def get_connection(self):
        """Get database connection"""
        try:
            conn = pyodbc.connect(self.connection_string)
            logger.debug("Database connection established successfully")
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise HTTPException(500, f"Database connection failed: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query with Azure SQL optimizations"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                start_time = datetime.now()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                columns = [column[0] for column in cursor.description] if cursor.description else []
                
                rows = []
                for row in cursor.fetchall():
                    row_dict = {}
                    for i, value in enumerate(row):
                        if isinstance(value, (datetime, date)):
                            row_dict[columns[i]] = value.isoformat()
                        elif isinstance(value, bytes):
                            # Handle binary data
                            row_dict[columns[i]] = f"<binary data: {len(value)} bytes>"
                        else:
                            row_dict[columns[i]] = value
                    rows.append(row_dict)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                if LOG_SQL_QUERIES:
                    logger.info(f"Query executed in {execution_time:.3f}s, returned {len(rows)} rows")
                
                return rows
                
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise HTTPException(400, f"Query execution failed: {str(e)}")

# Initialize database manager
db_manager = DatabaseManager()

# ============================================================================
# MCP TOOLS FUNCTIONS
# ============================================================================

def query_database_tool(query: str, limit: int = 100) -> Dict[str, Any]:
    """Execute a SQL query against the database - Azure optimized"""
    query_upper = query.strip().upper()
    
    # Security: Only allow SELECT statements
    if not query_upper.startswith('SELECT'):
        return {
            "error": "Only SELECT queries are allowed for security reasons",
            "query": query,
            "server": DB_SERVER
        }
    
    # Apply row limits for Azure SQL Database efficiency
    effective_limit = min(limit, MAX_ROWS_RETURNED)
    
    if 'LIMIT' not in query_upper and 'TOP' not in query_upper:
        query = query.strip()
        if query_upper.startswith('SELECT DISTINCT'):
            query = f"SELECT DISTINCT TOP {effective_limit} " + query[15:]
        else:
            query = f"SELECT TOP {effective_limit} " + query[6:]
    
    try:
        results = db_manager.execute_query(query)
        return {
            "success": True,
            "results": results,
            "row_count": len(results),
            "query": query,
            "server": DB_SERVER,
            "database": DB_DATABASE,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "server": DB_SERVER,
            "timestamp": datetime.now().isoformat()
        }

def list_tables_tool() -> Dict[str, Any]:
    """Get a list of all tables in the Azure SQL Database"""
    try:
        query = """
        SELECT 
            TABLE_SCHEMA,
            TABLE_NAME,
            TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
        results = db_manager.execute_query(query)
        
        # Format results for better readability
        tables = []
        for row in results:
            if row['TABLE_SCHEMA'] == 'dbo':
                tables.append(row['TABLE_NAME'])
            else:
                tables.append(f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}")
        
        return {
            "success": True,
            "tables": tables,
            "table_count": len(tables),
            "server": DB_SERVER,
            "database": DB_DATABASE,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "server": DB_SERVER,
            "timestamp": datetime.now().isoformat()
        }

def describe_table_tool(table_name: str) -> Dict[str, Any]:
    """Get schema information for a specific table in Azure SQL Database"""
    try:
        query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION,
            NUMERIC_SCALE,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
        """
        schema = db_manager.execute_query(query, (table_name,))
        
        return {
            "success": True,
            "table_name": table_name,
            "columns": schema,
            "column_count": len(schema),
            "server": DB_SERVER,
            "database": DB_DATABASE,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "table_name": table_name,
            "server": DB_SERVER,
            "timestamp": datetime.now().isoformat()
        }

def get_database_info_tool() -> Dict[str, Any]:
    """Get comprehensive information about the Azure SQL Database"""
    try:
        # Get database version and info
        queries = {
            "version": "SELECT @@VERSION as version",
            "database_name": "SELECT DB_NAME() as database_name",
            "server_name": "SELECT @@SERVERNAME as server_name",
            "current_user": "SELECT CURRENT_USER as current_user",
            "database_size": """
                SELECT 
                    SUM(CAST(FILEPROPERTY(name, 'SpaceUsed') AS bigint) * 8192.) / 1024 / 1024 as size_mb
                FROM sys.database_files 
                WHERE type in (0,1)
            """
        }
        
        info = {}
        for key, query in queries.items():
            try:
                result = db_manager.execute_query(query)
                info[key] = result[0][list(result[0].keys())[0]] if result else "Unknown"
            except:
                info[key] = "Not available"
        
        # Get table count
        table_result = db_manager.execute_query("""
            SELECT COUNT(*) as table_count 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
        """)
        
        return {
            "success": True,
            "server": DB_SERVER,
            "database": DB_DATABASE,
            "database_name": info.get("database_name", DB_DATABASE),
            "server_name": info.get("server_name", DB_SERVER),
            "version": info.get("version", "Azure SQL Database"),
            "current_user": info.get("current_user", DB_USERNAME),
            "database_size_mb": info.get("database_size", "Unknown"),
            "table_count": table_result[0]["table_count"] if table_result else 0,
            "connection_info": {
                "server": DB_SERVER,
                "database": DB_DATABASE,
                "username": DB_USERNAME,
                "encrypted": True,
                "azure_sql": True
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "server": DB_SERVER,
            "database": DB_DATABASE,
            "timestamp": datetime.now().isoformat()
        }

def execute_stored_procedure_tool(procedure_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a stored procedure in Azure SQL Database"""
    if not ENABLE_WRITE_OPERATIONS:
        return {
            "success": False,
            "error": "Stored procedure execution is disabled for security",
            "procedure_name": procedure_name,
            "server": DB_SERVER
        }
    
    try:
        if parameters:
            param_list = []
            param_values = []
            for key, value in parameters.items():
                param_list.append(f"@{key} = ?")
                param_values.append(value)
            
            query = f"EXEC {procedure_name} {', '.join(param_list)}"
            results = db_manager.execute_query(query, tuple(param_values))
        else:
            query = f"EXEC {procedure_name}"
            results = db_manager.execute_query(query)
        
        return {
            "success": True,
            "procedure_name": procedure_name,
            "parameters": parameters,
            "results": results,
            "row_count": len(results),
            "server": DB_SERVER,
            "database": DB_DATABASE,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "procedure_name": procedure_name,
            "parameters": parameters,
            "server": DB_SERVER,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# MCP TOOLS REGISTRY
# ============================================================================

MCP_TOOLS = {
    "query_database": {
        "function": query_database_tool,
        "description": "Execute a SQL SELECT query against the Azure SQL Database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "SQL SELECT query to execute (SELECT statements only)"
                },
                "limit": {
                    "type": "integer", 
                    "description": f"Maximum rows to return (default: 100, max: {MAX_ROWS_RETURNED})", 
                    "default": 100,
                    "maximum": MAX_ROWS_RETURNED
                }
            },
            "required": ["query"]
        }
    },
    "list_tables": {
        "function": list_tables_tool,
        "description": "Get a list of all tables in the Azure SQL Database.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    "describe_table": {
        "function": describe_table_tool,
        "description": "Get detailed schema information for a specific table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string", 
                    "description": "Name of the table to describe"
                }
            },
            "required": ["table_name"]
        }
    },
    "get_database_info": {
        "function": get_database_info_tool,
        "description": "Get comprehensive information about the Azure SQL Database connection and server.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    "execute_stored_procedure": {
        "function": execute_stored_procedure_tool,
        "description": "Execute a stored procedure (if enabled for security).",
        "parameters": {
            "type": "object",
            "properties": {
                "procedure_name": {
                    "type": "string", 
                    "description": "Name of the stored procedure to execute"
                },
                "parameters": {
                    "type": "object", 
                    "description": "Parameters for the stored procedure (optional)"
                }
            },
            "required": ["procedure_name"]
        }
    }
}

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Create FastAPI app
app = FastAPI(
    title=MCP_SERVER_NAME,
    description=MCP_DESCRIPTION,
    version=MCP_SERVER_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_CREDENTIALS,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# ============================================================================
# CUSTOM ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - redirects to test page"""
    return {
        "name": MCP_SERVER_NAME,
        "version": MCP_SERVER_VERSION,
        "description": MCP_DESCRIPTION,
        "endpoints": {
            "health": "/health",
            "test": "/test", 
            "mcp": "/mcp",
            "capabilities": "/mcp/capabilities",
            "debug": "/debug/tools",
            "docs": "/docs"
        },
        "server_info": {
            "database_server": DB_SERVER,
            "database_name": DB_DATABASE,
            "total_tools": len(MCP_TOOLS)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Azure App Service"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 as health_check")
            result = cursor.fetchone()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "server": DB_SERVER,
            "database_name": DB_DATABASE,
            "mcp_tools": len(MCP_TOOLS),
            "version": MCP_SERVER_VERSION
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "timestamp": datetime.now().isoformat(),
            "database": "disconnected",
            "server": DB_SERVER,
            "error": str(e),
            "version": MCP_SERVER_VERSION
        }

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """Test page showing server status and available endpoints"""
    # Check database connectivity
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as table_count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
            table_count = cursor.fetchone()[0]
        db_status = f"✅ Connected ({table_count} tables)"
    except Exception as e:
        db_status = f"❌ Connection failed: {str(e)}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{MCP_SERVER_NAME} - Test Page</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .content {{ padding: 30px; }}
            .endpoint {{ 
                background: #f8f9fa; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 8px; 
                border-left: 4px solid #3498db;
                font-family: monospace;
            }}
            .status {{ 
                padding: 8px 16px; 
                border-radius: 20px; 
                color: white; 
                font-weight: bold;
                display: inline-block;
                margin: 5px 0;
            }}
            .success {{ background: #27ae60; }}
            .error {{ background: #e74c3c; }}
            .info {{ background: #3498db; }}
            .tool {{ 
                background: #ecf0f1; 
                padding: 12px; 
                margin: 8px 0; 
                border-radius: 6px;
                border-left: 3px solid #2ecc71;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }}
            .card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}
            @media (max-width: 768px) {{
                .grid {{ grid-template-columns: 1fr; }}
                .container {{ margin: 10px; }}
                body {{ padding: 10px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 {MCP_SERVER_NAME}</h1>
                <p>{MCP_DESCRIPTION}</p>
                <div class="status success">RUNNING</div>
            </div>
            
            <div class="content">
                <div class="grid">
                    <div class="card">
                        <h2>🔗 Connection Status</h2>
                        <p><strong>Server:</strong> {DB_SERVER}</p>
                        <p><strong>Database:</strong> {DB_DATABASE}</p>
                        <p><strong>Status:</strong> {db_status}</p>
                        <p><strong>Version:</strong> {MCP_SERVER_VERSION}</p>
                    </div>
                    
                    <div class="card">
                        <h2>📊 Server Info</h2>
                        <p><strong>Host:</strong> {HOST}:{PORT}</p>
                        <p><strong>Tools:</strong> {len(MCP_TOOLS)} MCP tools available</p>
                        <p><strong>Security:</strong> SELECT queries only</p>
                        <p><strong>Max Rows:</strong> {MAX_ROWS_RETURNED}</p>
                    </div>
                </div>
                
                <h2>🌐 Available Endpoints</h2>
                <div class="endpoint"><strong>GET /</strong> - Root endpoint (this page info)</div>
                <div class="endpoint"><strong>GET /health</strong> - Health check for Azure monitoring</div>
                <div class="endpoint"><strong>GET /test</strong> - This test page</div>
                <div class="endpoint"><strong>GET /debug/tools</strong> - List all MCP tools</div>
                <div class="endpoint"><strong>POST /mcp</strong> - Main MCP protocol endpoint</div>
                <div class="endpoint"><strong>GET /mcp/capabilities</strong> - MCP server capabilities</div>
                <div class="endpoint"><strong>GET /docs</strong> - FastAPI documentation</div>
                
                <h2>🛠️ Available MCP Tools</h2>
    """
    
    # Add tools dynamically
    for name, config in MCP_TOOLS.items():
        html_content += f"""
                <div class="tool">
                    <strong>{name}</strong> - {config["description"]}
                </div>
        """
    
    html_content += f"""
                
                <h2>🎯 Copilot Studio Integration</h2>
                <div class="card">
                    <p><strong>MCP Server URL:</strong> <code>https://your-webapp-url.azurewebsites.net/mcp</code></p>
                    <p><strong>Protocol:</strong> JSON-RPC 2.0 over HTTPS</p>
                    <p><strong>Transport:</strong> HTTP POST</p>
                    <p><strong>Tools Available:</strong> {len(MCP_TOOLS)} database operations</p>
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background: #e8f5e8; border-radius: 8px; border-left: 4px solid #27ae60;">
                    <h3>✅ Ready for Production!</h3>
                    <p>Your MCP server is configured and ready for Copilot Studio integration. Use the MCP endpoint URL above in your custom connector configuration.</p>
                </div>
                
                <footer style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 8px; text-align: center; color: #666;">
                    <small>
                        Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}<br>
                        Powered by FastAPI • Model Context Protocol • Azure SQL Database
                    </small>
                </footer>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/mcp/capabilities")
async def mcp_capabilities():
    """Get MCP server capabilities - standard MCP endpoint"""
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {
                "listChanged": True
            },
            "resources": {},
            "prompts": {}
        },
        "serverInfo": {
            "name": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
            "description": MCP_DESCRIPTION
        }
    }

@app.get("/debug/tools")
async def list_available_tools():
    """Debug endpoint to list all MCP tools with full details"""
    tools = []
    
    for name, config in MCP_TOOLS.items():
        tools.append({
            "name": name,
            "description": config["description"],
            "parameters": config["parameters"],
            "enabled": True
        })
    
    return {
        "server_info": {
            "name": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
            "description": MCP_DESCRIPTION,
            "total_tools": len(tools),
            "database": {
                "server": DB_SERVER,
                "database": DB_DATABASE,
                "username": DB_USERNAME,
                "azure_sql": True
            }
        },
        "available_tools": tools,
        "configuration": {
            "max_rows": MAX_ROWS_RETURNED,
            "query_timeout": QUERY_TIMEOUT,
            "write_operations": ENABLE_WRITE_OPERATIONS,
            "log_queries": LOG_SQL_QUERIES
        },
        "timestamp": datetime.now().isoformat(),
        "status": "All tools registered and ready for Copilot Studio!"
    }

# ============================================================================
# MCP PROTOCOL HANDLER
# ============================================================================

@app.post("/mcp")
async def mcp_handler(request: dict):
    """Handle MCP JSON-RPC requests - Main integration endpoint"""
    try:
        logger.debug(f"Received MCP request: {request.get('method', 'unknown')}")
        
        # Handle initialize request
        if request.get("method") == "initialize":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "listChanged": True
                        },
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": {
                        "name": MCP_SERVER_NAME,
                        "version": MCP_SERVER_VERSION,
                        "description": MCP_DESCRIPTION
                    }
                },
                "id": request.get("id", 1)
            }
        
        # Handle tools/list request
        elif request.get("method") == "tools/list":
            tools = []
            for name, config in MCP_TOOLS.items():
                tools.append({
                    "name": name,
                    "description": config["description"],
                    "inputSchema": config["parameters"]
                })
            
            return {
                "jsonrpc": "2.0",
                "result": {
                    "tools": tools
                },
                "id": request.get("id", 1)
            }
        
        # Handle tools/call request
        elif request.get("method") == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name not in MCP_TOOLS:
                logger.warning(f"Tool '{tool_name}' not found")
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found. Available tools: {', '.join(MCP_TOOLS.keys())}"
                    },
                    "id": request.get("id", 1)
                }
            
            # Execute the tool
            tool_function = MCP_TOOLS[tool_name]["function"]
            try:
                logger.info(f"Executing tool: {tool_name} with args: {arguments}")
                result = tool_function(**arguments)
                
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2, ensure_ascii=False)
                            }
                        ],
                        "isError": not result.get("success", True)
                    },
                    "id": request.get("id", 1)
                }
                
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_name}: {e}")
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "success": False,
                                    "error": str(e),
                                    "tool": tool_name,
                                    "timestamp": datetime.now().isoformat()
                                }, indent=2)
                            }
                        ],
                        "isError": True
                    },
                    "id": request.get("id", 1)
                }
        
        else:
            logger.warning(f"Unknown method: {request.get('method')}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method '{request.get('method')}' not found"
                },
                "id": request.get("id", 1)
            }
            
    except Exception as e:
        logger.error(f"MCP handler error: {e}")
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": f"Internal server error: {str(e)}"
            },
            "id": request.get("id", 1) if isinstance(request, dict) else 1
        }

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("="*50)
    logger.info(f"🚀 {MCP_SERVER_NAME} v{MCP_SERVER_VERSION} Starting...")
    logger.info(f"📊 Database: {DB_SERVER}/{DB_DATABASE}")
    logger.info(f"🛠️  MCP Tools: {len(MCP_TOOLS)} registered")
    logger.info(f"🌐 Server: {HOST}:{PORT}")
    logger.info(f"🔒 Security: SELECT queries only, max {MAX_ROWS_RETURNED} rows")
    logger.info("="*50)
    
    # Test database connection on startup
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
            table_count = cursor.fetchone()[0]
            logger.info(f"✅ Database connected successfully - {table_count} tables available")
    except Exception as e:
        logger.error(f"❌ Database connection failed on startup: {e}")

if __name__ == "__main__":
    # For local development
    logger.info(f"Starting {MCP_SERVER_NAME} locally on {HOST}:{PORT}")
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT, 
        log_level="info",
        access_log=True
    )
