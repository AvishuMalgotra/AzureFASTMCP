"""
SQL Server MCP - AZURE PRODUCTION VERSION with pymssql
Using original database settings
"""

import os
import sys
import logging
import json
from typing import Any, Dict, List, Optional
import pymssql
from datetime import datetime, date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# ============================================================================
# ORIGINAL DATABASE SETTINGS - HARDCODED FOR AZURE
# ============================================================================

# Your Original Database Configuration
DB_SERVER = "sqltpt.database.windows.net"
DB_DATABASE = "NorthWinds Updated" 
DB_USERNAME = "sqladmin"
DB_PASSWORD = "Wind0wsazure@123"

# Azure Configuration
PORT = int(os.environ.get('PORT', 8000))
HOST = "0.0.0.0"

# MCP Server Configuration-m
MCP_SERVER_NAME = "SQL Server MCP"
MCP_SERVER_VERSION = "1.0.0"
MCP_DESCRIPTION = "Model Context Protocol server for MS SQL Server with pymssql"

# Settings
QUERY_TIMEOUT = 30
MAX_ROWS_RETURNED = 1000
LOG_SQL_QUERIES = True

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE MANAGER WITH PYMSSQL
# ============================================================================

class DatabaseManager:
    """Database manager using pymssql for Azure compatibility"""
    
    def __init__(self):
        self.server = DB_SERVER
        self.database = DB_DATABASE  
        self.username = DB_USERNAME
        self.password = DB_PASSWORD
        logger.info(f"Database manager initialized for {self.server}/{self.database}")
    
    def get_connection(self):
        """Get database connection using pymssql"""
        try:
            conn = pymssql.connect(
                server=self.server,
                user=self.username,
                password=self.password,
                database=self.database,
                timeout=QUERY_TIMEOUT
            )
            logger.debug("Database connection established with pymssql")
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise HTTPException(500, f"Database connection failed: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query with pymssql"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(as_dict=True) as cursor:
                    start_time = datetime.now()
                    
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    
                    results = cursor.fetchall()
                    
                    # Format results properly
                    formatted_results = []
                    for row in results:
                        formatted_row = {}
                        for key, value in row.items():
                            if isinstance(value, (datetime, date)):
                                formatted_row[key] = value.isoformat()
                            elif isinstance(value, bytes):
                                formatted_row[key] = f"<binary data: {len(value)} bytes>"
                            else:
                                formatted_row[key] = value
                        formatted_results.append(formatted_row)
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    if LOG_SQL_QUERIES:
                        logger.info(f"Query executed in {execution_time:.3f}s, returned {len(formatted_results)} rows")
                    
                    return formatted_results
                    
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise HTTPException(400, f"Query execution failed: {str(e)}")

# Initialize database manager
db_manager = DatabaseManager()

# ============================================================================
# MCP TOOLS FUNCTIONS
# ============================================================================

def query_database_tool(query: str, limit: int = 100) -> Dict[str, Any]:
    """Execute SQL query against database"""
    query_upper = query.strip().upper()
    
    if not query_upper.startswith('SELECT'):
        return {
            "success": False,
            "error": "Only SELECT queries are allowed for security reasons",
            "query": query,
            "server": DB_SERVER,
            "timestamp": datetime.now().isoformat()
        }
    
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
            "connection_type": "pymssql",
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
    """Get list of all tables in database"""
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
            "connection_type": "pymssql",
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
    """Get schema information for specific table"""
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
        WHERE TABLE_NAME = %s
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
            "connection_type": "pymssql",
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
    """Get comprehensive database information"""
    try:
        queries = {
            "version": "SELECT @@VERSION as version",
            "database_name": "SELECT DB_NAME() as database_name",
            "server_name": "SELECT @@SERVERNAME as server_name",
            "current_user": "SELECT CURRENT_USER as current_user"
        }
        
        info = {}
        for key, query in queries.items():
            try:
                result = db_manager.execute_query(query)
                info[key] = result[0][list(result[0].keys())[0]] if result else "Unknown"
            except:
                info[key] = "Not available"
        
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
            "table_count": table_result[0]["table_count"] if table_result else 0,
            "connection_info": {
                "server": DB_SERVER,
                "database": DB_DATABASE,
                "username": DB_USERNAME,
                "connection_type": "pymssql",
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
    """Execute stored procedure"""
    try:
        if parameters:
            param_list = []
            param_values = []
            for key, value in parameters.items():
                param_list.append(f"@{key} = %s")
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
            "connection_type": "pymssql",
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
        "description": "Execute a SQL SELECT query against the database. Returns structured data with metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "SQL SELECT query to execute (SELECT statements only for security)"
                },
                "limit": {
                    "type": "integer", 
                    "description": f"Maximum rows to return (default: 100, max: {MAX_ROWS_RETURNED})", 
                    "default": 100,
                    "minimum": 1,
                    "maximum": MAX_ROWS_RETURNED
                }
            },
            "required": ["query"]
        }
    },
    "list_tables": {
        "function": list_tables_tool,
        "description": "Get a comprehensive list of all tables in the database.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    "describe_table": {
        "function": describe_table_tool,
        "description": "Get detailed schema information for a specific table including columns, data types, and constraints.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string", 
                    "description": "Name of the table to describe (case-sensitive)"
                }
            },
            "required": ["table_name"]
        }
    },
    "get_database_info": {
        "function": get_database_info_tool,
        "description": "Get comprehensive information about the database connection, server details, and database statistics.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    "execute_stored_procedure": {
        "function": execute_stored_procedure_tool,
        "description": "Execute a stored procedure in the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "procedure_name": {
                    "type": "string", 
                    "description": "Name of the stored procedure to execute"
                },
                "parameters": {
                    "type": "object", 
                    "description": "Parameters for the stored procedure as key-value pairs (optional)"
                }
            },
            "required": ["procedure_name"]
        }
    }
}

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title=MCP_SERVER_NAME,
    description=MCP_DESCRIPTION,
    version=MCP_SERVER_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "name": MCP_SERVER_NAME,
        "version": MCP_SERVER_VERSION,
        "description": MCP_DESCRIPTION,
        "status": "operational",
        "connection_type": "pymssql",
        "endpoints": {
            "health": "/health",
            "test": "/test", 
            "mcp_protocol": "/mcp",
            "mcp_capabilities": "/mcp/capabilities",
            "debug_tools": "/debug/tools",
            "api_docs": "/docs"
        },
        "server_info": {
            "host": HOST,
            "port": PORT,
            "database_server": DB_SERVER,
            "database_name": DB_DATABASE,
            "total_mcp_tools": len(MCP_TOOLS)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Azure monitoring"""
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 as health_check, GETDATE() as server_time")
                result = cursor.fetchone()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": {
                "status": "connected",
                "server": DB_SERVER,
                "database": DB_DATABASE,
                "connection_type": "pymssql",
                "server_time": result[1].isoformat() if result else None
            },
            "application": {
                "name": MCP_SERVER_NAME,
                "version": MCP_SERVER_VERSION,
                "tools_count": len(MCP_TOOLS)
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "timestamp": datetime.now().isoformat(),
            "database": {
                "status": "disconnected",
                "server": DB_SERVER,
                "error": str(e)
            }
        }

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """Test page showing server status"""
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor(as_dict=True) as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as table_count,
                        DB_NAME() as database_name,
                        @@SERVERNAME as server_name
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_TYPE = 'BASE TABLE'
                """)
                db_info = cursor.fetchone()
        db_status = f"✅ Connected via pymssql - {db_info['table_count']} tables available"
    except Exception as e:
        db_status = f"❌ Connection failed: {str(e)}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{MCP_SERVER_NAME} - Test Page</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            .status {{ padding: 8px 16px; border-radius: 20px; color: white; font-weight: bold; background: #27ae60; }}
            .endpoint {{ background: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 {MCP_SERVER_NAME}</h1>
            <p>Using <strong>pymssql</strong> for Azure compatibility</p>
            <div class="status">OPERATIONAL</div>
            
            <h2>Database Status</h2>
            <p><strong>Server:</strong> {DB_SERVER}</p>
            <p><strong>Database:</strong> {DB_DATABASE}</p>
            <p><strong>Status:</strong> {db_status}</p>
            <p><strong>Connection:</strong> pymssql (Azure compatible)</p>
            
            <h2>Available Endpoints</h2>
            <div class="endpoint"><strong>GET /health</strong> - Health check</div>
            <div class="endpoint"><strong>POST /mcp</strong> - MCP protocol endpoint</div>
            <div class="endpoint"><strong>GET /debug/tools</strong> - List MCP tools</div>
            <div class="endpoint"><strong>GET /docs</strong> - API documentation</div>
            
            <h2>MCP Tools ({len(MCP_TOOLS)})</h2>
            <ul>
                <li>query_database - Execute SQL queries</li>
                <li>list_tables - List database tables</li>
                <li>describe_table - Get table schema</li>
                <li>get_database_info - Database information</li>
                <li>execute_stored_procedure - Execute procedures</li>
            </ul>
            
            <p><strong>Ready for Copilot Studio integration!</strong></p>
            <p><small>Timestamp: {datetime.now().isoformat()}</small></p>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/mcp/capabilities")
async def mcp_capabilities():
    """MCP server capabilities endpoint"""
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
    """Debug endpoint with MCP tools information"""
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
            "total_tools": len(tools),
            "connection_type": "pymssql",
            "database": {
                "server": DB_SERVER,
                "database": DB_DATABASE,
                "username": DB_USERNAME
            }
        },
        "available_tools": tools,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# MCP PROTOCOL HANDLER
# ============================================================================

@app.post("/mcp")
async def mcp_handler(request: dict):
    """Main MCP JSON-RPC protocol handler"""
    try:
        method = request.get("method", "unknown")
        request_id = request.get("id", 1)
        
        logger.info(f"MCP request received: {method}")
        
        if method == "initialize":
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
                "id": request_id
            }
        
        elif method == "tools/list":
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
                "id": request_id
            }
        
        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name not in MCP_TOOLS:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found. Available: {', '.join(MCP_TOOLS.keys())}"
                    },
                    "id": request_id
                }
            
            tool_function = MCP_TOOLS[tool_name]["function"]
            try:
                logger.info(f"Executing tool: {tool_name}")
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
                    "id": request_id
                }
                
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
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
                                    "connection_type": "pymssql",
                                    "timestamp": datetime.now().isoformat()
                                }, indent=2)
                            }
                        ],
                        "isError": True
                    },
                    "id": request_id
                }
        
        else:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found"
                },
                "id": request_id
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
    logger.info("=" * 50)
    logger.info(f"🚀 {MCP_SERVER_NAME} v{MCP_SERVER_VERSION}")
    logger.info(f"🔗 Connection: pymssql (Azure compatible)")
    logger.info(f"📊 Database: {DB_SERVER}/{DB_DATABASE}")
    logger.info(f"🛠️ MCP Tools: {len(MCP_TOOLS)} registered")
    logger.info(f"🌐 Server: {HOST}:{PORT}")
    logger.info("=" * 50)
    
    # Test database connection on startup
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor(as_dict=True) as cursor:
                cursor.execute("SELECT COUNT(*) as table_count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
                result = cursor.fetchone()
                logger.info(f"✅ Database connected via pymssql - {result['table_count']} tables available")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
