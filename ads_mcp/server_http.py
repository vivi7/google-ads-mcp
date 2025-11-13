# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP/SSE transport for the MCP server."""

import asyncio
import json
import os
from typing import AsyncGenerator, Dict, Any
from collections import deque

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn

from ads_mcp.coordinator import mcp

# The following imports are necessary to register the tools with the `mcp`
# object, even though they are not directly used in this file.
# The `# noqa: F401` comment tells the linter to ignore the "unused import"
# warning.
from ads_mcp.tools import search, core  # noqa: F401

app = FastAPI(title="Google Ads MCP Server")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active SSE connections and message queues
sse_connections: Dict[str, deque] = {}


async def process_mcp_request(request_data: dict) -> dict:
    """Process an MCP request through the FastMCP server."""
    try:
        method = request_data.get("method")
        request_id = request_data.get("id")

        # Handle tool calls
        if method == "tools/call":
            params = request_data.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            # Call the tool directly from FastMCP's registry
            # FastMCP stores tools in _tools attribute
            if hasattr(mcp, "_tools") and tool_name in mcp._tools:
                tool_func = mcp._tools[tool_name]
                # Call the tool function with the arguments
                result = tool_func(**arguments)

                # Format the result according to MCP protocol
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, default=str)
                            }
                        ]
                    }
                }
            else:
                # Try to import and call the tool functions directly
                if tool_name == "search":
                    from ads_mcp.tools.search import search
                    result = search(**arguments)
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, default=str)
                                }
                            ]
                        }
                    }
                elif tool_name == "list_accessible_customers":
                    from ads_mcp.tools.core import list_accessible_customers
                    result = list_accessible_customers()
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, default=str)
                                }
                            ]
                        }
                    }
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Tool '{tool_name}' not found"
                        }
                    }

        # Handle initialize request
        elif method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "google-ads-mcp",
                        "version": "0.0.1"
                    }
                }
            }

        # Handle tools/list request
        elif method == "tools/list":
            # Get tools from FastMCP
            tools_list = []
            if hasattr(mcp, "_tools"):
                for tool_name, tool_info in mcp._tools.items():
                    tools_list.append({
                        "name": tool_name,
                        "description": getattr(tool_info, "__doc__", f"Tool: {tool_name}")
                    })
            else:
                # Fallback to known tools
                tools_list = [
                    {
                        "name": "search",
                        "description": "Fetches data from the Google Ads API using the search method"
                    },
                    {
                        "name": "list_accessible_customers",
                        "description": "Returns ids of customers directly accessible by the user authenticating the call."
                    }
                ]

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": tools_list
                }
            }

        # Handle ping request
        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {}
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found"
                }
            }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_data.get("id"),
            "error": {
                "code": -32000,
                "message": str(e)
            }
        }


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE endpoint for MCP protocol - server sends messages to client."""
    connection_id = str(id(request))
    message_queue = deque()
    sse_connections[connection_id] = message_queue

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'jsonrpc': '2.0', 'method': 'connection', 'params': {'status': 'connected', 'connectionId': connection_id}})}\n\n"

            # Keep connection alive and process messages from queue
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                # Process messages from queue
                if message_queue:
                    message = message_queue.popleft()
                    yield f"data: {json.dumps(message)}\n\n"
                else:
                    # Send keepalive
                    await asyncio.sleep(1)
                    yield ": keepalive\n\n"
        finally:
            # Clean up connection
            if connection_id in sse_connections:
                del sse_connections[connection_id]

    return EventSourceResponse(event_generator())


@app.post("/message")
async def post_message(request: Request):
    """HTTP POST endpoint for sending MCP messages from client to server."""
    try:
        body = await request.json()

        # Validate JSON-RPC format
        if not isinstance(body, dict) or "jsonrpc" not in body:
            raise HTTPException(status_code=400, detail="Invalid JSON-RPC format")

        # Process the MCP message
        response = await process_mcp_request(body)

        return response
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": body.get("id") if isinstance(body, dict) else None,
            "error": {
                "code": -32000,
                "message": str(e)
            }
        }


@app.post("/sse-message")
async def post_sse_message(request: Request):
    """HTTP POST endpoint for sending messages that will be delivered via SSE."""
    try:
        body = await request.json()
        connection_id = request.headers.get("X-Connection-ID")

        if connection_id and connection_id in sse_connections:
            # Process the message and queue the response
            response = await process_mcp_request(body)
            sse_connections[connection_id].append(response)
            return {"status": "queued"}
        else:
            # Process directly if no SSE connection
            return await process_mcp_request(body)
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": body.get("id") if isinstance(body, dict) else None,
            "error": {
                "code": -32000,
                "message": str(e)
            }
        }


@app.get("/")
async def root():
    """Root endpoint - redirects to health check."""
    return {
        "status": "healthy",
        "service": "google-ads-mcp",
        "transport": "sse/http",
        "endpoints": {
            "health": "/health",
            "sse": "/sse",
            "message": "/message",
            "sse_message": "/sse-message"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "google-ads-mcp",
        "transport": "sse/http"
    }


def run_http_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the MCP server in HTTP/SSE mode."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    run_http_server(host=host, port=port)

