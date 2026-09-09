"""Conversation-addressed APIs with workspace and terminal-history context.

Local processes and desktop/VSCode services still share the host; these path
checks are routing safeguards, not a sandbox for arbitrary shell commands.
"""

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.routing import APIRoute

from openhands.agent_server.bash_router import bash_router
from openhands.agent_server.config import Config
from openhands.agent_server.dependencies import (
    check_session_api_key,
    get_conversation_service,
    get_event_service,
)
from openhands.agent_server.desktop_router import desktop_router
from openhands.agent_server.docker_runtime.proxy import proxy_http
from openhands.agent_server.docker_runtime.routers import (
    _build_upstream_path,
    _workspace_or_404,
    get_registry,
)
from openhands.agent_server.file_router import file_router
from openhands.agent_server.git_router import git_router
from openhands.agent_server.init_router import require_initialized
from openhands.agent_server.mcp_router import MCPTestResponse, test_mcp_server
from openhands.agent_server.vscode_router import (
    VSCodeUrlResponse,
    get_vscode_url,
    vscode_router,
)


async def require_local_runtime(
    runtime_conversation_id: UUID, request: Request
) -> None:
    service = get_conversation_service(request)
    event_service = await get_event_service(runtime_conversation_id, service)
    request.state.runtime_event_service = event_service
    root = Path(event_service.get_conversation().workspace.working_dir).resolve()
    for name in ("path", "workspace_dir"):
        path = request.path_params.get(name) or request.query_params.get(name)
        if path is not None and (
            not Path(path).is_absolute()
            or not Path(path).resolve().is_relative_to(root)
        ):
            raise HTTPException(
                422, f"{name} must be inside the conversation workspace"
            )
    trajectory_id = request.path_params.get("conversation_id")
    if trajectory_id is not None:
        try:
            matches = UUID(trajectory_id) == runtime_conversation_id
        except ValueError as exc:
            raise HTTPException(422, "Invalid trajectory conversation id") from exc
        if not matches:
            raise HTTPException(
                422, "Trajectory must belong to the selected conversation"
            )


class ConversationRuntimeRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        local_handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            config: Config = request.app.state.config
            if config.conversation_runtime == "docker":
                check_session_api_key(request, request.headers.get("x-session-api-key"))
                require_initialized(request)
                try:
                    conversation_id = UUID(
                        request.path_params["runtime_conversation_id"]
                    )
                except ValueError as exc:
                    raise HTTPException(422, "Invalid conversation id") from exc
                workspace = await _workspace_or_404(
                    get_registry(request), conversation_id
                )
                if not workspace.scoped_runtime_verified:
                    await _require_scoped_runtime_image(
                        workspace.host, workspace.api_key
                    )
                    workspace.scoped_runtime_verified = True
                return await proxy_http(
                    request,
                    workspace,
                    upstream_path=_build_upstream_path(request, request.url.path),
                )
            return await local_handler(request)

        return handle


class RuntimeRouter(APIRouter):
    def add_api_route(
        self, path: str, endpoint: Callable[..., Any], **kwargs: Any
    ) -> None:
        kwargs["route_class_override"] = ConversationRuntimeRoute
        super().add_api_route(path, endpoint, **kwargs)


def create_runtime_router() -> APIRouter:
    router = RuntimeRouter(
        prefix="/conversations/{runtime_conversation_id}",
        dependencies=[Depends(require_local_runtime)],
    )
    for source in (
        bash_router,
        file_router,
        git_router,
        desktop_router,
    ):
        router.include_router(source)
    router.add_api_route("/vscode/url", get_runtime_vscode_url, methods=["GET"])
    for route in vscode_router.routes:
        if isinstance(route, APIRoute) and route.path != "/vscode/url":
            router.add_api_route(
                route.path, route.endpoint, methods=list(route.methods)
            )
    router.add_api_route(
        "/mcp/test",
        test_mcp_server,
        methods=["POST"],
        response_model=MCPTestResponse,
        response_model_exclude_none=True,
    )
    return router


async def _require_scoped_runtime_image(host: str, api_key: str | None) -> None:
    headers = {"X-Session-API-Key": api_key} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{host}/server_info", headers=headers)
        response.raise_for_status()
        supported = "conversation_runtime_routes_v1" in response.json().get(
            "capabilities", []
        )
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            502, "Could not check conversation runtime capabilities"
        ) from exc
    if not supported:
        raise HTTPException(
            409,
            "The configured conversation image does not support scoped runtime APIs. "
            "Build or select an agent-server image with conversation_runtime_routes_v1 "
            "and recreate the conversation container. "
            "Legacy runtime routes remain available.",
        )


async def get_runtime_vscode_url(
    request: Request,
    base_url: str | None = None,
    workspace_dir: str | None = None,
) -> VSCodeUrlResponse:
    event_service = request.state.runtime_event_service
    return await get_vscode_url(
        base_url,
        workspace_dir or event_service.get_conversation().workspace.working_dir,
    )
