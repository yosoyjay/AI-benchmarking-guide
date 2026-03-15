"""Reusable Docker container helpers for AMD ROCm benchmarks."""

import logging
from types import TracebackType
from typing import Any

import docker

logger = logging.getLogger(__name__)

# Options shared by every AMD ROCm benchmark container.
_AMD_BASE_OPTIONS = {
    "ipc_mode": "host",
    "network": "host",
    "group_add": ["render"],
    "privileged": True,
    "security_opt": ["seccomp=unconfined"],
    "cap_add": ["CAP_SYS_ADMIN", "SYS_PTRACE"],
    "devices": ["/dev/kfd", "/dev/dri", "/dev/mem"],
    "tty": True,
    "detach": True,
}


class AmdContainer:
    """Context manager that runs an AMD ROCm Docker container.

    Usage::

        with AmdContainer(image, work_dir="/data") as container:
            container.exec_run("python bench.py")
    """

    def __init__(
        self,
        image: str,
        work_dir: str,
        *,
        entrypoint: str | None = None,
        environment: dict[str, str] | None = None,
    ):
        self.image = image
        self.work_dir = work_dir
        self.entrypoint = entrypoint
        self.environment = environment

        self._client: docker.DockerClient | None = None
        self._container = None

    def __enter__(self) -> Any:
        self._client = docker.from_env()

        opts: dict = {**_AMD_BASE_OPTIONS}
        opts["volumes"] = {self.work_dir: {"bind": self.work_dir, "mode": "rw"}}

        if self.entrypoint is not None:
            opts["entrypoint"] = self.entrypoint
        if self.environment is not None:
            opts["environment"] = self.environment

        logger.info("Pulling and starting container %s ...", self.image)
        self._container = self._client.containers.run(self.image, **opts)
        logger.info("Created container %s", self._container.id)
        return self._container

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> bool:
        try:
            if self._container is not None:
                self._container.kill()
        except docker.errors.NotFound:
            pass  # already gone
        finally:
            if self._client is not None:
                self._client.close()
        return False
