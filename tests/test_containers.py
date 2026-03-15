"""Tests for infra.containers.AmdContainer."""

from unittest.mock import MagicMock, patch

import pytest

from infra.containers import AmdContainer


@pytest.fixture()
def mock_docker():
    """Patch docker.from_env and return (client, container) mocks."""
    with patch("infra.containers.docker") as docker_mod:
        client = MagicMock(name="DockerClient")
        container = MagicMock(name="Container")
        container.id = "abc123"
        client.containers.run.return_value = container
        docker_mod.from_env.return_value = client
        docker_mod.errors = __import__("docker").errors
        yield client, container


class TestBaseOptions:
    def test_privileged_and_ipc_mode(self, mock_docker):
        client, _container = mock_docker
        with AmdContainer("img:latest", "/work"):
            pass
        _opts = client.containers.run.call_args
        kwargs = _opts.kwargs if _opts.kwargs else _opts[1]
        assert kwargs["privileged"] is True
        assert kwargs["ipc_mode"] == "host"

    def test_amd_devices_mounted(self, mock_docker):
        client, _container = mock_docker
        with AmdContainer("img:latest", "/work"):
            pass
        kwargs = client.containers.run.call_args.kwargs
        assert "/dev/kfd" in kwargs["devices"]
        assert "/dev/dri" in kwargs["devices"]
        assert "/dev/mem" in kwargs["devices"]

    def test_work_dir_volume(self, mock_docker):
        client, _container = mock_docker
        with AmdContainer("img:latest", "/my/path"):
            pass
        kwargs = client.containers.run.call_args.kwargs
        assert kwargs["volumes"] == {"/my/path": {"bind": "/my/path", "mode": "rw"}}


class TestOptionalParams:
    def test_entrypoint_passed_when_set(self, mock_docker):
        client, _container = mock_docker
        with AmdContainer("img:latest", "/work", entrypoint="/bin/bash"):
            pass
        kwargs = client.containers.run.call_args.kwargs
        assert kwargs["entrypoint"] == "/bin/bash"

    def test_entrypoint_absent_when_not_set(self, mock_docker):
        client, _container = mock_docker
        with AmdContainer("img:latest", "/work"):
            pass
        kwargs = client.containers.run.call_args.kwargs
        assert "entrypoint" not in kwargs

    def test_environment_passed_when_set(self, mock_docker):
        client, _container = mock_docker
        env = {"HF_HOME": "/data"}
        with AmdContainer("img:latest", "/work", environment=env):
            pass
        kwargs = client.containers.run.call_args.kwargs
        assert kwargs["environment"] == {"HF_HOME": "/data"}

    def test_environment_absent_when_not_set(self, mock_docker):
        client, _container = mock_docker
        with AmdContainer("img:latest", "/work"):
            pass
        kwargs = client.containers.run.call_args.kwargs
        assert "environment" not in kwargs


class TestCleanup:
    def test_container_killed_on_exit(self, mock_docker):
        _client, container = mock_docker
        with AmdContainer("img:latest", "/work"):
            pass
        container.kill.assert_called_once()

    def test_client_closed_on_exit(self, mock_docker):
        client, _container = mock_docker
        with AmdContainer("img:latest", "/work"):
            pass
        client.close.assert_called_once()

    def test_client_closed_even_on_exception(self, mock_docker):
        client, _container = mock_docker
        with pytest.raises(RuntimeError):
            with AmdContainer("img:latest", "/work"):
                raise RuntimeError("boom")
        client.close.assert_called_once()

    def test_container_killed_even_on_exception(self, mock_docker):
        _client, container = mock_docker
        with pytest.raises(RuntimeError):
            with AmdContainer("img:latest", "/work"):
                raise RuntimeError("boom")
        container.kill.assert_called_once()

    def test_not_found_on_kill_is_swallowed(self, mock_docker):
        import docker as docker_pkg

        _client, container = mock_docker
        container.kill.side_effect = docker_pkg.errors.NotFound("gone")
        # Should not raise
        with AmdContainer("img:latest", "/work"):
            pass


class TestReturnValue:
    def test_context_manager_yields_container(self, mock_docker):
        _client, container = mock_docker
        with AmdContainer("img:latest", "/work") as ctx:
            assert ctx is container
