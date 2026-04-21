import os
import subprocess
from time import sleep, time


class DockerComposeService:
    def __init__(self, extra_path: str | None = "", serviceName: str = "server"):
        self.path = os.path.join(os.getcwd(), extra_path)
        self.serviceName = serviceName

    def start(self):
        subprocess.run(["docker", "compose", "up", "-d"], check=True,cwd=self.path)
        self._wait_until_ready()

    def stop(self):
        subprocess.run(["docker", "compose", "down"], check=True, cwd=self.path)
    
    def reset_database(self):
        self._wait_until_ready()
        subprocess.run(["docker", "compose", "down", "-v"], check=True, cwd=self.path)
        self.start()

    def _is_ready(self) -> bool:
        result = subprocess.run(["docker", "compose", "ps", "--services", "--status=running"],
                                capture_output=True, text=True, check=True, cwd=self.path)
        return result.returncode == 0 and self.serviceName in result.stdout

    def _wait_until_ready(self, timeout: int = 30, interval: float = 1.0):
        deadline = time() + timeout
        while time() < deadline:
            if self._is_ready():
                return
            sleep(interval)
        raise TimeoutError(
            f"{self.__class__.__name__} did not become ready within {timeout}s"
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()