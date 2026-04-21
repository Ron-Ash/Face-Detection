
import os
import threading
from time import sleep
from contextlib import contextmanager

import weaviate

from concurrency.dockerComposeService import DockerComposeService
from concurrency.interruptTimer import InterruptibleTimer

class WeaviateClientManager:
    def __init__(self, host: str = os.getenv("DB_HOST", 'localhost'), 
                 port: int = 8080, grpc_port: int = 50051, 
                 username: str = os.getenv("DB_USER", 'user'), 
                 password: str = os.getenv("DB_PASSWORD",'password')):
        self.host, self.port, self.grpc_port, self.username, self.password = host, port, grpc_port, username, password
        self.composeService = DockerComposeService(serviceName="weaviate")
        self._active_calls = 0
        self._lock = threading.Lock()
        self.timer = InterruptibleTimer(5*60, self._stop)
    
    def _connect_with_retry(self, retries:int = 10, delay: int = 2):
        for i in range(retries):
            client = None
            try:
                client = weaviate.connect_to_local(host=self.host, port=self.port, grpc_port=self.grpc_port)
                if client.is_ready(): return client
            except Exception as e:
                print(f"Retry {i+1}/{retries}: Weaviate not ready yet...")
            finally:
                if client is not None:
                    try: 
                        if not client.is_ready():
                            client.close()
                    except: pass
            sleep(delay)
        raise RuntimeError("Failed to connect to Weaviate after retries")
    
    def _acquire(self):
        with self._lock:
            self.timer.interrupt()
            if not self.composeService._is_ready(): self.composeService.start()
            self._active_calls += 1
    
    def _release(self):
        with self._lock:
            self._active_calls -= 1
            if self._active_calls == 0: self.timer.start()

    @contextmanager
    def _call(self):
        self._acquire()
        client = self._connect_with_retry()
        try:
            yield client
        finally:
            client.close()
            self._release()
    
    def _stop(self):
        with self._lock:
            if self.timer.interrupted or self._active_calls > 0: return
            if self.composeService._is_ready():
                self.composeService.stop()

# if __name__ == "__main__":
#     clientManager = WeaviateClientManager()

#     with clientManager._call() as client:
#         if client.is_ready():
#             print("Weaviate is ready")
#     sleep(60*2)
#     print("stopped")
