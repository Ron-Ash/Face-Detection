import threading


class InterruptibleTimer:
    def __init__(self, timeout: float, on_expire):
        self.lock = threading.Lock()
        self._timeout = timeout
        self._on_expire = on_expire
        self._timer: threading.Timer | None = None
        self._interrupted = False
    
    @property
    def interrupted(self) -> bool:
        with self.lock:
            return self._interrupted
    
    def _interrupt_unsafe(self):
        """Must be called with self.lock held."""
        self._interrupted = True
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _start_unsafe(self):
        """Must be called with self.lock held."""
        if self._timer is not None: return
        self._timer = threading.Timer(self._timeout, self._on_expire)
        self._timer.daemon = True
        self._timer.start()
        self._interrupted = False

    def interrupt(self):
        with self.lock:
            self._interrupt_unsafe()

    def start(self):
        with self.lock:
            self._start_unsafe()

    def reset(self):
        with self.lock:
            self._interrupt_unsafe()
            self._start_unsafe()
    