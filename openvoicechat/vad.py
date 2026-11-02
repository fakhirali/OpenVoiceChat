"""Silero VAD v5 running on onnxruntime."""

import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URLS = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx",
)
MODEL_PATH = Path.home() / ".cache" / "openvoicechat" / "silero_vad.onnx"
WINDOW = 512
CONTEXT = 64
STATE_SHAPE = (2, 1, 128)


class VAD:
    """Frame-level Silero VAD. Input frames are PCM16 little-endian 16 kHz."""

    def __init__(self, threshold: float = 0.6, sample_rate: int = 16000):
        import onnxruntime as ort

        self.threshold = threshold
        self.sample_rate = sample_rate
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        self._session = _load_session(ort)

    def __call__(self, frame: bytes) -> bool:
        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        audio = audio[:WINDOW]
        if len(audio) < WINDOW:
            audio = np.pad(audio, (0, WINDOW - len(audio)))
        window = np.concatenate((self._context, audio.reshape(1, -1)), axis=1)
        out, state = self._session.run(
            None,
            {
                "input": window,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        self._state = state.astype(np.float32, copy=False)
        self._context = window[:, -CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0]) >= self.threshold

    def reset(self) -> None:
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)


def _load_session(ort):
    path = MODEL_PATH
    if path.exists():
        try:
            return _validated_session(ort, path)
        except Exception:  # noqa: BLE001 - redownload any unreadable cache entry
            path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in MODEL_URLS:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as file:
                shutil.copyfileobj(response, file)
            session = _validated_session(ort, tmp)
            os.replace(tmp, path)
            return session
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Unable to download the Silero VAD model: {last_error}")


def _validated_session(ort, path: Path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    names = {item.name for item in session.get_inputs()}
    if names != {"input", "state", "sr"}:
        raise RuntimeError(f"Unexpected Silero VAD inputs: {sorted(names)}")
    session.run(
        None,
        {
            "input": np.zeros((1, WINDOW + CONTEXT), dtype=np.float32),
            "state": np.zeros(STATE_SHAPE, dtype=np.float32),
            "sr": np.array(16000, dtype=np.int64),
        },
    )
    return session
