"""路径与 .env 读取。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

_WRITABLE = [RAW_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR]


def ensure_dirs() -> None:
    for d in _WRITABLE:
        d.mkdir(parents=True, exist_ok=True)


def load_dotenv() -> dict:
    """读项目根的 .env，返回 dict；文件不存在返回 {}。"""
    p = PROJECT_ROOT / ".env"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


if __name__ == "__main__":
    ensure_dirs()
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
