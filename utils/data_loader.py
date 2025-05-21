# utils/data_loader.py
from __future__ import annotations
import io
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import urllib.request

# URL & локальные пути ---------------------------------------------------------
URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00294/CCPP.zip"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DATA_DIR / "ccpp.zip"
EXCEL_NAME = "CCPP/Folds5x2_pp.xlsx"


# ---------------------------- служебные функции ------------------------------
def _download_if_needed(url: str = URL, dst: Path = ZIP_PATH) -> None:
    if dst.exists() and dst.stat().st_size > 1_000:
        return                       # уже скачан и похоже на zip
    print(f"Downloading {url} → {dst} …")
    try:
        with urllib.request.urlopen(url) as resp, open(dst, "wb") as f_out:
            f_out.write(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"❌ download failed: {e}") from e
    if dst.stat().st_size < 1_000:
        raise RuntimeError("❌ downloaded file is too small – looks like an error page.")
    print("✓ download complete")


def _extract_excel(zip_path: Path = ZIP_PATH,
                   member: str = EXCEL_NAME) -> io.BytesIO:
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            raise RuntimeError(f"❌ {member} not found in archive. "
                               "Check the namelist: " + str(zf.namelist()[:10]))
        with zf.open(member) as excel_file:
            return io.BytesIO(excel_file.read())


# ------------------------------ публичный API --------------------------------
def load_ccpp(
    test_size: float = 0.2,
    scale: bool = True,
    random_state: int | None = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает Combined Cycle Power Plant dataset и возвращает
    (X_train, X_test, y_train, y_test).

    Parameters
    ----------
    test_size : float
        Доля объектов в тестовой выборке.
    scale : bool
        Если True — стандартизирует признаки и цель с помощью StandardScaler.
    random_state : int | None
        Число для воспроизводимости train_test_split.
    """
    # 1. загрузка / распаковка -------------------------------------------------
    _download_if_needed()
    excel_bytes = _extract_excel()

    df = pd.read_excel(excel_bytes)

    # 2. разделение признаков и цели ------------------------------------------
    X = df.drop("PE", axis=1).values.astype(np.float32)   # (T, V, AP, RH)
    y = df["PE"].values.reshape(-1, 1).astype(np.float32)

    # 3. train / test split ----------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. стандартизация --------------------------------------------------------
    if scale:
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        X_tr = x_scaler.fit_transform(X_tr)
        X_te = x_scaler.transform(X_te)
        y_tr = y_scaler.fit_transform(y_tr)
        y_te = y_scaler.transform(y_te)

    return X_tr, X_te, y_tr, y_te
