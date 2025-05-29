"""
utils/data_loader.py
"""

import csv
import io, zipfile, pathlib, requests, pandas as pd

__all__ = ["load_uci"]

ROOT = "https://archive.ics.uci.edu/ml/machine-learning-databases"

def _download(dataset: str, file: str | None) -> bytes:
    url = f"{ROOT}/{dataset}/{file or ''}".rstrip("/")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content

def _open_zip(buf: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        member = next((m for m in zf.namelist()
                       if m.endswith((".csv", ".data", ".xls", ".xlsx"))), None)
        if member is None:
            raise ValueError("Не нашёл табличных файлов в архиве")
        return _read_table(zf.read(member), pathlib.Path(member).suffix)

def _read_table(buf: bytes, suffix: str, *, header="infer") -> pd.DataFrame:
    """
    Считывает табличный файл в pandas.DataFrame.
    - для .csv /.data  ➜ автоматический выбор разделителя
    - для .xls /.xlsx  ➜ через read_excel
    """
    if suffix in (".csv", ".data"):
        sample = buf[:2048].decode("utf-8", errors="ignore")
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            sep = dialect.delimiter
        except csv.Error:
            sep = ","  # fallback
        return pd.read_csv(io.BytesIO(buf), sep=sep, header=header, na_filter=True)

    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(io.BytesIO(buf), header=header, engine="openpyxl")

    raise ValueError(f"Не умею читать файлы с расширением {suffix}")


def _to_xy(df: pd.DataFrame, target: str | int):
    if isinstance(target, int):
        y = df.iloc[:, target].values
        X = df.drop(df.columns[target], axis=1).values
        feat_names = df.drop(df.columns[target], axis=1).columns.to_list()
    else:
        y = df[target].values
        X = df.drop(target, axis=1).values
        feat_names = df.drop(target, axis=1).columns.to_list()
    return X, y, feat_names

def load_uci(dataset: str, file: str | None = None, *,
             target: str | int,
             cache_dir: str | pathlib.Path | None = "~/.ucicache"):
    """Скачивает <dataset>/<file> с UCI, выделяет целевой столбец ``target``."""
    cache_dir = pathlib.Path(cache_dir).expanduser() if cache_dir else None
    fname     = (file or dataset.split("/")[-1])
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / fname
        if not path.exists():
            path.write_bytes(_download(dataset, file))
        buf = path.read_bytes()
    else:
        buf = _download(dataset, file)

    if fname.endswith(".zip"):
        df = _open_zip(buf)
    else:
        df = _read_table(buf, pathlib.Path(fname).suffix)

    print(f"[DataLoader] shape={df.shape}")
    return _to_xy(df, target)
