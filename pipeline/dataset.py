"""
DatasetManager: Handles downloading, extracting, and loading the Find-It-Again dataset.

Dataset:  https://l3i-share.univ-lr.fr/2023Finditagain/index.html
Download: http://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip

=== Dataset Structure (after extraction) ===
data/raw/findit2/
    train/          ← PNG images + OCR .txt files for training split
    val/            ← PNG images + OCR .txt files for validation split
    test/           ← PNG images + OCR .txt files for test split
    train.txt       ← CSV label file: image, digital annotation, handwritten annotation,
                        forged, forgery annotations
    val.txt         ← same format
    test.txt        ← same format

=== Label Format ===
The split files (train.txt, val.txt, test.txt) are CSV files where:
  - "forged" column: "True" (FAKE) or "False" (REAL)
  - "digital annotation" / "handwritten annotation": True/False flags for
    non-fraudulent marks — these are NOT forgeries (important hard negatives).
  - "forgery annotations": JSON with region-level forgery details (for forged=True rows).

=== Pre-collected Metadata ===
data/dataset/findit2/
    train_data.csv  ← enriched CSV with image metadata (width, height, blur, etc.)
    val_data.csv
    test_data.csv

=== Key Statistics ===
Total: 988 receipts | Forged: 163 (~16.5%) | Real: 825 (~83.5%)
Dominant forgery type: CPI - Copy-Paste Inside (~77.6% of all modifications)
Most targeted entity: Total/Payment (~51.4% of all modifications)
"""

from __future__ import annotations

import csv
import hashlib
import logging
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging — reemplaza los print() dispersos por un logger configurable.
# ---------------------------------------------------------------------------
logger = logging.getLogger("dataset_manager")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[dataset] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "sampling.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Hash SHA-256 esperado del ZIP original (None = se omite la verificación).
# Tras la primera descarga exitosa, se puede fijar aquí el hash para
# garantizar integridad en descargas futuras.
EXPECTED_ZIP_SHA256: str | None = None

# Configuración de reintentos para la descarga
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_RETRY_DELAY = 5        # segundos entre reintentos
DOWNLOAD_CONNECT_TIMEOUT = 30   # segundos para establecer conexión
DOWNLOAD_READ_TIMEOUT = 300     # segundos para lectura de datos


class DownloadError(Exception):
    """Error específico para fallos en la descarga del dataset."""
    pass


class ZipValidationError(Exception):
    """Error específico para fallos en la validación del archivo ZIP."""
    pass


class DatasetManager:
    """
    Manages the Find-It-Again receipt dataset.

    Primary usage:
        dm = DatasetManager()
        dm.download()                     # Download findit2.zip if not present
        dm.extract()                      # Extract to data/raw/findit2/
        labels = dm.load_labels()         # {stem: "REAL"|"FAKE"} from train split
        labels = dm.load_labels("test")   # Load a specific split

    Advanced:
        info = dm.load_split_info("train")  # Full dict with metadata for each image
        img  = dm.find_image("X000...", "train")   # Locate image in split directory
        txt  = dm.find_ocr_txt("X000...", "train") # Locate paired OCR text
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        ds_cfg = cfg["dataset"]
        self.download_url: str = ds_cfg["download_url"]
        self.raw_dir: Path = Path(ds_cfg["raw_dir"])
        self.findit2_dir: Path = Path(ds_cfg["findit2_dir"])
        self.metadata_dir: Path = Path(ds_cfg["metadata_dir"])
        self.samples_dir: Path = Path(ds_cfg["samples_dir"])
        self.labels_file: Path = Path(ds_cfg["labels_file"])
        self.splits_cfg: dict = ds_cfg.get("splits", {})

        s_cfg = cfg["sampling"]
        self.default_split: str = s_cfg.get("split", "train")

    # ------------------------------------------------------------------
    # Download & extract
    # ------------------------------------------------------------------

    def download(
        self,
        force: bool = False,
        expected_sha256: str | None = EXPECTED_ZIP_SHA256,
        max_retries: int = DOWNLOAD_MAX_RETRIES,
        retry_delay: float = DOWNLOAD_RETRY_DELAY,
    ) -> Path:
        """
        Descarga findit2.zip con barra de progreso, reintentos y validación.

        Flujo completo:
          1. Verificar si el ZIP ya existe (y es válido).
          2. Probar la conexión al servidor (HEAD request).
          3. Descargar con barra de progreso y feedback en tiempo real.
          4. Validar integridad del ZIP (estructura + hash opcional).

        Args:
            force:           Si True, re-descarga aunque el archivo ya exista.
            expected_sha256: Hash SHA-256 esperado. None = omitir verificación.
            max_retries:     Número máximo de reintentos ante fallos de red.
            retry_delay:     Segundos de espera entre reintentos.

        Returns:
            Path al archivo ZIP descargado.

        Raises:
            DownloadError:       Si la descarga falla tras todos los reintentos.
            ZipValidationError:  Si el archivo descargado no es un ZIP válido
                                 o no coincide con el hash esperado.
        """
        zip_path = self.raw_dir / "findit2.zip"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # --- Verificar si ya existe un ZIP válido ---
        if zip_path.exists() and not force:
            logger.info("ZIP ya existe en %s. Verificando integridad…", zip_path)
            try:
                self._validate_zip(zip_path, expected_sha256)
                logger.info("ZIP existente es válido. Descarga omitida.")
                return zip_path
            except ZipValidationError as exc:
                logger.warning(
                    "ZIP existente está corrupto o no coincide con el hash "
                    "esperado: %s. Se procederá a re-descargar.", exc
                )

        # --- Sondeo de conexión (HEAD request) ---
        remote_size = self._probe_connection()

        # --- Descarga con reintentos ---
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Intento %d/%d — descargando desde %s …",
                    attempt, max_retries, self.download_url,
                )
                self._download_with_progress(zip_path, remote_size)
                break  # descarga exitosa, salir del bucle
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
                IOError,
            ) as exc:
                last_error = exc
                logger.error(
                    "Intento %d/%d fallido: %s", attempt, max_retries, exc
                )
                if attempt < max_retries:
                    logger.info(
                        "Reintentando en %d segundos…", retry_delay
                    )
                    time.sleep(retry_delay)
        else:
            # Se ejecuta si el for termina sin 'break' (todos los intentos fallaron)
            raise DownloadError(
                f"La descarga falló tras {max_retries} intentos. "
                f"Último error: {last_error}"
            )

        # --- Validación post-descarga ---
        logger.info("Descarga completada. Validando archivo ZIP…")
        self._validate_zip(zip_path, expected_sha256)
        logger.info("✓ ZIP válido: %s", zip_path)

        return zip_path

    def _probe_connection(self) -> int | None:
        """
        Envía un HEAD request al servidor para verificar que la URL es
        alcanzable y, de paso, obtener el tamaño del archivo remoto.

        Returns:
            Tamaño en bytes del archivo remoto (o None si el servidor no
            reporta Content-Length).

        Raises:
            DownloadError: Si el servidor no responde o devuelve un error HTTP.
        """
        logger.info("Verificando conexión con el servidor…")
        try:
            head = requests.head(
                self.download_url,
                timeout=DOWNLOAD_CONNECT_TIMEOUT,
                allow_redirects=True,
            )
            head.raise_for_status()
        except requests.ConnectionError as exc:
            raise DownloadError(
                f"No se pudo establecer conexión con {self.download_url}. "
                f"Verifica tu conexión a Internet y que la URL sea correcta.\n"
                f"Detalle: {exc}"
            ) from exc
        except requests.Timeout as exc:
            raise DownloadError(
                f"Tiempo de espera agotado al contactar {self.download_url} "
                f"(timeout={DOWNLOAD_CONNECT_TIMEOUT}s).\n"
                f"Detalle: {exc}"
            ) from exc
        except requests.HTTPError as exc:
            raise DownloadError(
                f"El servidor respondió con un error HTTP: {exc}"
            ) from exc

        remote_size = head.headers.get("Content-Length")
        if remote_size is not None:
            remote_size = int(remote_size)
            size_mb = remote_size / (1024 * 1024)
            logger.info(
                "✓ Conexión exitosa. Tamaño del archivo: %.1f MB", size_mb
            )
        else:
            logger.info(
                "✓ Conexión exitosa. (El servidor no reportó el tamaño del archivo.)"
            )

        return remote_size

    def _download_with_progress(
        self, zip_path: Path, expected_size: int | None
    ) -> None:
        """
        Descarga el archivo con barra de progreso usando tqdm.

        Tras la descarga, verifica que el tamaño local coincida con el
        tamaño reportado por el servidor (si estaba disponible).

        Args:
            zip_path:      Ruta destino del archivo ZIP.
            expected_size: Tamaño esperado en bytes (de Content-Length), o None.

        Raises:
            requests.HTTPError / ConnectionError / Timeout: ante fallos de red.
            DownloadError: si el tamaño descargado no coincide.
        """
        response = requests.get(
            self.download_url,
            stream=True,
            timeout=(DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT),
        )
        response.raise_for_status()

        # Si el HEAD no reportó tamaño, intentar obtenerlo del GET
        total = expected_size or int(response.headers.get("Content-Length", 0))

        # Escribir a un archivo temporal primero (evitar dejar un ZIP
        # parcial si la descarga se interrumpe)
        tmp_path = zip_path.with_suffix(".zip.part")
        downloaded_bytes = 0

        try:
            with (
                open(tmp_path, "wb") as f,
                tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="findit2.zip",
                    ncols=80,
                    bar_format=(
                        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}]"
                    ),
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    pbar.update(len(chunk))
        except Exception:
            # Limpiar archivo parcial si la descarga falló
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        # Verificar tamaño descargado vs esperado
        if total and downloaded_bytes != total:
            tmp_path.unlink(missing_ok=True)
            raise DownloadError(
                f"Tamaño descargado ({downloaded_bytes:,} bytes) no coincide "
                f"con el esperado ({total:,} bytes). Posible descarga incompleta."
            )

        # Renombrar .part → .zip (operación atómica en la mayoría de FS)
        tmp_path.replace(zip_path)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info("Archivo guardado en %s (%.1f MB)", zip_path, size_mb)

    def _validate_zip(
        self, zip_path: Path, expected_sha256: str | None = None
    ) -> None:
        """
        Valida la integridad del archivo ZIP descargado en dos niveles:

        1. **Validación estructural**: Verifica que el archivo sea un ZIP
           legítimo y que todos sus miembros pasen el CRC check interno.
        2. **Validación de hash** (opcional): Compara el SHA-256 del archivo
           completo contra un hash esperado.

        Args:
            zip_path:        Ruta al archivo ZIP.
            expected_sha256: Hash SHA-256 esperado (hex). None para omitir.

        Raises:
            ZipValidationError: Si cualquier validación falla.
        """
        # --- Nivel 1: ¿Es un ZIP válido? ---
        if not zipfile.is_zipfile(zip_path):
            raise ZipValidationError(
                f"El archivo {zip_path} no es un ZIP válido."
            )

        # --- Nivel 2: Verificar CRC de cada miembro ---
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                bad_file = zf.testzip()
                if bad_file is not None:
                    raise ZipValidationError(
                        f"CRC check fallido para '{bad_file}' dentro del ZIP. "
                        f"El archivo probablemente está corrupto."
                    )
        except zipfile.BadZipFile as exc:
            raise ZipValidationError(
                f"No se pudo abrir el ZIP: {exc}"
            ) from exc

        # --- Nivel 3: Hash SHA-256 (si se proporcionó) ---
        if expected_sha256 is not None:
            actual_hash = self._compute_sha256(zip_path)
            if actual_hash != expected_sha256.lower():
                raise ZipValidationError(
                    f"Hash SHA-256 no coincide.\n"
                    f"  Esperado: {expected_sha256.lower()}\n"
                    f"  Obtenido: {actual_hash}\n"
                    f"El archivo puede haber sido modificado o la descarga "
                    f"se corrompió."
                )
            logger.info("✓ Hash SHA-256 verificado correctamente.")

    @staticmethod
    def _compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
        """
        Calcula el hash SHA-256 de un archivo leyendo en bloques.

        Usa lectura incremental para no cargar el archivo completo en
        memoria (importante para archivos grandes).

        Args:
            file_path:  Ruta al archivo.
            chunk_size: Tamaño del bloque de lectura en bytes.

        Returns:
            Hash SHA-256 en formato hexadecimal (minúsculas).
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(chunk_size), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def extract(self, force: bool = False) -> Path:
        """Extract the ZIP archive into data/raw/, producing data/raw/findit2/."""
        zip_path = self.raw_dir / "findit2.zip"
        extracted_marker = self.raw_dir / ".extracted"

        if extracted_marker.exists() and not force:
            logger.info("Ya extraído en %s. Omitiendo.", self.findit2_dir)
            return self.findit2_dir

        if not zip_path.exists():
            raise FileNotFoundError(
                f"ZIP no encontrado en {zip_path}. Ejecuta download() primero."
            )

        logger.info("Extrayendo %s → %s …", zip_path, self.raw_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.raw_dir)

        extracted_marker.touch()
        logger.info("Extracción completada. Raíz del dataset: %s", self.findit2_dir)
        return self.findit2_dir

    # ------------------------------------------------------------------
    # Label loading (primary interface)
    # ------------------------------------------------------------------

    def load_labels(self, split: str | None = None) -> dict[str, str]:
        """
        Load ground-truth labels for a given split.

        Returns a dict mapping filename stem → "REAL" | "FAKE".

        Priority order for reading labels:
          1. Pre-collected metadata CSV (data/dataset/findit2/<split>_data.csv)
          2. Raw split file from extracted dataset (data/raw/findit2/<split>.txt)

        Args:
            split: "train", "val", or "test". Defaults to sampling config split.
        """
        split = split or self.default_split
        self._validate_split(split)

        # Try pre-collected CSV first (already processed and enriched)
        meta_csv = self.metadata_dir / self.splits_cfg[split]["metadata_csv"]
        if meta_csv.exists():
            labels = self._load_labels_from_metadata_csv(meta_csv)
            self._print_label_summary(labels, split, source=str(meta_csv))
            return labels

        # Fall back to raw split txt file
        split_txt = self.findit2_dir / self.splits_cfg[split]["txt"]
        if split_txt.exists():
            labels = self._load_labels_from_split_txt(split_txt)
            self._print_label_summary(labels, split, source=str(split_txt))
            return labels

        raise FileNotFoundError(
            f"Cannot find label file for split '{split}'. "
            f"Looked for:\n  {meta_csv}\n  {split_txt}\n"
            f"Run `dm.download()` and `dm.extract()` first, or ensure "
            f"data/dataset/findit2/{split}_data.csv exists."
        )

    def load_all_splits(self) -> dict[str, dict[str, str]]:
        """
        Load labels for all three splits.

        Returns:
            {"train": {...}, "val": {...}, "test": {...}}
        """
        return {split: self.load_labels(split) for split in ["train", "val", "test"]}

    def load_split_info(self, split: str | None = None) -> list[dict[str, Any]]:
        """
        Load full per-image metadata for a split.

        Returns a list of dicts with all CSV columns, plus:
          - "label": "REAL" or "FAKE"
          - "split": split name

        Args:
            split: "train", "val", or "test".
        """
        split = split or self.default_split
        self._validate_split(split)

        meta_csv = self.metadata_dir / self.splits_cfg[split]["metadata_csv"]
        if not meta_csv.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {meta_csv}. "
                "Ensure data/dataset/findit2/ contains the pre-collected CSVs."
            )

        rows = []
        with open(meta_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                forged_raw = row.get("forged", "False").strip()
                label = "FAKE" if forged_raw in ("True", "1", "true") else "REAL"
                row["label"] = label
                row["split"] = split
                rows.append(dict(row))

        return rows

    # ------------------------------------------------------------------
    # Image and OCR text lookup
    # ------------------------------------------------------------------

    def find_image(self, stem: str, split: str | None = None) -> Path | None:
        """
        Find the PNG image for a given filename stem in a split directory.
        """
        split = split or self.default_split

        if split in self.splits_cfg:
            split_dir = self.findit2_dir / self.splits_cfg[split]["images_subdir"]
            for ext in IMAGE_EXTENSIONS:
                candidate = split_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate

        for ext in IMAGE_EXTENSIONS:
            for candidate in self.raw_dir.rglob(f"{stem}{ext}"):
                return candidate

        return None

    def find_ocr_txt(self, stem: str, split: str | None = None) -> Path | None:
        """
        Find the paired OCR .txt file for a given image stem.
        """
        split = split or self.default_split

        if split in self.splits_cfg:
            split_dir = self.findit2_dir / self.splits_cfg[split]["images_subdir"]
            candidate = split_dir / f"{stem}.txt"
            if candidate.exists():
                return candidate

        for candidate in self.raw_dir.rglob(f"{stem}.txt"):
            return candidate

        return None

    def all_images(self, split: str | None = None) -> list[Path]:
        """
        Return all image paths for a given split (or all images if split is None).
        """
        if split is not None:
            self._validate_split(split)
            split_dir = self.findit2_dir / self.splits_cfg[split]["images_subdir"]
            return sorted(
                p for p in split_dir.rglob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )

        return sorted(
            p for p in self.raw_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_labels_from_metadata_csv(self, csv_path: Path) -> dict[str, str]:
        labels: dict[str, str] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = row.get("image", "").strip()
                if not image:
                    continue
                stem = Path(image).stem
                forged_raw = row.get("forged", "False").strip()
                if forged_raw in ("True", "1", "true", "TRUE"):
                    labels[stem] = "FAKE"
                else:
                    labels[stem] = "REAL"
        return labels

    def _load_labels_from_split_txt(self, txt_path: Path) -> dict[str, str]:
        labels: dict[str, str] = {}
        with open(txt_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, quotechar='"')
            for row in reader:
                image = row.get("image", "").strip()
                if not image:
                    continue
                stem = Path(image).stem
                forged_raw = row.get("forged", "False").strip()
                if forged_raw in ("True", "1", "true", "TRUE"):
                    labels[stem] = "FAKE"
                else:
                    labels[stem] = "REAL"
        return labels

    def _validate_split(self, split: str) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Invalid split '{split}'. Must be one of: train, val, test"
            )

    @staticmethod
    def _print_label_summary(
        labels: dict[str, str], split: str, source: str
    ) -> None:
        real = sum(1 for v in labels.values() if v == "REAL")
        fake = sum(1 for v in labels.values() if v == "FAKE")
        logger.info(
            "Loaded %d labels from %s split (%d REAL, %d FAKE) ← %s",
            len(labels), split, real, fake, source,
        )