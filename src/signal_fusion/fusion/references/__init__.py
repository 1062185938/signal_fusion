"""Reference documents used when interpreting signal evidence."""

from importlib.resources import files
from pathlib import Path


TECHNOLOGY_REFERENCE_FILENAME = "technology_reference_lte_wifi_dvbt.md"
PERIODICITY_REFERENCE_FILENAME = "ofdm_periodicity_gate_reference.md"


def technology_reference_path() -> Path:
    """Return the installed LTE/WiFi/DVB-T interpretation guide."""

    return Path(str(files(__package__).joinpath(TECHNOLOGY_REFERENCE_FILENAME)))


def periodicity_reference_path() -> Path:
    """Return the installed LTE/DVB-T periodicity-gate interpretation guide."""

    return Path(str(files(__package__).joinpath(PERIODICITY_REFERENCE_FILENAME)))


__all__ = [
    "PERIODICITY_REFERENCE_FILENAME",
    "TECHNOLOGY_REFERENCE_FILENAME",
    "periodicity_reference_path",
    "technology_reference_path",
]
