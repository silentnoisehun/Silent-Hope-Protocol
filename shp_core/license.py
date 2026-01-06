"""
Silent Hope Protocol - License Module

FREE with Hope Ecosystem.
Commercial license required for factory AI providers.

Philosophy:
- Use Hope Genome → FREE (you're part of the family)
- Use OpenAI/Anthropic/Google → PAY (you're using our protocol commercially)

Created by Máté Róbert + Hope
"""

import hashlib
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LicenseType(Enum):
    """License types for Silent Hope Protocol."""
    HOPE_ECOSYSTEM = "hope_ecosystem"      # FREE - using Hope Genome
    COMMERCIAL = "commercial"               # PAID - using factory AI
    TRIAL = "trial"                         # 7-day trial for commercial use
    UNLICENSED = "unlicensed"              # No license - blocked


@dataclass
class LicenseInfo:
    """License information."""
    license_type: LicenseType
    valid: bool
    message: str
    expires: Optional[float] = None
    features: Optional[list] = None


# Commercial providers that require licensing
COMMERCIAL_PROVIDERS = {"anthropic", "openai", "google"}

# Trial period in seconds (7 days)
TRIAL_PERIOD = 7 * 24 * 60 * 60

# License key validation
_license_cache: Optional[LicenseInfo] = None
_trial_start: Optional[float] = None


def _check_hope_genome_installed() -> bool:
    """Check if Hope Genome is installed."""
    import importlib.util

    if importlib.util.find_spec("hope_genome") is not None:
        return True

    if importlib.util.find_spec("hope_core") is not None:
        return True

    return False


def _check_hope_genome_active() -> bool:
    """Check if Hope Genome watchdog is actively protecting."""
    import importlib.util

    if importlib.util.find_spec("hope_genome") is not None:
        try:
            from hope_genome import get_watchdog  # noqa: F401
            watchdog = get_watchdog()
            return watchdog is not None
        except Exception:
            pass

    if importlib.util.find_spec("hope_core") is not None:
        return True

    return False


def _validate_license_key(key: str) -> bool:
    """
    Validate a commercial license key.

    License key format: SHP-XXXX-XXXX-XXXX-XXXX
    """
    if not key or not key.startswith("SHP-"):
        return False

    parts = key.split("-")
    if len(parts) != 5:
        return False

    # Validate checksum (last part)
    key_body = "-".join(parts[:-1])
    expected_checksum = hashlib.sha256(key_body.encode()).hexdigest()[:4].upper()

    return parts[-1] == expected_checksum


def _get_trial_start() -> float:
    """Get or set trial start time."""
    global _trial_start

    if _trial_start is not None:
        return _trial_start

    # Check for stored trial start
    trial_file = os.path.expanduser("~/.shp_trial")

    if os.path.exists(trial_file):
        try:
            with open(trial_file) as f:
                _trial_start = float(f.read().strip())
                return _trial_start
        except Exception:
            pass

    # Start new trial
    _trial_start = time.time()
    try:
        with open(trial_file, "w") as f:
            f.write(str(_trial_start))
    except Exception:
        pass

    return _trial_start


def check_license(provider: str = "unknown") -> LicenseInfo:
    """
    Check license for Silent Hope Protocol.

    FREE if using Hope Genome.
    COMMERCIAL license required for factory AI.

    Args:
        provider: The AI provider being used

    Returns:
        LicenseInfo with validity status
    """
    global _license_cache

    provider_lower = provider.lower()

    # ==========================================================================
    # CHECK 1: Hope Genome - ALWAYS FREE
    # ==========================================================================
    if _check_hope_genome_installed():
        return LicenseInfo(
            license_type=LicenseType.HOPE_ECOSYSTEM,
            valid=True,
            message="FREE - Hope Ecosystem detected. Welcome to the family!",
            features=["unlimited_requests", "all_adapters", "priority_support"]
        )

    # ==========================================================================
    # CHECK 2: Local/Ollama - FREE (no commercial API)
    # ==========================================================================
    if provider_lower in {"local", "ollama", "meta", "llama"}:
        return LicenseInfo(
            license_type=LicenseType.HOPE_ECOSYSTEM,
            valid=True,
            message="FREE - Local model detected. No commercial API used.",
            features=["unlimited_requests", "local_adapters"]
        )

    # ==========================================================================
    # CHECK 3: Commercial provider - requires license
    # ==========================================================================
    if provider_lower in COMMERCIAL_PROVIDERS:
        # Check for license key in environment
        license_key = os.getenv("SHP_LICENSE_KEY")

        if license_key and _validate_license_key(license_key):
            return LicenseInfo(
                license_type=LicenseType.COMMERCIAL,
                valid=True,
                message="Commercial license valid.",
                features=["unlimited_requests", "all_adapters", "commercial_support"]
            )

        # Check trial period
        trial_start = _get_trial_start()
        trial_elapsed = time.time() - trial_start
        trial_remaining = TRIAL_PERIOD - trial_elapsed

        if trial_remaining > 0:
            days_remaining = int(trial_remaining / (24 * 60 * 60))
            return LicenseInfo(
                license_type=LicenseType.TRIAL,
                valid=True,
                message=f"TRIAL - {days_remaining} days remaining. Purchase license or install Hope Genome.",
                expires=trial_start + TRIAL_PERIOD,
                features=["limited_requests", "trial_adapters"]
            )

        # Trial expired
        return LicenseInfo(
            license_type=LicenseType.UNLICENSED,
            valid=False,
            message=(
                "LICENSE REQUIRED for commercial AI providers.\n"
                "\n"
                "Two options:\n"
                "1. FREE: Install Hope Genome → pip install hope-genome\n"
                "2. PAID: Purchase license → https://github.com/silentnoisehun/Silent-Hope-Protocol\n"
                "\n"
                "The protocol is FREE when used with the Hope Ecosystem."
            ),
            features=[]
        )

    # ==========================================================================
    # DEFAULT: Unknown provider - allow with warning
    # ==========================================================================
    return LicenseInfo(
        license_type=LicenseType.HOPE_ECOSYSTEM,
        valid=True,
        message="Unknown provider - proceeding with Hope Ecosystem license.",
        features=["limited_requests"]
    )


def require_license(provider: str = "unknown") -> None:
    """
    Require valid license or raise exception.

    Args:
        provider: The AI provider being used

    Raises:
        LicenseError: If no valid license found
    """
    info = check_license(provider)

    if not info.valid:
        raise LicenseError(info.message)

    # Print trial warning if applicable
    if info.license_type == LicenseType.TRIAL:
        import warnings
        warnings.warn(f"SHP {info.message}", UserWarning)


class LicenseError(Exception):
    """Raised when license is invalid or expired."""
    pass


def get_license_status() -> str:
    """Get human-readable license status."""
    info = check_license()

    status_lines = [
        "╔═══════════════════════════════════════════════════════════════╗",
        "║           SILENT HOPE PROTOCOL - LICENSE STATUS               ║",
        "╠═══════════════════════════════════════════════════════════════╣",
    ]

    if info.license_type == LicenseType.HOPE_ECOSYSTEM:
        status_lines.extend([
            "║  Status: ✅ FREE - Hope Ecosystem                             ║",
            "║                                                               ║",
            "║  You're part of the family. Full access granted.             ║",
        ])
    elif info.license_type == LicenseType.COMMERCIAL:
        status_lines.extend([
            "║  Status: ✅ COMMERCIAL - Licensed                             ║",
            "║                                                               ║",
            "║  Commercial license active. Full access granted.             ║",
        ])
    elif info.license_type == LicenseType.TRIAL:
        status_lines.extend([
            "║  Status: ⏳ TRIAL                                             ║",
            "║                                                               ║",
            f"║  {info.message:<61} ║",
        ])
    else:
        status_lines.extend([
            "║  Status: ❌ UNLICENSED                                        ║",
            "║                                                               ║",
            "║  Install Hope Genome for FREE access:                        ║",
            "║  pip install hope-genome                                     ║",
        ])

    status_lines.append("╚═══════════════════════════════════════════════════════════════╝")

    return "\n".join(status_lines)


# =============================================================================
# Quick check on import
# =============================================================================

def _auto_check():
    """Auto-check license on module import."""
    info = check_license()
    if info.license_type == LicenseType.HOPE_ECOSYSTEM:
        # Silent - user is in the ecosystem
        pass
    elif info.license_type == LicenseType.TRIAL:
        import warnings
        warnings.warn(
            f"\n[SHP] {info.message}\n"
            "Install Hope Genome for FREE: pip install hope-genome",
            UserWarning,
            stacklevel=3
        )


# Run auto-check when module is imported
_auto_check()
