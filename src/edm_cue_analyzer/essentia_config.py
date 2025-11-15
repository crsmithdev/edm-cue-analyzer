"""Centralized Essentia configuration to avoid duplicate code."""

# Configure Essentia logging BEFORE importing essentia.standard
# This prevents the "MusicExtractorSVM: no classifier models" message
try:
    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    essentia.log.errorActive = True

    import essentia.standard as es

    ESSENTIA_AVAILABLE = True
except ImportError:
    es = None
    ESSENTIA_AVAILABLE = False


def enable_verbose_logging():
    """Enable Essentia INFO and WARNING messages for verbose mode."""
    if ESSENTIA_AVAILABLE:
        import essentia
        essentia.log.infoActive = True
        essentia.log.warningActive = True


__all__ = ["es", "ESSENTIA_AVAILABLE", "enable_verbose_logging"]
