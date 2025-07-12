import sys
import logging
from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def load_model(name: str, max_length: int) -> CrossEncoder:
    logger.info("Loading CrossEncoder model: %s", name)
    try:
        model = CrossEncoder(name, max_length=max_length)
        logger.info("Model loaded successfully.")
        return model
    except Exception:
        logger.exception("Failed to load model %s", name)
        sys.exit(1)
