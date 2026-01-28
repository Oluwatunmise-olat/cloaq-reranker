import os
import sys
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    reranker_model_name: str
    grpc_server_address: str
    workers: int
    model_max_length: int
    max_message_length: int
    max_receive_message_length: int
    batch_size: int


def load_config() -> Config:
    load_dotenv()
    try:
        model_name = os.getenv(
            "RERANKER_MODEL_NAME",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        grpc_addr = os.getenv("GRPC_SERVER_ADDRESS", "0.0.0.0:40051")
        workers = int(os.getenv("WORKERS", "10"))
        max_len = int(os.getenv("MODEL_MAX_LENGTH", "512"))
        max_msg = int(os.getenv("MAX_MESSAGE_LENGTH", str(50 * 1024**2)))
        batch_size = int(os.getenv("BATCH_SIZE", "32"))
    except ValueError as e:
        logger.error("Configuration parsing error: %s", e)
        sys.exit(1)

    if workers < 1:
        logger.error("WORKERS must be >= 1")
        sys.exit(1)
    if max_len < 1:
        logger.error("MODEL_MAX_LENGTH must be >= 1")
        sys.exit(1)

    return Config(
        reranker_model_name=model_name,
        grpc_server_address=grpc_addr,
        workers=workers,
        model_max_length=max_len,
        max_message_length=max_msg,
        max_receive_message_length=max_msg,
        batch_size=batch_size,
    )
