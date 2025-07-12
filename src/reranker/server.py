import signal
import logging
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from .config import load_config
from .model import load_model
from .service import RerankerService
from protos import reranker_pb2, reranker_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def serve() -> None:
    cfg = load_config()
    model = load_model(cfg.reranker_model_name, cfg.model_max_length)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cfg.workers),
        options=[
            ("grpc.max_send_message_length", cfg.max_message_length),
            ("grpc.max_receive_message_length", cfg.max_receive_message_length),
        ],
    )

    reranker_pb2_grpc.add_RerankerServiceServicer_to_server(
        RerankerService(model, cfg.batch_size), server
    )

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    service_names = (
        reranker_pb2.DESCRIPTOR.services_by_name["RerankerService"].full_name,
        health.SERVICE_NAME,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    server.add_insecure_port(cfg.grpc_server_address)
    server.start()
    logger.info("Server listening on %s", cfg.grpc_server_address)

    def _graceful_shutdown(signum, frame):
        logger.info("Signal %s received, shutting down...", signum)
        server.stop(grace=30)

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    server.wait_for_termination()
