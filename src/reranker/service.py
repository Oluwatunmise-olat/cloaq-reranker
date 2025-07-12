import threading
import logging
import grpc
from sentence_transformers import CrossEncoder
from protos import reranker_pb2, reranker_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class RerankerService(reranker_pb2_grpc.RerankerServiceServicer):
    def __init__(self, model: CrossEncoder, batch_size: int):
        self.model = model
        self.batch_size = batch_size
        self.lock = threading.Lock()

    def Rerank(
        self,
        request: reranker_pb2.RerankRequest,
        context: grpc.ServicerContext,
    ) -> reranker_pb2.RerankResponse:
        if not request.documents:
            logger.info("No documents; returning empty response.")
            return reranker_pb2.RerankResponse(results=[])

        logger.info(
            "Rerank request: query=%r, docs=%d",
            request.query,
            len(request.documents),
        )

        pairs = [[request.query, doc.page_content] for doc in request.documents]

        try:
            with self.lock:
                scores = self.model.predict(pairs, batch_size=self.batch_size).tolist()
        except Exception:
            logger.exception("Error during model.predict")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Model prediction failed")
            return reranker_pb2.RerankResponse()

        items = [
            reranker_pb2.RerankResultItem(score=s, index=i)
            for i, s in enumerate(scores)
        ]
        items.sort(key=lambda it: it.score, reverse=True)
        logger.info("Returning %d items", len(items))

        return reranker_pb2.RerankResponse(results=items)
