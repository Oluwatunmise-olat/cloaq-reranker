import grpc
import numpy as np
import pytest
from unittest.mock import MagicMock

from protos import reranker_pb2
from reranker.service import RerankerService


def make_service(scores=None, batch_size=32):
    model = MagicMock()
    if scores is not None:
        model.predict.return_value = np.array(scores)
    return RerankerService(model=model, batch_size=batch_size), model


def make_request(query, page_contents):
    docs = [reranker_pb2.Document(page_content=c) for c in page_contents]
    return reranker_pb2.RerankRequest(query=query, documents=docs)


def make_context():
    ctx = MagicMock(spec=grpc.ServicerContext)
    return ctx


class TestRerankEmptyDocuments:
    def test_returns_empty_results(self):
        service, model = make_service()
        request = reranker_pb2.RerankRequest(query="hello", documents=[])
        response = service.Rerank(request, make_context())

        assert list(response.results) == []
        model.predict.assert_not_called()


class TestRerankSingleDocument:
    def test_returns_one_result(self):
        service, _ = make_service(scores=[0.9])
        request = make_request("query", ["doc A"])
        response = service.Rerank(request, make_context())

        assert len(response.results) == 1
        assert response.results[0].index == 0
        assert pytest.approx(response.results[0].score, abs=1e-5) == 0.9


class TestRerankSorting:
    def test_results_sorted_by_score_descending(self):
        service, _ = make_service(scores=[0.1, 0.9, 0.5])
        request = make_request("query", ["low", "high", "mid"])
        response = service.Rerank(request, make_context())

        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_original_indices_preserved_after_sort(self):
        service, _ = make_service(scores=[0.2, 0.8, 0.5])
        request = make_request("query", ["doc0", "doc1", "doc2"])
        response = service.Rerank(request, make_context())

        assert response.results[0].index == 1
        assert response.results[1].index == 2
        assert response.results[2].index == 0

    def test_identical_scores_all_returned(self):
        service, _ = make_service(scores=[0.5, 0.5, 0.5])
        request = make_request("q", ["a", "b", "c"])
        response = service.Rerank(request, make_context())

        assert len(response.results) == 3


class TestRerankPairs:
    def test_passes_query_doc_pairs_to_model(self):
        service, model = make_service(scores=[0.7, 0.3])
        model.predict.return_value = np.array([0.7, 0.3])
        request = make_request("my query", ["first doc", "second doc"])
        service.Rerank(request, make_context())

        call_args = model.predict.call_args
        pairs = call_args[0][0]
        assert pairs == [["my query", "first doc"], ["my query", "second doc"]]

    def test_passes_batch_size_to_model(self):
        service, model = make_service(scores=[0.5], batch_size=8)
        model.predict.return_value = np.array([0.5])
        request = make_request("q", ["doc"])
        service.Rerank(request, make_context())

        call_args = model.predict.call_args
        assert call_args[1]["batch_size"] == 8


class TestRerankErrorHandling:
    def test_model_exception_sets_grpc_internal_code(self):
        service, model = make_service()
        model.predict.side_effect = RuntimeError("GPU OOM")
        ctx = make_context()
        request = make_request("query", ["doc"])

        response = service.Rerank(request, ctx)

        ctx.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
        ctx.set_details.assert_called_once_with("Model prediction failed")
        assert list(response.results) == []

    def test_model_exception_returns_empty_response(self):
        service, model = make_service()
        model.predict.side_effect = ValueError("bad input")
        response = service.Rerank(make_request("q", ["d"]), make_context())

        assert isinstance(response, reranker_pb2.RerankResponse)
        assert len(response.results) == 0


class TestRerankThreadSafety:
    def test_lock_is_acquired_during_predict(self):
        service, model = make_service(scores=[0.5])
        model.predict.return_value = np.array([0.5])

        # Replace the instance lock with a MagicMock that still works as a context manager
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=False)
        service.lock = mock_lock

        service.Rerank(make_request("q", ["doc"]), make_context())

        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()
