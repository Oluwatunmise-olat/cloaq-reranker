import pytest
from unittest.mock import patch, MagicMock


def test_load_model_success():
    mock_model = MagicMock()
    with patch("reranker.model.CrossEncoder", return_value=mock_model) as mock_cls:
        from reranker.model import load_model

        result = load_model("cross-encoder/ms-marco-MiniLM-L-6-v2", 512)

    mock_cls.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    assert result is mock_model


def test_load_model_passes_max_length():
    mock_model = MagicMock()
    with patch("reranker.model.CrossEncoder", return_value=mock_model) as mock_cls:
        from reranker.model import load_model

        load_model("some-model", 256)

    mock_cls.assert_called_once_with("some-model", max_length=256)


def test_load_model_failure_exits():
    with patch("reranker.model.CrossEncoder", side_effect=Exception("download failed")):
        from reranker.model import load_model

        with pytest.raises(SystemExit) as exc:
            load_model("nonexistent/model", 512)

    assert exc.value.code == 1
