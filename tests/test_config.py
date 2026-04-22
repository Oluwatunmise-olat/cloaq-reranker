import pytest
from unittest.mock import patch


def test_load_config_defaults():
    with patch.dict("os.environ", {}, clear=True):
        with patch("dotenv.load_dotenv"):
            from reranker.config import load_config

            config = load_config()

    assert config.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.grpc_server_address == "0.0.0.0:40051"
    assert config.workers == 10
    assert config.model_max_length == 512
    assert config.max_message_length == 50 * 1024**2
    assert config.max_receive_message_length == 50 * 1024**2
    assert config.batch_size == 32


def test_load_config_custom_env():
    env = {
        "RERANKER_MODEL_NAME": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "GRPC_SERVER_ADDRESS": "127.0.0.1:50051",
        "WORKERS": "4",
        "MODEL_MAX_LENGTH": "256",
        "MAX_MESSAGE_LENGTH": "1048576",
        "BATCH_SIZE": "16",
    }
    with patch.dict("os.environ", env, clear=True):
        with patch("dotenv.load_dotenv"):
            from reranker.config import load_config

            config = load_config()

    assert config.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2"
    assert config.grpc_server_address == "127.0.0.1:50051"
    assert config.workers == 4
    assert config.model_max_length == 256
    assert config.max_message_length == 1048576
    assert config.max_receive_message_length == 1048576
    assert config.batch_size == 16


@pytest.mark.parametrize(
    ("env_overrides",),
    [
        ({"WORKERS": "not-a-number"},),
        ({"WORKERS": "0"},),
        ({"WORKERS": "-5"},),
        ({"MODEL_MAX_LENGTH": "abc"},),
        ({"MODEL_MAX_LENGTH": "0"},),
    ],
)
def test_load_config_invalid_numeric_values_exit(env_overrides):
    with patch.dict("os.environ", env_overrides, clear=False):
        with patch("dotenv.load_dotenv"):
            from reranker.config import load_config

            with pytest.raises(SystemExit) as exc:
                load_config()
    assert exc.value.code == 1


def test_load_config_max_receive_matches_max_message():
    with patch.dict("os.environ", {"MAX_MESSAGE_LENGTH": "2097152"}, clear=False):
        with patch("dotenv.load_dotenv"):
            from reranker.config import load_config

            config = load_config()
    assert config.max_message_length == config.max_receive_message_length
