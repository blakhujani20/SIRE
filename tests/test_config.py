from src import config

def test_config_vars():
    assert hasattr(config, "IMAGE_DIR")
    assert hasattr(config, "MODEL_NAME")
