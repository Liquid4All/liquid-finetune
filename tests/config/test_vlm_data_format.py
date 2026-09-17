import json

from PIL import Image

from leap_finetune.data_loading.validate_dataset_format import (
    get_row_filter,
    normalize_columns,
)


def test_normalize_vlm_sft_decodes_arrow_json_content_items(tmp_path):
    """HF Arrow may expose heterogeneous typed content items as JSON strings."""
    image_path = tmp_path / "face.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    row = {
        "messages": [
            {
                "role": "user",
                "content": [json.dumps({"type": "image", "image": str(image_path)})],
            },
            {
                "role": "assistant",
                "content": [json.dumps({"type": "text", "text": "none"})],
            },
        ]
    }

    normalized = normalize_columns("vlm_sft")(row)

    assert normalized["messages"][0]["content"] == [
        {"type": "image", "image": str(image_path)}
    ]
    assert normalized["messages"][1]["content"] == [{"type": "text", "text": "none"}]
    assert get_row_filter("vlm_sft")(normalized)
