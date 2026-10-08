from pathlib import Path

from PIL import Image

from robustness_lab.parameter_diagrams import generate_parameter_diagrams


def test_all_rs_parameter_diagrams_are_generated_and_nonuniform(tmp_path):
    result = generate_parameter_diagrams(str(tmp_path))
    assert result["diagram_count"] == 11
    assert Path(result["index"]).is_file()
    assert Path(result["contact_sheet"]).is_file()
    assert Path(result["universal_modifier_diagram"]).is_file()
    assert [item["code"] for item in result["records"]] == ["rs%d" % index for index in range(1, 12)]
    for record in result["records"]:
        image = Image.open(record["png"]).convert("RGB")
        assert image.size == (1800, 1000)
        assert len(image.getcolors(maxcolors=image.width * image.height)) > 8
        assert record["parameters"]
