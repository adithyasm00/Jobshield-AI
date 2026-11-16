from types import SimpleNamespace

from app import build_scam_category_data


def test_build_scam_category_data_aggregates_real_scan_categories():
    scans = [
        SimpleNamespace(category="Cryptocurrency / Task Scam"),
        SimpleNamespace(category="Application / Registration Fee Scam"),
        SimpleNamespace(category="Cryptocurrency / Task Scam"),
        SimpleNamespace(category="General Job Posting"),
        SimpleNamespace(category=None),
        SimpleNamespace(category="Application / Registration Fee Scam"),
    ]

    result = build_scam_category_data(scans)

    assert result[0]["label"] == "Cryptocurrency / Task Scam"
    assert result[0]["count"] == 2
    assert result[1]["label"] == "Application / Registration Fee Scam"
    assert result[1]["count"] == 2
    assert any(item["label"] == "General Job Posting" for item in result)
