"""Dữ liệu DEV (~36 câu) — chấm điểm train, không trùng train_data.py."""

from data.train_data import _make


C, P, S = "CATEGORY", "PRODUCT", "SPEC"


DEV_DATA = [
    # P1 - CATEGORY only (8)
    _make([("máy in", C)]),
    _make([("máy chiếu", C)]),
    _make([("nồi cơm", C)]),
    _make([("balo laptop", C)]),
    _make([("dán màn hình", C)]),
    _make([("cáp sạc", C)]),
    _make([("giá đỡ", C)]),
    _make([("phụ kiện ipad", C)]),

    # P2 - PRODUCT only (8)
    _make([("iphone 14", P)]),
    _make([("galaxy s23 ultra", P)]),
    _make([("macbook pro m2", P)]),
    _make([("dji mavic", P)]),
    _make([("sony a7", P)]),
    _make([("bose quietcomfort", P)]),
    _make([("razer basilisk", P)]),
    _make([("samsung galaxy tab s9", P)]),

    # P3 - CATEGORY + PRODUCT (10)
    _make([("laptop", C), ("acer", P)]),
    _make([("điện thoại", C), ("oppo", P)]),
    _make([("điện thoại", C), ("vivo", P)]),
    _make([("tai nghe", C), ("bose", P)]),
    _make([("ốp lưng", C), ("samsung galaxy", P)]),
    _make([("sạc", C), ("xiaomi", P)]),
    _make([("loa", C), ("marshall", P)]),
    _make([("máy ảnh", C), ("nikon", P)]),
    _make([("tivi", C), ("sony", P)]),
    _make([("máy giặt", C), ("panasonic", P)]),

    # P4 - CATEGORY + PRODUCT + SPEC (10)
    _make([("laptop", C), ("acer nitro", P), ("16gb", S)]),
    _make([("điện thoại", C), ("oppo reno", P), ("256gb", S)]),
    _make([("sạc", C), ("baseus", P), ("30w", S)]),
    _make([("ốp lưng", C), ("samsung s24", P), ("bao da", S)]),
    _make([("tai nghe", C), ("bose", P), ("bluetooth 5.3", S)]),
    _make([("màn hình", C), ("lg", P), ("34 inch", S)]),
    _make([("tivi", C), ("sony", P), ("75 inch", S)]),
    _make([("thẻ nhớ", C), ("kingston", P), ("64gb", S)]),
    _make([("pin dự phòng", C), ("baseus", P), ("30000mah", S)]),
    _make([("ipad", C), ("air", P), ("m2", S)]),
]


def _verify(name, data):
    for text, ann in data:
        for start, end, label in ann["entities"]:
            span = text[start:end]
            assert span, f"{name}: empty span in {text!r} at [{start}:{end}]"
            assert span.strip() == span, (
                f"{name}: whitespace bug in {text!r} -> {span!r}"
            )
            assert label in {C, P, S}, f"{name}: unknown label {label}"


_verify("DEV_DATA", DEV_DATA)


def _check_no_overlap():
    """Đảm bảo không câu dev nào trùng train."""
    from data.train_data import TRAIN_DATA
    train_texts = {t for t, _ in TRAIN_DATA}
    for text, _ in DEV_DATA:
        assert text not in train_texts, f"DEV trùng TRAIN: {text!r}"


_check_no_overlap()
