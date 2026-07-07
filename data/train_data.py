"""Dữ liệu TRAIN (~150 câu). Nhãn: CATEGORY, PRODUCT, SPEC. Dùng `_make()` — xem docs/training_data_spec.md."""


def _make(tokens):
    """Ghép các token (mỗi token 1 space giữa) và tự tính char offset.

    tokens: list of (text, label|None). label=None nghĩa là không gắn nhãn.
    Trả về: (full_text, {"entities": [(start, end, label), ...]}).
    """
    parts = []
    entities = []
    pos = 0
    for i, (text, label) in enumerate(tokens):
        if i > 0:
            parts.append(" ")
            pos += 1
        start = pos
        parts.append(text)
        pos += len(text)
        if label is not None:
            entities.append((start, pos, label))
    return ("".join(parts), {"entities": entities})


C, P, S = "CATEGORY", "PRODUCT", "SPEC"


TRAIN_DATA = [
    # =========================================================================
    # P1 - CHỈ CATEGORY (30 câu) — head query khám phá
    # =========================================================================
    _make([("laptop", C)]),
    _make([("điện thoại", C)]),
    _make([("máy giặt", C)]),
    _make([("máy rửa bát", C)]),
    _make([("máy ép chậm", C)]),
    _make([("máy lọc nước", C)]),
    _make([("máy xay thịt", C)]),
    _make([("máy hút bụi", C)]),
    _make([("máy ảnh", C)]),
    _make([("máy tính bảng", C)]),
    _make([("tivi", C)]),
    _make([("loa", C)]),
    _make([("chuột", C)]),
    _make([("bàn phím cơ", C)]),
    _make([("tai nghe", C)]),
    _make([("ốp lưng", C)]),
    _make([("kính cường lực", C)]),
    _make([("sạc", C)]),
    _make([("cáp", C)]),
    _make([("pin dự phòng", C)]),
    _make([("đế sạc", C)]),
    _make([("túi chống sốc", C)]),
    _make([("ba lô", C)]),
    _make([("thẻ nhớ", C)]),
    _make([("flycam", C)]),
    _make([("camera", C)]),
    _make([("quạt", C)]),
    _make([("điều hòa", C)]),
    _make([("tủ lạnh", C)]),
    _make([("phụ kiện iphone", C)]),

    # =========================================================================
    # P2 - CHỈ PRODUCT (30 câu) — user tìm theo brand/model
    # =========================================================================
    _make([("iphone 15", P)]),
    _make([("iphone 15 pro max", P)]),
    _make([("iphone 16", P)]),
    _make([("galaxy s24", P)]),
    _make([("galaxy a37", P)]),
    _make([("galaxy z fold 5", P)]),
    _make([("macbook air", P)]),
    _make([("macbook air m2", P)]),
    _make([("macbook pro m3", P)]),
    _make([("dji avata", P)]),
    _make([("dji mini 3", P)]),
    _make([("jbl partybox", P)]),
    _make([("jbl charge 5", P)]),
    _make([("sony wh-1000xm5", P)]),
    _make([("apple watch series 9", P)]),
    _make([("canon eos 4000d", P)]),
    _make([("xiaomi pad 8 pro", P)]),
    _make([("honor 600", P)]),
    _make([("honor 600 lite", P)]),
    _make([("honor 600 pro", P)]),
    _make([("honor power 2", P)]),
    _make([("tecno pova curve 2", P)]),
    _make([("lenovo legion tab", P)]),
    _make([("logitech mx master 3s", P)]),
    _make([("keychron k2 pro", P)]),
    _make([("aula s98 pro", P)]),
    _make([("dell xps", P)]),
    _make([("sandisk", P)]),
    _make([("anker", P)]),
    _make([("dreame h16", P)]),

    # =========================================================================
    # P3 - CATEGORY + PRODUCT (40 câu)
    # =========================================================================
    _make([("laptop", C), ("dell", P)]),
    _make([("laptop", C), ("asus", P)]),
    _make([("laptop", C), ("lenovo", P)]),
    _make([("laptop", C), ("hp", P)]),
    _make([("laptop", C), ("macbook air", P)]),
    _make([("điện thoại", C), ("samsung", P)]),
    _make([("điện thoại", C), ("apple", P)]),
    _make([("điện thoại", C), ("xiaomi", P)]),
    _make([("điện thoại", C), ("honor", P)]),
    _make([("điện thoại", C), ("tecno", P)]),
    _make([("tai nghe", C), ("sony", P)]),
    _make([("tai nghe", C), ("jbl", P)]),
    _make([("tai nghe", C), ("apple", P)]),
    _make([("ốp lưng", C), ("iphone", P)]),
    _make([("ốp lưng", C), ("ipad", P)]),
    _make([("ốp lưng", C), ("samsung", P)]),
    _make([("kính cường lực", C), ("iphone", P)]),
    _make([("kính cường lực", C), ("ipad", P)]),
    _make([("sạc", C), ("anker", P)]),
    _make([("sạc", C), ("apple", P)]),
    _make([("loa", C), ("jbl", P)]),
    _make([("loa", C), ("sony", P)]),
    _make([("flycam", C), ("dji", P)]),
    _make([("chuột", C), ("logitech", P)]),
    _make([("chuột", C), ("razer", P)]),
    _make([("bàn phím cơ", C), ("keychron", P)]),
    _make([("bàn phím cơ", C), ("aula", P)]),
    _make([("màn hình", C), ("dell", P)]),
    _make([("màn hình", C), ("samsung", P)]),
    _make([("máy ảnh", C), ("canon", P)]),
    _make([("máy ảnh", C), ("sony", P)]),
    _make([("tivi", C), ("samsung", P)]),
    _make([("tivi", C), ("lg", P)]),
    _make([("tivi", C), ("tcl", P)]),
    _make([("thẻ nhớ", C), ("sandisk", P)]),
    _make([("thẻ nhớ", C), ("samsung", P)]),
    _make([("máy giặt", C), ("lg", P)]),
    _make([("máy giặt", C), ("samsung", P)]),
    _make([("máy hút bụi", C), ("dreame", P)]),
    _make([("pin dự phòng", C), ("anker", P)]),

    # =========================================================================
    # P4 - CATEGORY + PRODUCT + SPEC (50 câu) — long-tail đa thuộc tính
    # =========================================================================
    _make([("laptop", C), ("dell xps", P), ("16gb", S)]),
    _make([("laptop", C), ("asus", P), ("8gb", S)]),
    _make([("laptop", C), ("lenovo legion", P), ("32gb", S)]),
    _make([("laptop", C), ("macbook air", P), ("m2", S)]),
    _make([("laptop", C), ("hp pavilion", P), ("cũ", S)]),
    _make([("laptop", C), ("dell", P), ("256gb ssd", S)]),
    _make([("laptop", C), ("asus rog", P), ("144hz", S)]),
    _make([("điện thoại", C), ("samsung a37", P), ("8gb", S)]),
    _make([("điện thoại", C), ("honor 600", P), ("128gb", S)]),
    _make([("điện thoại", C), ("xiaomi 17", P), ("5g", S)]),
    _make([("điện thoại", C), ("samsung s24", P), ("5g", S)]),
    _make([("điện thoại", C), ("honor 600 pro", P), ("chính hãng", S)]),
    _make([("điện thoại", C), ("iphone 15", P), ("256gb", S)]),
    _make([("điện thoại", C), ("iphone 14", P), ("chính hãng", S)]),
    _make([("sạc", C), ("anker", P), ("65w", S)]),
    _make([("sạc", C), ("anker", P), ("usb-c", S)]),
    _make([("sạc", C), ("apple", P), ("20w", S)]),
    _make([("sạc", C), ("xiaomi", P), ("33w", S)]),
    _make([("ốp lưng", C), ("iphone 15", P), ("pitaka", S)]),
    _make([("ốp lưng", C), ("ipad air", P), ("bao da", S)]),
    _make([("ốp lưng", C), ("ipad mini", P), ("có chân đứng", S)]),
    _make([("ốp lưng", C), ("iphone 16", P), ("silicone", S)]),
    _make([("tai nghe", C), ("sony", P), ("bluetooth", S)]),
    _make([("tai nghe", C), ("apple", P), ("chống nước", S)]),
    _make([("tai nghe", C), ("jbl", P), ("chống nước", S)]),
    _make([("tai nghe", C), ("bose", P), ("chống ồn", S)]),
    _make([("flycam", C), ("dji mini 3", P), ("combo", S)]),
    _make([("flycam", C), ("dji avata", P), ("chính hãng", S)]),
    _make([("flycam", C), ("dji", P), ("body only", S)]),
    _make([("màn hình", C), ("dell", P), ("27 inch", S)]),
    _make([("màn hình", C), ("samsung", P), ("32 inch", S)]),
    _make([("màn hình", C), ("gaming", P), ("144hz", S)]),
    _make([("màn hình", C), ("lg", P), ("4k", S)]),
    _make([("tivi", C), ("samsung", P), ("55 inch", S)]),
    _make([("tivi", C), ("lg", P), ("65 inch", S)]),
    _make([("tivi", C), ("sony", P), ("4k", S)]),
    _make([("thẻ nhớ", C), ("sandisk", P), ("128gb", S)]),
    _make([("thẻ nhớ", C), ("sandisk", P), ("200mb/s", S)]),
    _make([("thẻ nhớ", C), ("samsung", P), ("256gb", S)]),
    _make([("máy ảnh", C), ("canon eos 4000d", P), ("chính hãng", S)]),
    _make([("loa", C), ("jbl charge 5", P), ("chống nước", S)]),
    _make([("pin dự phòng", C), ("anker", P), ("20000mah", S)]),
    _make([("pin dự phòng", C), ("xiaomi", P), ("10000mah", S)]),
    _make([("máy giặt", C), ("lg", P), ("9kg", S)]),
    _make([("máy giặt", C), ("samsung", P), ("inverter", S)]),
    _make([("ipad", C), ("pro", P), ("11 inch", S)]),
    _make([("ipad", C), ("mini", P), ("128gb", S)]),
    _make([("kính cường lực", C), ("iphone 15", P), ("chống trầy", S)]),
    _make([("bàn phím cơ", C), ("keychron k2 pro", P), ("bluetooth", S)]),
    _make([("chuột", C), ("logitech mx master 3s", P), ("chính hãng", S)]),
]


def _verify(name, data):
    """Kiểm tra offset — chạy ngay khi import."""
    for text, ann in data:
        for start, end, label in ann["entities"]:
            span = text[start:end]
            assert span, f"{name}: empty span in {text!r} at [{start}:{end}]"
            assert span.strip() == span, (
                f"{name}: whitespace bug in {text!r} -> {span!r}"
            )
            assert label in {C, P, S}, f"{name}: unknown label {label}"


_verify("TRAIN_DATA", TRAIN_DATA)
