"""Query test sau train — không dùng khi train. Chạy: make test-predict."""

TEST_QUERIES = [
    # Head queries — dễ
    "laptop",
    "iphone",
    "tai nghe",
    "máy giặt",

    # PRODUCT chi tiết
    "iphone 16 pro max",
    "galaxy s24 ultra",
    "macbook air m3",
    "dji mini 4 pro",

    # CATEGORY + PRODUCT
    "laptop lenovo thinkpad",
    "tai nghe sony wireless",
    "ốp lưng iphone 16",
    "máy ảnh fujifilm",

    # Long-tail đa thuộc tính (P4)
    "laptop dell xps 15 inch 16gb",
    "điện thoại samsung s24 256gb",
    "sạc anker 65w usb-c chính hãng",
    "tai nghe apple airpods pro chống nước",
    "thẻ nhớ sandisk 128gb 200mb/s",
    "ipad pro 11 inch 128gb",

    # Query có kèm giá / stopword — model chỉ cần nhận CATEGORY/PRODUCT/SPEC;
    # phần "giá rẻ", "mua", "tìm"... sẽ do logic khác lọc.
    "iphone 16 giá rẻ",
    "laptop dell dưới 20 triệu",
    "tai nghe bluetooth từ 500k đến 1 triệu",
    "mua đồng hồ thông minh giá rẻ",
    "tìm ipad pro 12.9 inch",
    "còn iphone 16 pro max không",
    "bán macbook pro m2 cũ dưới 25 triệu",
]
