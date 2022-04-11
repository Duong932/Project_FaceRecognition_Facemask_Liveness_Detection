import face_recognition
import cv2
import numpy as np
import imutils

# LAU 6
#ip = 'rtsp://admin:123456@172.16.12.151'

# LAU 7
#ip = 'rtsp://admin:LacV!3t$@192.168.100.83:554/Streaming/Channels/101/'

video_capture = cv2.VideoCapture(1)

# Tải ảnh người cần nhận diện (đầu tiên là mình) va ma hoa anh do.
#0
Nguyen_image = face_recognition.load_image_file("Nguyen.jpg")
Nguyen_face_encoding = face_recognition.face_encodings(Nguyen_image)[0]

#1
Duong_image = face_recognition.load_image_file("Duong.png")
Duong_face_encoding = face_recognition.face_encodings(Duong_image)[0]

#2
Hoang_image = face_recognition.load_image_file("Hoang.png")
Hoang_face_encoding = face_recognition.face_encodings(Hoang_image)[0]

#3
bqthai_image = face_recognition.load_image_file("bqthai.png")
bqthai_face_encoding = face_recognition.face_encodings(bqthai_image)[0]

#4
hcchien_image = face_recognition.load_image_file("hcchien.png")
hcchien_face_encoding = face_recognition.face_encodings(hcchien_image)[0]

#5
Giang_image = face_recognition.load_image_file("Giang.png")
Giang_face_encoding = face_recognition.face_encodings(Giang_image)[0]

#6
buihoangnam_image = face_recognition.load_image_file("buihoangnam.png")
buihoangnam_face_encoding = face_recognition.face_encodings(buihoangnam_image)[0]

#7
ChiTam_image = face_recognition.load_image_file("ChiTam.png")
ChiTam_face_encoding = face_recognition.face_encodings(ChiTam_image)[0]

#8
daominhtu_image = face_recognition.load_image_file("daominhtu.png")
daominhtu_face_encoding = face_recognition.face_encodings(daominhtu_image)[0]

#9
dinhcongngoctien_image = face_recognition.load_image_file("dinhcongngoctien.png")
dinhcongngoctien_face_encoding = face_recognition.face_encodings(dinhcongngoctien_image)[0]

#10
dinhvantien_image = face_recognition.load_image_file("dinhvantien.png")
dinhvantien_face_encoding = face_recognition.face_encodings(dinhvantien_image)[0]

#11
doanvanhien_image = face_recognition.load_image_file("doanvanhien.png")
doanvanhien_face_encoding = face_recognition.face_encodings(doanvanhien_image)[0]

#12
duongthilequyen_image = face_recognition.load_image_file("duongthilequyen.png")
duongthilequyen_face_encoding = face_recognition.face_encodings(duongthilequyen_image)[0]

#13
giathinh_image = face_recognition.load_image_file("giathinh.png")
giathinh_face_encoding = face_recognition.face_encodings(giathinh_image)[0]

#14
HaDuyAn_image = face_recognition.load_image_file("HaDuyAn.png")
HaDuyAn_face_encoding = face_recognition.face_encodings(HaDuyAn_image)[0]

#15
HaThan_image = face_recognition.load_image_file("HaThan.png")
HaThan_face_encoding = face_recognition.face_encodings(HaThan_image)[0]

#16
Hiep_image = face_recognition.load_image_file("Hiep.png")
Hiep_face_encoding = face_recognition.face_encodings(Hiep_image)[0]

#17
honhattruong_image = face_recognition.load_image_file("honhattruong.png")
honhattruong_face_encoding = face_recognition.face_encodings(honhattruong_image)[0]

#18
hovinhquang_image = face_recognition.load_image_file("hovinhquang.png")
hovinhquang_face_encoding = face_recognition.face_encodings(hovinhquang_image)[0]

#19
Hoang7_image = face_recognition.load_image_file("Hoang7.png")
Hoang7_face_encoding = face_recognition.face_encodings(Hoang7_image)[0]

#20
Huy_image = face_recognition.load_image_file("Huy.png")
Huy_face_encoding = face_recognition.face_encodings(Huy_image)[0]

#21
huynhphuochoa_image = face_recognition.load_image_file("huynhphuochoa.png")
huynhphuochoa_face_encoding = face_recognition.face_encodings(huynhphuochoa_image)[0]

#22
khang_image = face_recognition.load_image_file("khang.png")
khang_face_encoding = face_recognition.face_encodings(khang_image)[0]

#23
lephamhoaithuong_image = face_recognition.load_image_file("lephamhoaithuong.png")
lephamhoaithuong_face_encoding = face_recognition.face_encodings(lephamhoaithuong_image)[0]

#24
lethanhtan_image = face_recognition.load_image_file("lethanhtan.png")
lethanhtan_face_encoding = face_recognition.face_encodings(lethanhtan_image)[0]

#25
lethithuong_image = face_recognition.load_image_file("lethithuong.png")
lethithuong_face_encoding = face_recognition.face_encodings(lethithuong_image)[0]

#26
Long_image = face_recognition.load_image_file("Long.png")
Long_face_encoding = face_recognition.face_encodings(Long_image)[0]

#27
luonghung_image = face_recognition.load_image_file("luonghung.png")
luonghung_face_encoding = face_recognition.face_encodings(luonghung_image)[0]

#28
luongxuanquang_image = face_recognition.load_image_file("luongxuanquang.png")
luongxuanquang_face_encoding = face_recognition.face_encodings(luongxuanquang_image)[0]

#29
maidinhtoan_image = face_recognition.load_image_file("maidinhtoan.png")
maidinhtoan_face_encoding = face_recognition.face_encodings(maidinhtoan_image)[0]

#30
Man_image = face_recognition.load_image_file("Man.png")
Man_face_encoding = face_recognition.face_encodings(Man_image)[0]

#31
ngothaohoanganh_image = face_recognition.load_image_file("ngothaohoanganh.png")
ngothaohoanganh_face_encoding = face_recognition.face_encodings(ngothaohoanganh_image)[0]

#32
NgocDanh_image = face_recognition.load_image_file("NgocDanh.png")
NgocDanh_face_encoding = face_recognition.face_encodings(NgocDanh_image)[0]

#33
nguyenanhuy_image = face_recognition.load_image_file("nguyenanhuy.png")
nguyenanhuy_face_encoding = face_recognition.face_encodings(nguyenanhuy_image)[0]

#34
nguyenbaoduy_image = face_recognition.load_image_file("nguyenbaoduy.png")
nguyenbaoduy_face_encoding = face_recognition.face_encodings(nguyenbaoduy_image)[0]

#35
nguyendinhhieu_image = face_recognition.load_image_file("nguyendinhhieu.png")
nguyendinhhieu_face_encoding = face_recognition.face_encodings(nguyendinhhieu_image)[0]

#36
nguyendinhtrien_image = face_recognition.load_image_file("nguyendinhtrien.png")
nguyendinhtrien_face_encoding = face_recognition.face_encodings(nguyendinhtrien_image)[0]

#37
nguyenduyquang_image = face_recognition.load_image_file("nguyenduyquang.png")
nguyenduyquang_face_encoding = face_recognition.face_encodings(nguyenduyquang_image)[0]

#38
nguyenduythiem_image = face_recognition.load_image_file("nguyenduythiem.png")
nguyenduythiem_face_encoding = face_recognition.face_encodings(nguyenduythiem_image)[0]

#39
nguyenhoangduc_image = face_recognition.load_image_file("nguyenhoangduc.png")
nguyenhoangduc_face_encoding = face_recognition.face_encodings(nguyenhoangduc_image)[0]

#40
nguyenhoangngocanh_image = face_recognition.load_image_file("nguyenhoangngocanh.png")
nguyenhoangngocanh_face_encoding = face_recognition.face_encodings(nguyenhoangngocanh_image)[0]

#41
nguyenhoangphuong_image = face_recognition.load_image_file("nguyenhoangphuong.png")
nguyenhoangphuong_face_encoding = face_recognition.face_encodings(nguyenhoangphuong_image)[0]

#42
nguyenlevietphi_image = face_recognition.load_image_file("nguyenlevietphi.png")
nguyenlevietphi_face_encoding = face_recognition.face_encodings(nguyenlevietphi_image)[0]

#43
nguyenluongbinh_image = face_recognition.load_image_file("nguyenluongbinh.png")
nguyenluongbinh_face_encoding = face_recognition.face_encodings(nguyenluongbinh_image)[0]

#44
nguyenngoctotran_image = face_recognition.load_image_file("nguyenngoctotran.png")
nguyenngoctotran_face_encoding = face_recognition.face_encodings(nguyenngoctotran_image)[0]

#45
nguyenquocthanh_image = face_recognition.load_image_file("nguyenquocthanh.png")
nguyenquocthanh_face_encoding = face_recognition.face_encodings(nguyenquocthanh_image)[0]

#46
nguyenthanhdat_image = face_recognition.load_image_file("nguyenthanhdat.png")
nguyenthanhdat_face_encoding = face_recognition.face_encodings(nguyenthanhdat_image)[0]

#47
nguyenthanhtrung_image = face_recognition.load_image_file("nguyenthanhtrung.png")
nguyenthanhtrung_face_encoding = face_recognition.face_encodings(nguyenthanhtrung_image)[0]

#48
nguyenthiaithu_image = face_recognition.load_image_file("nguyenthiaithu.png")
nguyenthiaithu_face_encoding = face_recognition.face_encodings(nguyenthiaithu_image)[0]

#49
nguyenthibichvan_image = face_recognition.load_image_file("nguyenthibichvan.png")
nguyenthibichvan_face_encoding = face_recognition.face_encodings(nguyenthibichvan_image)[0]

#50
nguyenthiloi_image = face_recognition.load_image_file("nguyenthiloi.png")
nguyenthiloi_face_encoding = face_recognition.face_encodings(nguyenthiloi_image)[0]

#51
nguyenthithugiang_image = face_recognition.load_image_file("nguyenthithugiang.png")
nguyenthithugiang_face_encoding = face_recognition.face_encodings(nguyenthithugiang_image)[0]

#52
nguyenthithuha_image = face_recognition.load_image_file("nguyenthithuha.png")
nguyenthithuha_face_encoding = face_recognition.face_encodings(nguyenthithuha_image)[0]

#53
nguyenthithuy_image = face_recognition.load_image_file("nguyenthithuy.png")
nguyenthithuy_face_encoding = face_recognition.face_encodings(nguyenthithuy_image)[0]

#54
nguyentrungchien_image = face_recognition.load_image_file("nguyentrungchien.png")
nguyentrungchien_face_encoding = face_recognition.face_encodings(nguyentrungchien_image)[0]

#55
nguyenvuhao_image = face_recognition.load_image_file("nguyenvuhao.png")
nguyenvuhao_face_encoding = face_recognition.face_encodings(nguyenvuhao_image)[0]

#56
nguyenxuancuong_image = face_recognition.load_image_file("nguyenxuancuong.png")
nguyenxuancuong_face_encoding = face_recognition.face_encodings(nguyenxuancuong_image)[0]

#57
phamanhquoc_image = face_recognition.load_image_file("phamanhquoc.png")
phamanhquoc_face_encoding = face_recognition.face_encodings(phamanhquoc_image)[0]

#58
phamhuuthoi_image = face_recognition.load_image_file("phamhuuthoi.png")
phamhuuthoi_face_encoding = face_recognition.face_encodings(phamhuuthoi_image)[0]

#59
phandangvui_image = face_recognition.load_image_file("phandangvui.png")
phandangvui_face_encoding = face_recognition.face_encodings(phandangvui_image)[0]

#60
phanlevutrami_image = face_recognition.load_image_file("phanlevutrami.png")
phanlevutrami_face_encoding = face_recognition.face_encodings(phanlevutrami_image)[0]

#61
phannguyenanhduy_image = face_recognition.load_image_file("phannguyenanhduy.png")
phannguyenanhduy_face_encoding = face_recognition.face_encodings(phannguyenanhduy_image)[0]

#62
phantuanhung_image = face_recognition.load_image_file("phantuanhung.png")
phantuanhung_face_encoding = face_recognition.face_encodings(phantuanhung_image)[0]

#63
phanthilachoa_image = face_recognition.load_image_file("phanthilachoa.png")
phanthilachoa_face_encoding = face_recognition.face_encodings(phanthilachoa_image)[0]

#64
phantruonghuy_image = face_recognition.load_image_file("phantruonghuy.png")
phantruonghuy_face_encoding = face_recognition.face_encodings(phantruonghuy_image)[0]

#65
Phuoc_image = face_recognition.load_image_file("Phuoc.png")
Phuoc_face_encoding = face_recognition.face_encodings(Phuoc_image)[0]

#66
QuocKy_image = face_recognition.load_image_file("QuocKy.png")
QuocKy_face_encoding = face_recognition.face_encodings(QuocKy_image)[0]

#67
soraya_image = face_recognition.load_image_file("soraya.png")
soraya_face_encoding = face_recognition.face_encodings(soraya_image)[0]

#68
Tan_image = face_recognition.load_image_file("Tan.png")
Tan_face_encoding = face_recognition.face_encodings(Tan_image)[0]

#69
tonnuhoangvi_image = face_recognition.load_image_file("tonnuhoangvi.png")
tonnuhoangvi_face_encoding = face_recognition.face_encodings(tonnuhoangvi_image)[0]

#70
tonthattuan_image = face_recognition.load_image_file("tonthattuan.png")
tonthattuan_face_encoding = face_recognition.face_encodings(tonthattuan_image)[0]

#71
Tuoi_image = face_recognition.load_image_file("Tuoi.png")
Tuoi_face_encoding = face_recognition.face_encodings(Tuoi_image)[0]

#72
Tuu_image = face_recognition.load_image_file("Tuu.png")
Tuu_face_encoding = face_recognition.face_encodings(Tuu_image)[0]

#73
thaihoa_image = face_recognition.load_image_file("thaihoa.png")
thaihoa_face_encoding = face_recognition.face_encodings(thaihoa_image)[0]

#74
ThanhHien_image = face_recognition.load_image_file("ThanhHien.png")
ThanhHien_face_encoding = face_recognition.face_encodings(ThanhHien_image)[0]

#75
Thao_image = face_recognition.load_image_file("Thao.png")
Thao_face_encoding = face_recognition.face_encodings(Thao_image)[0]

#76
thoang_image = face_recognition.load_image_file("thoang.png")
thoang_face_encoding = face_recognition.face_encodings(thoang_image)[0]

#77
thuytrang_image = face_recognition.load_image_file("thuytrang.png")
thuytrang_face_encoding = face_recognition.face_encodings(thuytrang_image)[0]

#78
tranminhquan_image = face_recognition.load_image_file("tranminhquan.png")
tranminhquan_face_encoding = face_recognition.face_encodings(tranminhquan_image)[0]

#79
tranquochoan_image = face_recognition.load_image_file("tranquochoan.png")
tranquochoan_face_encoding = face_recognition.face_encodings(tranquochoan_image)[0]

#80
tranthevinh_image = face_recognition.load_image_file("tranthevinh.png")
tranthevinh_face_encoding = face_recognition.face_encodings(tranthevinh_image)[0]

#81
tranthihang_image = face_recognition.load_image_file("tranthihang.png")
tranthihang_face_encoding = face_recognition.face_encodings(tranthihang_image)[0]

#82
tranthihuynhnhu_image = face_recognition.load_image_file("tranthihuynhnhu.png")
tranthihuynhnhu_face_encoding = face_recognition.face_encodings(tranthihuynhnhu_image)[0]

#83
trang_image = face_recognition.load_image_file("trang.png")
trang_face_encoding = face_recognition.face_encodings(trang_image)[0]

#84
trinh_image = face_recognition.load_image_file("trinh.png")
trinh_face_encoding = face_recognition.face_encodings(trinh_image)[0]

#85
Trung_image = face_recognition.load_image_file("Trung.png")
Trung_face_encoding = face_recognition.face_encodings(Trung_image)[0]

#86
trungkien_image = face_recognition.load_image_file("trungkien.png")
trungkien_face_encoding = face_recognition.face_encodings(trungkien_image)[0]

#87
truonglam_image = face_recognition.load_image_file("truonglam.png")
truonglam_face_encoding = face_recognition.face_encodings(truonglam_image)[0]

#88
truongthihongtham_image = face_recognition.load_image_file("truongthihongtham.png")
truongthihongtham_face_encoding = face_recognition.face_encodings(truongthihongtham_image)[0]

#89
VDHoang_image = face_recognition.load_image_file("VDHoang.png")
VDHoang_face_encoding = face_recognition.face_encodings(VDHoang_image)[0]

#90
vtthuy_image = face_recognition.load_image_file("vtthuy.png")
vtthuy_face_encoding = face_recognition.face_encodings(vtthuy_image)[0]

#91
vuvanduong_image = face_recognition.load_image_file("vuvanduong.png")
vuvanduong_face_encoding = face_recognition.face_encodings(vuvanduong_image)[0]

#92
Vy_image = face_recognition.load_image_file("Vy.png")
Vy_face_encoding = face_recognition.face_encodings(Vy_image)[0]

#tHÊM 1 NGƯỜI = CÁCH
"""
name_image = face_recognition.load_image_file("name.jpg")
name_face_encoding = face_recognition.face_encodings(name_image)[0]
"""

# Tạo 1 mang cac khuon mat da duoc ma hoa va ten
known_face_encodings = [

    Nguyen_face_encoding,
    Duong_face_encoding,
    Hoang_face_encoding,
    bqthai_face_encoding,
    hcchien_face_encoding,
    Giang_face_encoding,
    buihoangnam_face_encoding,
    ChiTam_face_encoding,
    daominhtu_face_encoding,
    dinhcongngoctien_face_encoding,
    dinhvantien_face_encoding,
    doanvanhien_face_encoding,
    duongthilequyen_face_encoding,
    giathinh_face_encoding,
    HaDuyAn_face_encoding,
    HaThan_face_encoding,
    Hiep_face_encoding,
    honhattruong_face_encoding,
    hovinhquang_face_encoding,
    Hoang7_face_encoding,
    Huy_face_encoding,
    huynhphuochoa_face_encoding,
    khang_face_encoding,
    lephamhoaithuong_face_encoding,
    lethanhtan_face_encoding,
    lethithuong_face_encoding,
    Long_face_encoding,
    luonghung_face_encoding,
    luongxuanquang_face_encoding,
    maidinhtoan_face_encoding,
    Man_face_encoding,
    ngothaohoanganh_face_encoding,
    NgocDanh_face_encoding,
    nguyenanhuy_face_encoding,
    nguyenbaoduy_face_encoding,
    nguyendinhhieu_face_encoding,
    nguyendinhtrien_face_encoding,
    nguyenduyquang_face_encoding,
    nguyenduythiem_face_encoding,
    nguyenhoangduc_face_encoding,
    nguyenhoangngocanh_face_encoding,
    nguyenhoangphuong_face_encoding,
    nguyenlevietphi_face_encoding,
    nguyenluongbinh_face_encoding,
    nguyenngoctotran_face_encoding,
    nguyenquocthanh_face_encoding,
    nguyenthanhdat_face_encoding,
    nguyenthanhtrung_face_encoding,
    nguyenthiaithu_face_encoding,
    nguyenthibichvan_face_encoding,
    nguyenthiloi_face_encoding,
    nguyenthithugiang_face_encoding,
    nguyenthithuha_face_encoding,
    nguyenthithuy_face_encoding,
    nguyentrungchien_face_encoding,
    nguyenvuhao_face_encoding,
    nguyenxuancuong_face_encoding,
    phamanhquoc_face_encoding,
    phandangvui_face_encoding,
    phanlevutrami_face_encoding,
    phannguyenanhduy_face_encoding,
    phantuanhung_face_encoding,
    phanthilachoa_face_encoding,
    phantruonghuy_face_encoding,
    Phuoc_face_encoding,
    QuocKy_face_encoding,
    soraya_face_encoding,
    Tan_face_encoding,
    tonnuhoangvi_face_encoding,
    tonthattuan_face_encoding,
    Tuoi_face_encoding,
    Tuu_face_encoding,
    thaihoa_face_encoding,
    ThanhHien_face_encoding,
    Thao_face_encoding,
    thoang_face_encoding,
    thuytrang_face_encoding,
    tranminhquan_face_encoding,
    tranquochoan_face_encoding,
    tranthevinh_face_encoding,
    tranthihang_face_encoding,
    tranthihuynhnhu_face_encoding,
    trang_face_encoding,
    trinh_face_encoding,
    Trung_face_encoding,
    trungkien_face_encoding,
    truonglam_face_encoding,
    truongthihongtham_face_encoding,
    VDHoang_face_encoding,
    vtthuy_face_encoding,
    vuvanduong_face_encoding,
    Vy_face_encoding

]
known_face_names = [
    "NGUYEN",
    "DUONG",
    "HOANG",
    "THAI",
    "CHIEN",
    "GIANG",
    "NAM",
    "CHI TAM",
    "DAO MINH TU",
    "DINH CONG NGOC TIEN",
    "DINH VAN TIEN",
    "DOAN VAN HIEN",
    "DUONG THI LE QUYEN",
    "GIA THINH",
    "HA DUY AN"
    "HA THAN",
    "HIEP",
    "HO NHAT TRUONG",
    "HO VINH QUANG",
    "HOANG7",
    "HUY",
    "HUYNH PHUOC HOA",
    "KHANG",
    "LE PHAM HOAI THUONG",
    "LE THANH TAN",
    "LE THI THUONG",
    "LONG",
    "LUONG HUNG",
    "LUONG XUAN QUANG",
    "MAI DINH TOAN",
    "MAN",
    "NGO THAO HOANG ANH",
    "NGOC DANH",
    "NGUYEN AN HUY",
    "NGUYEN BAO DUY",
    "NGUYEN DINH HIEU",
    "NGUYEN DINH TRIEN",
    "NGUYEN DUY QUANG",
    "NGUYEN DUY THIEM",
    "NGUYEN HOANG DUC",
    "NGUYEN HOANG NGOC ANH",
    "NGUYEN HOANG PHUONG",
    "NGUYEN LE VIET PHI",
    "NGUYEN LUONG BINH",
    "NGUYEN NGOC TO TRAN",
    "NGUYEN QUOC THANH",
    "NGUYEN THANH DAT",
    "NGUYEN THANH TRUNG",
    "NGUYEN THI AI THU",
    "NGUYEN THI BICH VAN",
    "NGUYEN THI LOI",
    "NGUYEN THI THU GIANG",
    "NGUYEN THI THU HA",
    "NGUYEN THI THUY",
    "NGUYEN TRUNG CHIEN",
    "NGUYEN VU HAO",
    "NGUYEN XUAN CUONG",
    "PHAM ANH QUOC",
    "PHAM HUU THOI",
    "PHAN DANG VUI",
    "PHAN LE VU TRA MI",
    "PHAN NGUYEN ANH DUY",
    "PHAN TUAN HUNG",
    "PHAN THI LAC HOA",
    "PHAN TRUONG HUY",
    "PHUOC",
    "QUOC KY",
    "SORAYA",
    "Tan",
    "TON NU HOANG VI",
    "TON THAT TUAN",
    "TUOI",
    "TUU",
    "THAI HOA",
    "THANH HIEN",
    "THAO",
    "THOANG",
    "THUY TRANG",
    "TRAN MINH QUAN",
    "TRAN QUOC HOAN",
    "TRAN THE VINH",
    "TRAN THI HANG",
    "TRAN THI HUYNH NHU",
    "TRANG",
    "TRINH",
    "TRUNG",
    "TRUNG KIEN",
    "TRUONG LAM",
    "TRUONG THI HONG THAM",
    "VDHOANG",
    "VTTHUY",
    "VU VAN DUONG",
    "VY"
]

while True:

    #frame = vs.read()

    # to have a maximum width of 400 pixels

    # Grab a single frame of video
    ret, frame = video_capture.read()

    frame = imutils.resize(frame, width=800)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

