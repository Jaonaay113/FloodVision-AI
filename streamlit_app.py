
import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="FloodVision AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# 2. API CONFIG
# ==========================================
API_URL = "https://floodvision-ai-2.onrender.com/predict"

MAX_FILE_MB = 8
MAX_IMAGE_SIDE = 2048
REQUEST_TIMEOUT = 180

# ==========================================
# 3. PURPLE & WHITE THEME
# ==========================================
st.markdown("""
<style>
.stApp {
    background: #F8F7FF;
    color: #35284D;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #4C347D !important;
    font-weight: 700;
}

p, label, .stMarkdown {
    color: #514568;
}

.hero {
    background: linear-gradient(135deg, #EDE7FF, #FFFFFF);
    border: 1px solid #E5DDF8;
    border-radius: 24px;
    padding: 30px 20px;
    margin-bottom: 25px;
    text-align: center;
}

.hero-title {
    color: #4C347D;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #75658E;
    font-size: 1rem;
    line-height: 1.8;
}

div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px solid #E5DDF8;
    border-radius: 16px;
    padding: 12px;
}

[data-testid="stFileUploaderDropzone"] {
    background: #FDFCFF;
    border: 2px dashed #C8B8EF;
    border-radius: 14px;
}

.stButton > button {
    background: #8B72D9;
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #7054BD;
    color: #FFFFFF;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8E0F7;
    padding: 18px;
    border-radius: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #75658E;
}

div[data-testid="stMetricValue"] {
    color: #4C347D;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stImage"] {
    background: #FFFFFF;
    border-radius: 14px;
}

hr {
    border-color: #E5DDF8;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: rgba(248, 247, 255, 0.8);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. HEADER
# ==========================================
st.markdown("""
<div class="hero">
    <div class="hero-title">FloodVision AI</div>
    <div class="hero-subtitle">
        AI-powered Flood Image Segmentation
        <br>
        วิเคราะห์และตรวจจับพื้นที่น้ำท่วมด้วยปัญญาประดิษฐ์
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("## วิเคราะห์ภาพน้ำท่วม")
st.write(
    "อัปโหลดภาพที่ต้องการตรวจสอบ "
    "แล้วกดปุ่มเพื่อให้ AI วิเคราะห์พื้นที่น้ำท่วม"
)

# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================
def prepare_image(uploaded_file):
    """
    ตรวจสอบภาพ แก้ EXIF orientation
    แปลงเป็น RGB และลดขนาดภาพที่ใหญ่เกินไป
    """
    raw_bytes = uploaded_file.getvalue()

    if not raw_bytes:
        raise ValueError("ไฟล์ภาพว่างเปล่า")

    if len(raw_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(
            f"ไฟล์ใหญ่เกิน {MAX_FILE_MB} MB "
            "กรุณาเลือกภาพที่มีขนาดเล็กลง"
        )

    with Image.open(BytesIO(raw_bytes)) as source:
        source.verify()

    with Image.open(BytesIO(raw_bytes)) as source:
        image = ImageOps.exif_transpose(source)
        image = image.convert("RGB")

    original_size = image.size

    # ลดเฉพาะภาพที่มีด้านใดด้านหนึ่งใหญ่เกินกำหนด
    image.thumbnail(
        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
        Image.Resampling.LANCZOS,
    )

    processed_size = image.size

    # ส่งเป็น JPEG มาตรฐาน
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)

    return image, buffer.getvalue(), original_size, processed_size


# ==========================================
# 6. API REQUEST
# ==========================================
def analyze_image(image_bytes):
    """
    ส่งภาพไปยัง Backend
    คืนผล JSON เมื่อ API ทำงานสำเร็จ
    """
    try:
        response = requests.post(
            API_URL,
            files={
                "file": (
                    "uploaded_image.jpg",
                    image_bytes,
                    "image/jpeg",
                )
            },
            timeout=REQUEST_TIMEOUT,
        )

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Backend ใช้เวลาวิเคราะห์นานเกินไป "
            "กรุณาลองอีกครั้ง"
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "ไม่สามารถเชื่อมต่อ Backend ได้ "
            "โปรดตรวจสอบสถานะ API"
        ) from exc

    if response.status_code == 502:
        raise RuntimeError(
            "Backend ตอบกลับ HTTP 502 "
            "อาจเกิดจากบริการหยุดทำงาน ใช้หน่วยความจำสูง "
            "หรือประมวลผลภาพไม่สำเร็จ "
            "โปรดลองใหม่และตรวจสอบ Render Logs"
        )

    if not response.ok:
        detail = response.text[:500]

        raise RuntimeError(
            f"Backend ตอบกลับ HTTP {response.status_code}\n"
            f"{detail}"
        )

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Backend ไม่ได้ส่งข้อมูล JSON กลับมา"
        ) from exc

    return result


# ==========================================
# 7. DECODE MASK
# ==========================================
def decode_mask(result):
    """
    รองรับชื่อฟิลด์ mask ที่เป็น Base64
    และรูปแบบ Data URL
    """
    mask_data = result.get("mask")

    if not mask_data:
        raise ValueError(
            "ไม่พบฟิลด์ 'mask' ในผลลัพธ์ API "
            "โปรดตรวจสอบชื่อฟิลด์ใน Backend"
        )

    if not isinstance(mask_data, str):
        raise ValueError("ข้อมูล Mask ไม่ใช่ข้อความ Base64")

    if "," in mask_data and mask_data.startswith("data:"):
        mask_data = mask_data.split(",", 1)[1]

    try:
        mask_bytes = base64.b64decode(
            mask_data,
            validate=True,
        )

        with Image.open(BytesIO(mask_bytes)) as mask_source:
            mask_source.load()
            mask_image = mask_source.copy()

        return mask_image

    except Exception as exc:
        raise ValueError(
            "ไม่สามารถอ่าน Mask ที่ Backend ส่งกลับมาได้"
        ) from exc


# ==========================================
# 8. UPLOAD IMAGE
# ==========================================
uploaded_file = st.file_uploader(
    "เลือกภาพที่ต้องการวิเคราะห์",
    type=["jpg", "jpeg", "png"],
    help="รองรับภาพ JPG, JPEG และ PNG ขนาดไม่เกิน 8 MB",
)

if uploaded_file is not None:

    st.markdown("### ภาพที่เลือก")

    try:
        preview, prepared_bytes, original_size, processed_size = (
            prepare_image(uploaded_file)
        )

        st.image(
            preview,
            caption="ภาพต้นฉบับหลังปรับทิศทางและสี",
            use_container_width=True,
        )

        if original_size != processed_size:
            st.caption(
                f"ปรับขนาดจาก {original_size[0]} × "
                f"{original_size[1]} เป็น "
                f"{processed_size[0]} × {processed_size[1]} "
                "ก่อนส่งวิเคราะห์"
            )

        st.caption(
            f"ขนาดไฟล์ต้นฉบับ: "
            f"{len(uploaded_file.getvalue()) / (1024 * 1024):.2f} MB"
        )

    except (ValueError, UnidentifiedImageError, OSError) as exc:
        st.error(f"ไม่สามารถเตรียมภาพได้: {exc}")
        st.stop()

    # ======================================
    # 9. ANALYZE BUTTON
    # ======================================
    if st.button(
        "วิเคราะห์พื้นที่น้ำท่วม",
        use_container_width=True,
        type="primary",
    ):
        try:
            with st.spinner(
                "กำลังส่งภาพไปวิเคราะห์ กรุณารอสักครู่..."
            ):
                result = analyze_image(prepared_bytes)

            mask_image = decode_mask(result)

            flood_percent = float(
                result.get("flood_percent", 0)
            )

            if not 0 <= flood_percent <= 100:
                raise ValueError(
                    "ค่า flood_percent ต้องอยู่ระหว่าง 0 ถึง 100"
                )

            summary = result.get(
                "summary",
                "วิเคราะห์ภาพเสร็จเรียบร้อย",
            )

            # ==================================
            # 10. SHOW RESULTS
            # ==================================
            st.markdown("---")
            st.markdown("## ผลการวิเคราะห์")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ภาพต้นฉบับ")
                st.image(
                    preview,
                    use_container_width=True,
                )

            with col2:
                st.markdown("### Mask ที่ AI ทำนาย")
                st.image(
                    mask_image,
                    use_container_width=True,
                )

            st.markdown("### 💧 สรุปผล")

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "พื้นที่น้ำท่วมที่ตรวจพบ",
                    f"{flood_percent:.2f}%",
                )

            with metric2:
                st.metric(
                    "ขนาดภาพที่วิเคราะห์",
                    f"{processed_size[0]} × {processed_size[1]}",
                )

            st.info(str(summary))

            st.caption(
                "หมายเหตุ: ผลลัพธ์เป็นการประมาณจากโมเดล AI "
                "ควรตรวจสอบกับภาพจริงก่อนนำไปใช้ตัดสินใจ"
            )

        except RuntimeError as exc:
            st.error(str(exc))

            with st.expander("แนวทางตรวจสอบ"):
                st.markdown("""
                1. ตรวจสอบสถานะ Backend บน Render
                2. เปิด Logs ของ Backend เพื่อดู Error
                3. ทดลองภาพเดิมอีกครั้ง
                4. หากเกิดเฉพาะบางภาพ ให้ตรวจขนาดและรูปแบบภาพ
                """)

        except (ValueError, TypeError, OSError) as exc:
            st.error(f"เกิดปัญหากับผลลัพธ์จาก API: {exc}")

        except Exception as exc:
            st.error("เกิดข้อผิดพลาดที่ไม่คาดคิด")
            st.caption(str(exc))

else:
    st.markdown("""
    <div style="
        background:#FFFFFF;
        border:1px solid #E8E0F7;
        border-radius:18px;
        padding:24px;
        margin-top:20px;
    ">
        <h3> วิธีใช้งาน</h3>
        <p>
            1. เลือกภาพที่ต้องการตรวจสอบ
            <br>
            2. กดปุ่มวิเคราะห์พื้นที่น้ำท่วม
            <br>
            3. เปรียบเทียบภาพต้นฉบับกับ Mask ที่ AI ทำนาย
            <br>
            4. ตรวจสอบเปอร์เซ็นต์พื้นที่น้ำท่วม
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 11. FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    """
    <p style="
        text-align:center;
        color:#8B7BA5;
        font-size:0.9rem;
    ">
        FloodVision AI · Flood Image Segmentation
    </p>
    """,
    unsafe_allow_html=True,
)
