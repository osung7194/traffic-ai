import streamlit as st
import cv2
import tempfile
import numpy as np
import datetime
import pandas as pd
from ultralytics import YOLO

# 페이지 설정
st.set_page_config(page_title="OSUNG TRAFFIC AI", layout="wide")
st.title("🚦 오성개발 트래픽 마스터 (Web 배포용)")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 1. 분석 기본 설정")
target_dir = st.sidebar.radio("방향 선택", ["▼ 하행 (위에서 아래로)", "▲ 상행 (아래서 위로)"])
line_y = st.sidebar.slider("📏 카운팅 라인 위치", 0, 720, 400)

with st.sidebar.expander("🛠️ 차량 크기 및 픽셀 설정"):
    pixels_per_meter = st.number_input("1m당 픽셀 수", value=80)
    th_car = st.slider("승용/소형화물 최대(m)", 3.0, 15.0, 5.5)
    th_small_bus = st.slider("소형버스/중형화물 최대(m)", 3.0, 15.0, 8.5)
    th_large = st.slider("대형 최소(m)", 5.0, 20.0, 9.5)

st.sidebar.header("📁 2. 동영상 업로드")
uploaded_file = st.sidebar.file_uploader("영상을 올려주세요 (200MB 제한)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    temp_file_path = tfile.name

    cap = cv2.VideoCapture(temp_file_path)
    ret, first_frame = cap.read()
    cap.release()

    if ret:
        first_frame = cv2.resize(first_frame, (1280, 720))
        preview_img = first_frame.copy()
        cv2.line(preview_img, (0, line_y), (1280, line_y), (0, 255, 255), 3)
        st.subheader("📺 첫 화면 미리보기 (노란 선을 조절하세요)")
        st.image(preview_img, channels="BGR")

        if st.button("▶ 본격적인 분석 시작", use_container_width=True):
            cap = cv2.VideoCapture(temp_file_path)
            col_vid, col_data = st.columns([3, 1])
            frame_window = col_vid.empty()
            status_area = col_data.empty()
            
            counts = {"승용차": 0, "소형버스": 0, "대형버스": 0, "소형화물": 0, "중형화물": 0, "대형화물": 0}
            track_data = {}
            records = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (1280, 720))
                
                results = model.track(frame, persist=True, verbose=False, classes=[2, 5, 7], conf=0.25)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    ids = results[0].boxes.id.int().cpu().tolist()
                    clss = results[0].boxes.cls.int().cpu().tolist()
                    
                    for box, tid, cls in zip(boxes, ids, clss):
                        bx, by, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        cx, cy = bx, by
                        
                        if tid not in track_data:
                            track_data[tid] = {'path': [], 'done': False}
                        path = track_data[tid]['path']
                        path.append((cx, cy))
                        
                        if not track_data[tid]['done'] and len(path) >= 2:
                            prev_y, curr_y = path[-2][1], path[-1][1]
                            if (prev_y <= line_y <= curr_y) or (curr_y <= line_y <= prev_y):
                                moving_down = (curr_y - path[0][1]) > 0
                                if ("하행" in target_dir and moving_down) or ("상행" in target_dir and not moving_down):
                                    pixel_len = max(bw, bh)
                                    real_len = float(pixel_len / pixels_per_meter)
                                    
                                    v_type = "승용차"
                                    if cls == 2: v_type = "승용차" if real_len < th_car else "중형화물"
                                    elif cls == 5: v_type = "소형버스" if real_len < th_small_bus else "대형버스"
                                    elif cls == 7:
                                        if real_len < th_car: v_type = "소형화물"
                                        elif real_len < th_small_bus: v_type = "중형화물"
                                        else: v_type = "대형화물"
                                    
                                    counts[v_type] += 1
                                    track_data[tid]['done'] = True
                                    records.append([datetime.datetime.now().strftime("%H:%M:%S"), v_type, round(real_len, 1)])

                # --- 에러 방지를 위해 글자 쓰기 기능을 최소화하고 선만 그립니다 ---
                cv2.line(frame, (0, line_y), (1280, line_y), (0, 255, 255), 2)
                frame_window.image(frame, channels="BGR")
                
                res_txt = "### 📊 카운팅 현황\n"
                for k, v in counts.items():
                    res_txt += f"- **{k}**: {v}대\n"
                status_area.markdown(res_txt)

            cap.release()
            st.success("✅ 분석 완료!")
            if records:
                df = pd.DataFrame(records, columns=["통과시간", "차종분류", "추정길이(m)"])
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 결과 다운로드", data=csv, file_name="result.csv")
else:
    st.info("👈 영상을 올려주세요.")
