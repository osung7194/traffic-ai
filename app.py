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

# 모델 로드 (캐싱하여 속도 최적화)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ==========================================
# ⚙️ 항상 표시되는 사이드바 설정 메뉴
# ==========================================
st.sidebar.header("⚙️ 1. 분석 기본 설정")

# 1. 상행/하행 선택버튼 (항상 보임)
target_dir = st.sidebar.radio("방향 선택 (통과 기준)", ["▼ 하행 (위에서 아래로)", "▲ 상행 (아래서 위로)"])

# 2. 카운팅 라인 조절 (화면 픽셀 0~720 기준)
line_y = st.sidebar.slider("📏 카운팅 라인 위치 (위/아래)", 0, 720, 400)

# 3. 차량 크기 설정 (숨김 메뉴로 깔끔하게)
with st.sidebar.expander("🛠️ 차량 크기 및 픽셀 설정 (클릭)"):
    pixels_per_meter = st.number_input("1m당 픽셀 수 (기본 80)", value=80)
    th_car = st.slider("승용/소형화물 최대(m)", 3.0, 15.0, 5.5)
    th_small_bus = st.slider("소형버스/중형화물 최대(m)", 3.0, 15.0, 8.5)
    th_large = st.slider("대형 최소(m)", 5.0, 20.0, 9.5)

st.sidebar.markdown("---")

# ==========================================
# 📁 동영상 업로드 및 실행 영역
# ==========================================
st.sidebar.header("📁 2. 동영상 업로드")
uploaded_file = st.sidebar.file_uploader("영상을 올려주세요 (최대 2GB)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 영상 임시 저장
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    temp_file_path = tfile.name

    # 첫 화면 미리보기 추출
    cap = cv2.VideoCapture(temp_file_path)
    ret, first_frame = cap.read()
    cap.release()

    if ret:
        first_frame = cv2.resize(first_frame, (1280, 720))
        preview_img = first_frame.copy()
        
        # 라인 그리기
        cv2.line(preview_img, (0, line_y), (1280, line_y), (0, 255, 255), 3)
        cv2.putText(preview_img, "COUNTING LINE", (10, line_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        st.subheader("📺 첫 화면 미리보기 (왼쪽 설정바에서 노란 선을 조절하세요)")
        st.image(preview_img, channels="BGR")

        st.markdown("---")
        
        # 시작 버튼
        if st.button("▶ 본격적인 분석 시작 (클릭)", use_container_width=True):
            cap = cv2.VideoCapture(temp_file_path)
            
            # 실시간 화면과 데이터 창 나누기
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
                
                # YOLO AI 추적
                results = model.track(frame, persist=True, verbose=False, classes=[2, 5, 7], conf=0.25)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    ids = results[0].boxes.id.int().cpu().tolist()
                    clss = results[0].boxes.cls.int().cpu().tolist()
                    
                    for box, tid, cls in zip(boxes, ids, clss):
                        # [오류 해결] Tensor를 float 숫자로 완전 변환
                        bx, by, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        cx, cy = bx, by
                        
                        if tid not in track_data:
                            track_data[tid] = {'path': [], 'done': False}
                        
                        path = track_data[tid]['path']
                        path.append((cx, cy))
                        
                        if not track_data[tid]['done'] and len(path) >= 2:
                            prev_y, curr_y = path[-2][1], path[-1][1]
                            
                            # 선 통과 확인
                            if (prev_y <= line_y <= curr_y) or (curr_y <= line_y <= prev_y):
                                moving_down = (curr_y - path[0][1]) > 0
                                
                                # 방향 일치 확인
                                if ("하행" in target_dir and moving_down) or ("상행" in target_dir and not moving_down):
                                    
                                    pixel_len = max(bw, bh)
                                    real_len = float(pixel_len / pixels_per_meter)
                                    
                                    # 6종 분류
                                    v_type = "승용차"
                                    if cls == 2:
                                        v_type = "승용차" if real_len < th_car else "중형화물"
                                    elif cls == 5:
                                        v_type = "소형버스" if real_len < th_small_bus else "대형버스"
                                    elif cls == 7:
                                        if real_len < th_car: v_type = "소형화물"
                                        elif real_len < th_small_bus: v_type = "중형화물"
                                        else: v_type = "대형화물"
                                    
                                    counts[v_type] += 1
                                    track_data[tid]['done'] = True
                                    records.append([datetime.datetime.now().strftime("%H:%M:%S"), v_type, round(real_len, 1)])

                # 화면에 선 그리기
                cv2.line(frame, (0, line_y), (1280, line_y), (0, 255, 255), 3)
                frame_window.image(frame, channels="BGR")
                
                # 실시간 현황판
                res_txt = "### 📊 카운팅 현황\n"
                for k, v in counts.items():
                    res_txt += f"- **{k}**: {v}대\n"
                status_area.markdown(res_txt)

            cap.release()
            
            # --- 분석 종료 후 다운로드 ---
            st.success("✅ 분석 완료! 아래 버튼을 눌러 결과 엑셀(CSV)을 저장하세요.")
            if records:
                df = pd.DataFrame(records, columns=["통과시간", "차종분류", "추정길이(m)"])
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 분석 결과 (엑셀 CSV) 다운로드", data=csv, file_name="교통량결과.csv", mime="text/csv", use_container_width=True)
else:
    st.info("👈 왼쪽에서 분석할 방향, 라인 위치를 먼저 설정하고 동영상을 올려주세요.")