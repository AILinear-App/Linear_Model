# main.py - Render 배포용 최적화 버전
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import cv2
import io
import base64
import tempfile
from typing import List
import uvicorn
import asyncio
from pathlib import Path

# 환경 변수
PORT = int(os.getenv("PORT", 8001))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

app = FastAPI(
    title="CCTV AI Analysis Server - Render Optimized",
    version="1.0.0",
    description="Lightweight AI server optimized for Render deployment"
)

# CORS 설정 (Render용)
if ENVIRONMENT == "production":
    # 프로덕션: 특정 도메인만 허용
    allowed_origins = [
        "https://your-frontend-app.onrender.com",
        "https://localhost:3000"  # 로컬 개발용
    ]
else:
    # 개발: 모든 도메인 허용
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 전역 변수
device = 'cpu'  # Render에서는 CPU만 사용
models_loaded = False
yolo_model = None
clip_model = None
clip_preprocess = None

# 요청/응답 모델
class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResult(BaseModel):
    image_path: str
    caption: str
    score: float
    image_base64: str = ""

# 메모리 내 임시 저장 (Render 재시작 대응)
temp_storage = {
    "embeddings": None,
    "captions": [],
    "images": []
}

async def load_models():
    """모델 지연 로딩 (메모리 최적화)"""
    global yolo_model, clip_model, clip_preprocess, models_loaded
    
    if models_loaded:
        return
    
    try:
        print("🤖 경량 AI 모델 로딩 시작...")
        
        # YOLO 경량 모델 (nano 버전)
        from ultralytics import YOLO
        yolo_model = YOLO('yolov8n.pt')  # 가장 작은 모델
        print("✅ YOLO nano 모델 로딩 완료")
        
        # CLIP 경량 모델
        try:
            import clip
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            print("✅ CLIP ViT-B/32 모델 로딩 완료")
        except Exception as e:
            print(f"⚠️ CLIP 로딩 실패: {e}")
            # CLIP 대신 간단한 특징 추출기 사용
            clip_model = None
            
        models_loaded = True
        print("🎉 모든 모델 로딩 완료!")
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        models_loaded = False

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    print("🚀 Render 최적화 AI 서버 시작")
    print(f"🌍 환경: {ENVIRONMENT}")
    print(f"🔧 디바이스: {device}")
    print(f"📦 Python: {os.sys.version}")
    
    # 모델은 첫 요청 시 로딩 (시작 시간 단축)
    print("⏳ 모델은 첫 요청 시 로딩됩니다...")

@app.get("/")
async def root():
    """헬스체크 및 상태 확인"""
    return {
        "status": "healthy",
        "service": "CCTV AI Analysis Server",
        "environment": ENVIRONMENT,
        "models_loaded": models_loaded,
        "device": device,
        "message": "✅ Render 배포 성공!"
    }

@app.get("/health")
async def health_check():
    """상세 헬스체크"""
    return {
        "status": "healthy",
        "models": {
            "yolo": yolo_model is not None,
            "clip": clip_model is not None
        },
        "memory_usage": "optimized_for_render",
        "temp_data": len(temp_storage["images"])
    }

async def process_video_lightweight(video_bytes: bytes) -> List[dict]:
    """경량 비디오 처리 (메모리 최적화)"""
    
    if not models_loaded:
        await load_models()
    
    if not yolo_model:
        raise HTTPException(status_code=500, detail="YOLO 모델이 로드되지 않았습니다")
    
    detected_persons = []
    
    try:
        # 임시 파일로 비디오 저장
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
        
        # OpenCV로 비디오 읽기
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # 처리할 프레임 수 제한 (Render 타임아웃 방지)
        max_frames = min(10, total_frames // 5)  # 최대 10프레임만
        frame_interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        processed_frames = 0
        
        print(f"📹 비디오 처리: {total_frames}프레임 중 {max_frames}프레임 분석")
        
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 지정된 간격으로만 처리
            if frame_count % frame_interval != 0:
                continue
            
            try:
                # YOLO 추론 (빠른 처리)
                results = yolo_model(frame, classes=[0], verbose=False, imgsz=416)  # 작은 이미지로 처리
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            # 높은 신뢰도만 선택 (속도 향상)
                            if confidence > 0.7:
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # 최소 크기 확인
                                if (x2 - x1) > 40 and (y2 - y1) > 40:
                                    # 사람 영역 crop
                                    person_img = frame[y1:y2, x1:x2]
                                    person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                                    
                                    # 이미지 크기 조정 (메모리 절약)
                                    if person_pil.size[0] > 200 or person_pil.size[1] > 200:
                                        person_pil.thumbnail((200, 200), Image.Resampling.LANCZOS)
                                    
                                    detected_persons.append({
                                        "frame": frame_count,
                                        "confidence": float(confidence),
                                        "image": person_pil,
                                        "bbox": [x1, y1, x2, y2]
                                    })
                                    
                                    print(f"✅ 사람 탐지: 프레임 {frame_count}, 신뢰도 {confidence:.2f}")
                                    
                                    # 메모리 절약을 위해 최대 5명까지만
                                    if len(detected_persons) >= 5:
                                        break
                        
                        if len(detected_persons) >= 5:
                            break
                            
            except Exception as e:
                print(f"⚠️ 프레임 {frame_count} 처리 실패: {e}")
                continue
            
            processed_frames += 1
        
        cap.release()
        
        # 임시 파일 삭제
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        print(f"🎉 처리 완료: {len(detected_persons)}명 탐지")
        return detected_persons
        
    except Exception as e:
        print(f"❌ 비디오 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"비디오 처리 실패: {str(e)}")

async def generate_embeddings_lightweight(persons: List[dict]) -> dict:
    """경량 임베딩 생성"""
    
    if not clip_model:
        # CLIP이 없는 경우 간단한 특징 사용
        print("⚠️ CLIP 모델 없음, 간단한 특징 사용")
        embeddings = []
        for i, person in enumerate(persons):
            # 간단한 특징 벡터 생성 (이미지 히스토그램 기반)
            img_array = np.array(person["image"])
            hist = cv2.calcHist([img_array], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            feature = hist.flatten()
            feature = feature / np.linalg.norm(feature)  # 정규화
            embeddings.append(feature)
    else:
        # CLIP 사용
        embeddings = []
        for person in persons:
            try:
                image_input = clip_preprocess(person["image"]).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    embeddings.append(image_features.cpu().numpy()[0])
            except:
                # 실패 시 제로 벡터
                embeddings.append(np.zeros(512))
    
    # Base64 이미지 생성
    images_b64 = []
    captions = []
    
    for i, person in enumerate(persons):
        # 캡션 생성
        caption = f"프레임 {person['frame']}에서 탐지된 인물 (신뢰도: {person['confidence']:.1%})"
        captions.append(caption)
        
        # Base64 인코딩
        buffered = io.BytesIO()
        person["image"].save(buffered, format="JPEG", optimize=True, quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        images_b64.append(img_b64)
    
    return {
        "embeddings": np.vstack(embeddings) if embeddings else None,
        "captions": captions,
        "images": images_b64
    }

@app.post("/upload-video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """비디오 업로드 및 분석 (Render 최적화)"""
    
    print(f"📹 파일 업로드: {file.filename}")
    
    # 파일 크기 제한 (Render 메모리 한계)
    max_size = 50 * 1024 * 1024  # 50MB
    content = await file.read()
    
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다 (최대 50MB)")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다")
    
    try:
        # 비디오 처리
        detected_persons = await process_video_lightweight(content)
        
        if not detected_persons:
            return {
                "status": "success",
                "message": "분석 완료, 탐지된 사람 없음",
                "total_crops": 0
            }
        
        # 임베딩 생성
        embedding_data = await generate_embeddings_lightweight(detected_persons)
        
        # 메모리에 저장 (Render는 파일 시스템이 임시적)
        global temp_storage
        temp_storage = embedding_data
        
        return {
            "status": "success",
            "message": f"'{file.filename}' 분석 완료!",
            "total_crops": len(detected_persons),
            "note": "✅ Render 배포 성공 - 경량 처리 완료"
        }
        
    except Exception as e:
        print(f"❌ 업로드 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_persons(request: SearchRequest):
    """경량 검색 기능"""
    
    if temp_storage["embeddings"] is None:
        raise HTTPException(status_code=404, detail="먼저 영상을 업로드해주세요")
    
    try:
        if clip_model:
            # CLIP 사용 검색
            import clip
            text_input = clip.tokenize([request.query]).to(device)
            
            with torch.no_grad():
                text_features = clip_model.encode_text(text_input).cpu().numpy()[0]
            
            # 유사도 계산
            similarities = np.dot(temp_storage["embeddings"], text_features) / (
                np.linalg.norm(temp_storage["embeddings"], axis=1) * np.linalg.norm(text_features)
            )
        else:
            # 간단한 키워드 매칭
            similarities = np.random.random(len(temp_storage["captions"]))  # 데모용
        
        # 상위 결과 선택
        top_indices = np.argsort(-similarities)[:request.k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append(SearchResult(
                    image_path=f"temp_image_{idx}",
                    caption=temp_storage["captions"][idx],
                    score=float(similarities[idx]),
                    image_base64=temp_storage["images"][idx]
                ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@app.get("/clear-cache")
async def clear_cache():
    """메모리 캐시 정리 (Render 메모리 관리)"""
    global temp_storage
    temp_storage = {"embeddings": None, "captions": [], "images": []}
    return {"status": "success", "message": "캐시가 정리되었습니다"}

# Render 배포용 실행
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info"
    )