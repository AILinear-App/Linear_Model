# Render 최적화 버전 - 메모리 절약 + 안정성 개선
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import shutil
from typing import List, Dict, Any
import uvicorn
import base64
import io
import time
import uuid
from datetime import datetime
import logging
import gc  # 메모리 관리용

# 환경 변수
PORT = int(os.getenv("PORT", 8001))
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")  # Render는 기본 production

# Render 메모리 제한 고려한 설정
MEMORY_LIMIT_MODE = os.getenv("RENDER_MEMORY_LIMIT", "true").lower() == "true"

app = FastAPI(
    title="CCTV AI Analysis Server - Render Optimized",
    version="3.1.0",
    description="Render 최적화된 AI 서버"
)

# CORS 설정 (Render 배포용)
# CORS 설정 (Render 배포용) - 수정된 부분만
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-linear.vercel.app",  # 🆕 Vercel 프론트엔드 URL 추가
        "https://ai-linear-parkmoonas-projects.vercel.app",  # 🆕 추가 Vercel 도메인
        "http://localhost:3000",  # 로컬 개발용
        "http://localhost:5173",  # Vite 개발 서버
        "*"  # 임시로 모든 도메인 허용 (개발용)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 저장 폴더 (Render 임시 디스크 사용)
ROOT_DIR = Path('/tmp/cctv_data')  # Render에서는 /tmp 사용
VIDEO_DIR = ROOT_DIR / 'videos'
CROP_DIR = ROOT_DIR / 'crops'

# 폴더 생성
for folder in [ROOT_DIR, VIDEO_DIR, CROP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 전역 변수 - Lazy Loading으로 메모리 절약
device = 'cpu'  # Render는 CPU만 지원
yolo_model = None
clip_model = None
clip_preprocess = None

# Render 최적화 설정 - 사람 탐지 개선
render_config = {
    "yolo_confidence_threshold": 0.3,  # 🔥 0.6 → 0.3 (더 민감하게)
    "yolo_model_size": "yolov8n",  # 가장 작은 모델 유지
    "max_frames_per_video": 20,        # 🔥 10 → 20 (더 많은 프레임)
    "min_person_size": 30,             # 🔥 80 → 30 (작은 사람도 탐지)
    "search_similarity_threshold": 0.1, # 🔥 0.15 → 0.1 (더 민감하게)
    "max_video_size_mb": 50            # 최대 비디오 크기 제한 유지
}

# 요청/응답 모델
class SearchRequest(BaseModel):
    query: str
    k: int = 3  # 적은 결과로 메모리 절약

class SearchResult(BaseModel):
    image_path: str
    caption: str
    score: float
    image_base64: str = ""

# 분석된 데이터 저장 (메모리 제한)
analyzed_data = {
    "embeddings": None,
    "image_paths": [],
    "captions": [],
    "images": [],
    "last_analysis": None
}

def load_yolo_model():
    """YOLO 모델 지연 로딩"""
    global yolo_model
    if yolo_model is None:
        try:
            print("📦 YOLO 모델 로딩...")
            from ultralytics import YOLO
            yolo_model = YOLO(f"{render_config['yolo_model_size']}.pt")
            print("✅ YOLO 로딩 완료")
        except Exception as e:
            print(f"❌ YOLO 로딩 실패: {e}")
            raise
    return yolo_model

def load_clip_model():
    """CLIP 모델 지연 로딩"""
    global clip_model, clip_preprocess
    if clip_model is None:
        try:
            print("🧠 CLIP 모델 로딩...")
            import clip
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            print("✅ CLIP 로딩 완료")
        except Exception as e:
            print(f"❌ CLIP 로딩 실패: {e}")
            raise
    return clip_model, clip_preprocess

def cleanup_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.on_event("startup")
async def startup_event():
    """서버 시작 - 모델은 필요할 때 로딩"""
    print("🚀 Render 최적화 AI 서버 시작!")
    print(f"🌍 환경: {ENVIRONMENT}")
    print(f"🔧 디바이스: {device}")
    print(f"💾 메모리 제한 모드: {MEMORY_LIMIT_MODE}")
    print("📝 모델은 필요할 때 로딩됩니다 (메모리 절약)")

@app.get("/")
async def root():
    """메인 페이지"""
    return {
        "service": "CCTV AI Analysis Server - Render Optimized",
        "version": "3.1.0",
        "environment": ENVIRONMENT,
        "memory_limit_mode": MEMORY_LIMIT_MODE,
        "models_loaded": {
            "yolo": yolo_model is not None,
            "clip": clip_model is not None
        },
        "config": render_config,
        "message": "✅ Render 최적화 서버 실행중! 💪"
    }

@app.get("/health")
async def health_check():
    """상세 헬스체크"""
    return {
        "status": "healthy",
        "platform": "Render",
        "models_loaded": {
            "yolo": yolo_model is not None,
            "clip": clip_model is not None
        },
        "device": device,
        "analyzed_data_count": len(analyzed_data["image_paths"]),
        "memory_mode": "optimized" if MEMORY_LIMIT_MODE else "normal"
    }

def detect_persons_in_video_optimized(video_path: Path, max_frames: int = 20):
    """개선된 사람 탐지 - 더 민감하게"""
    
    model = load_yolo_model()
    print(f"🎬 개선된 비디오 분석 시작: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detected_persons = []
    frame_count = 0
    
    # 🔥 프레임 간격을 더 작게 (더 많은 프레임 분석)
    frame_interval = max(2, total_frames // max_frames)  # 최소 2프레임마다
    
    print(f"📊 총 프레임: {total_frames}, 분석할 프레임: {min(max_frames, total_frames // frame_interval)}")
    print(f"🎯 탐지 임계값: {render_config['yolo_confidence_threshold']}")
    print(f"📏 최소 사람 크기: {render_config['min_person_size']}px")
    
    try:
        while len(detected_persons) < 30:  # 🔥 20 → 30개로 증가
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % frame_interval != 0:
                continue
            
            # 프레임 크기는 유지 (너무 작게 하면 탐지 안됨)
            height, width = frame.shape[:2]
            if width > 800:  # 🔥 640 → 800으로 증가
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            try:
                # YOLO 추론 - 더 민감하게
                results = model(frame, classes=[0], verbose=False, conf=render_config['yolo_confidence_threshold'])
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        print(f"🔍 프레임 {frame_count}에서 {len(boxes)}개 객체 감지")
                        
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            print(f"   📦 박스: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), 신뢰도: {confidence:.2f}")
                            
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            width_box = x2 - x1
                            height_box = y2 - y1
                            
                            print(f"   📐 박스 크기: {width_box}x{height_box}")
                            
                            # 🔥 크기 조건 완화
                            if width_box > render_config["min_person_size"] and height_box > render_config["min_person_size"]:
                                person_img = frame[y1:y2, x1:x2]
                                person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                                
                                # 이미지 크기 제한 완화
                                if person_pil.size[0] > 400 or person_pil.size[1] > 400:
                                    person_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)
                                
                                filename = f"{video_path.stem}_f{frame_count}_p{len(detected_persons)}.jpg"
                                crop_path = CROP_DIR / filename
                                person_pil.save(crop_path, quality=90)  # 🔥 85 → 90 품질 향상
                                
                                detected_persons.append({
                                    "파일경로": str(crop_path),
                                    "프레임번호": frame_count,
                                    "신뢰도": float(confidence),
                                    "박스좌표": [x1, y1, x2, y2],
                                    "이미지": person_pil
                                })
                                
                                print(f"✅ 사람 탐지 성공: 프레임 {frame_count}, 신뢰도 {confidence:.2f}")
                            else:
                                print(f"❌ 크기 부족: {width_box}x{height_box} < {render_config['min_person_size']}")
                    else:
                        print(f"🔍 프레임 {frame_count}에서 객체 없음")
                        
                # 메모리 정리는 더 자주
                if frame_count % 5 == 0:
                    cleanup_memory()
            
            except Exception as e:
                print(f"⚠️ 프레임 {frame_count} 분석 실패: {e}")
                continue
    
    finally:
        cap.release()
        cleanup_memory()
    
    print(f"🎉 분석 완료: {len(detected_persons)}명의 사람 탐지")
    return detected_persons

def generate_clip_embeddings_optimized(detected_persons):
    """메모리 최적화된 CLIP 임베딩 생성"""
    
    clip_model, clip_preprocess = load_clip_model()
    print("🧠 CLIP 임베딩 생성 중...")
    
    embeddings = []
    image_paths = []
    captions = []
    images = []
    
    for i, person in enumerate(detected_persons):
        try:
            image = person["이미지"]
            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                image_features = image_features.cpu().numpy()[0]
            
            embeddings.append(image_features)
            image_paths.append(person["파일경로"])
            
            caption = f"프레임 {person['프레임번호']}에서 탐지된 인물 (신뢰도: {person['신뢰도']:.1%})"
            captions.append(caption)
            
            # Base64 인코딩 (크기 제한)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=80)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_base64)
            
            # 주기적 메모리 정리
            if i % 3 == 0:
                cleanup_memory()
            
        except Exception as e:
            print(f"⚠️ 임베딩 생성 실패 {i}: {e}")
            continue
    
    cleanup_memory()
    print(f"✅ {len(embeddings)}개의 임베딩 생성 완료")
    
    return {
        "embeddings": np.vstack(embeddings) if embeddings else None,
        "image_paths": image_paths,
        "captions": captions,
        "images": images
    }

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Render 최적화된 영상 업로드 및 AI 분석"""
    
    print(f"📹 Render AI 분석 시작: {file.filename}")
    
    # 파일 크기 체크
    file_size = 0
    content = await file.read()
    file_size = len(content) / 1024 / 1024  # MB
    
    if file_size > render_config["max_video_size_mb"]:
        raise HTTPException(
            status_code=413, 
            detail=f"파일이 너무 큽니다. 최대 {render_config['max_video_size_mb']}MB까지 지원합니다."
        )
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식")
    
    try:
        # 파일 저장
        video_path = VIDEO_DIR / file.filename
        with open(video_path, "wb") as buffer:
            buffer.write(content)
        
        print(f"📁 파일 저장됨: {file_size:.1f}MB")
        
        # 사람 탐지 (최적화됨)
        detected_persons = detect_persons_in_video_optimized(
            video_path, 
            render_config["max_frames_per_video"]
        )
        
        if not detected_persons:
            # 임시 파일 정리
            video_path.unlink(missing_ok=True)
            return {
                "status": "success",
                "message": "분석 완료, 탐지된 사람 없음",
                "total_crops": 0
            }
        
        # CLIP 임베딩 생성 (최적화됨)
        embedding_data = generate_clip_embeddings_optimized(detected_persons)
        
        # 글로벌 데이터 업데이트
        global analyzed_data
        analyzed_data = embedding_data
        analyzed_data["last_analysis"] = datetime.now().isoformat()
        
        # 비디오 파일 삭제 (메모리 절약)
        video_path.unlink(missing_ok=True)
        
        cleanup_memory()
        
        return {
            "status": "success",
            "message": f"'{file.filename}' Render AI 분석 완료! 🎉",
            "total_crops": len(detected_persons),
            "file_size_mb": round(file_size, 1),
            "분석결과": f"{len(detected_persons)}명의 실제 인물이 AI로 탐지되었습니다"
        }
        
    except Exception as e:
        # 오류 시 임시 파일 정리
        if 'video_path' in locals():
            video_path.unlink(missing_ok=True)
        
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_persons(request: SearchRequest):
    """Render 최적화된 CLIP 기반 검색"""
    
    if analyzed_data["embeddings"] is None:
        raise HTTPException(status_code=404, detail="먼저 영상을 업로드하고 분석해주세요")
    
    try:
        # CLIP 모델 로드
        clip_model, _ = load_clip_model()
        
        # CLIP 텍스트 인코딩
        import clip
        text_input = clip.tokenize([request.query]).to(device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(text_input)
            text_features = text_features.cpu().numpy()[0]
        
        # 유사도 계산
        image_embeddings = analyzed_data["embeddings"]
        similarities = np.dot(image_embeddings, text_features) / (
            np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(text_features)
        )
        
        # 상위 결과 선택 (제한됨)
        top_indices = np.argsort(-similarities)[:min(request.k, 5)]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity > render_config["search_similarity_threshold"]:
                results.append(SearchResult(
                    image_path=analyzed_data["image_paths"][idx],
                    caption=analyzed_data["captions"][idx],
                    score=float(similarity),
                    image_base64=analyzed_data["images"][idx]
                ))
        
        cleanup_memory()
        print(f"🔍 Render AI 검색 완료: {len(results)}개 결과")
        return results
        
    except Exception as e:
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Render 서버 통계"""
    return {
        "서버_상태": "Render 최적화 실행중",
        "플랫폼": "Render",
        "모델_로딩": {
            "YOLO": yolo_model is not None,
            "CLIP": clip_model is not None
        },
        "설정": render_config,
        "분석된_데이터": len(analyzed_data["image_paths"]),
        "마지막_분석": analyzed_data.get("last_analysis", "없음"),
        "메모리_모드": "최적화됨" if MEMORY_LIMIT_MODE else "일반"
    }

@app.get("/image/{filename}")
async def get_image(filename: str):
    """이미지 파일 반환"""
    image_path = CROP_DIR / filename
    if image_path.exists():
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

# 서버 실행
if __name__ == "__main__":
    print("🚀 Render 최적화 AI 서버 시작!")
    print("💪 7달러의 가치를 보여드리겠습니다!")
    print("=" * 50)
    print(f"📍 서버: http://localhost:{PORT}")
    print(f"💾 메모리 최적화: {MEMORY_LIMIT_MODE}")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)