# main.py - 실제 AI가 포함된 서버 (YOLO + CLIP)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import clip
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from ultralytics import YOLO
import shutil
from typing import List
import uvicorn
import json
import base64
import io

# FastAPI 앱 생성
app = FastAPI(title="CCTV AI 분석 서버 (실제 AI)", version="2.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 저장 폴더
ROOT_DIR = Path('./cctv_data')
VIDEO_DIR = ROOT_DIR / 'videos'
CROP_DIR = ROOT_DIR / 'crops'
DATA_DIR = ROOT_DIR / 'data'

# 폴더 생성
for folder in [ROOT_DIR, VIDEO_DIR, CROP_DIR, DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 전역 변수 (AI 모델들)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = None
clip_model = None
clip_preprocess = None

print(f"🔧 사용 중인 디바이스: {device}")

# 요청/응답 모델
class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResult(BaseModel):
    image_path: str
    caption: str
    score: float
    image_base64: str = ""

# 분석된 데이터 저장
analyzed_data = {
    "embeddings": None,
    "image_paths": [],
    "captions": [],
    "images": []
}

@app.on_event("startup")
async def startup_event():
    """서버 시작시 AI 모델 로드"""
    global yolo_model, clip_model, clip_preprocess
    
    try:
        print("🤖 AI 모델 로딩 중...")
        
        # YOLO 모델 로드 (사람 탐지용)
        print("📦 YOLO 모델 다운로드 중...")
        yolo_model = YOLO('yolov8n.pt')  # nano 버전 (빠르고 가벼움)
        
        # CLIP 모델 로드 (이미지-텍스트 매칭용)
        print("🧠 CLIP 모델 로딩 중...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        
        print("✅ 모든 AI 모델 로딩 완료!")
        print(f"   - YOLO: 사람 탐지 준비완료")
        print(f"   - CLIP: 자연어 검색 준비완료")
        print(f"   - 디바이스: {device}")
        
    except Exception as e:
        print(f"❌ AI 모델 로딩 실패: {e}")
        print("⚠️  기본 모드로 실행됩니다")

@app.get("/")
async def 메인페이지():
    """서버 메인 페이지"""
    return {
        "메시지": "🤖 실제 AI 분석 서버가 실행중입니다!",
        "YOLO_모델": "로딩됨" if yolo_model else "로딩실패",
        "CLIP_모델": "로딩됨" if clip_model else "로딩실패",
        "디바이스": device,
        "분석된_이미지": len(analyzed_data["image_paths"])
    }

@app.get("/health")
async def 상태확인():
    """서버 상태 확인"""
    return {
        "상태": "정상",
        "AI_준비": all([yolo_model, clip_model]),
        "디바이스": device,
        "분석된_데이터": len(analyzed_data["image_paths"])
    }

def detect_persons_in_video(video_path: Path, max_frames: int = 30):
    """비디오에서 사람 탐지 및 crop 이미지 생성"""
    
    if not yolo_model:
        raise Exception("YOLO 모델이 로드되지 않았습니다")
    
    print(f"🎬 비디오 분석 시작: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    detected_persons = []
    frame_count = 0
    
    # 프레임 간격 계산 (전체 영상에서 max_frames개만 분석)
    frame_interval = max(1, total_frames // max_frames)
    
    print(f"📊 총 프레임: {total_frames}, 분석할 프레임: {min(max_frames, total_frames)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 지정된 간격으로만 분석
        if frame_count % frame_interval != 0:
            continue
            
        try:
            # YOLO로 사람 탐지
            results = yolo_model(frame, classes=[0], verbose=False)  # class 0 = person
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # 신뢰도가 0.5 이상인 경우만
                        if confidence > 0.5:
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # 너무 작은 박스 제외
                            if (x2 - x1) > 50 and (y2 - y1) > 50:
                                # 사람 영역 crop
                                person_img = frame[y1:y2, x1:x2]
                                
                                # 이미지를 PIL로 변환
                                person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                                
                                # 파일명 생성
                                filename = f"{video_path.stem}_frame{frame_count}_person{len(detected_persons)}.jpg"
                                crop_path = CROP_DIR / filename
                                
                                # 이미지 저장
                                person_pil.save(crop_path)
                                
                                detected_persons.append({
                                    "파일경로": str(crop_path),
                                    "프레임번호": frame_count,
                                    "신뢰도": float(confidence),
                                    "박스좌표": [x1, y1, x2, y2],
                                    "이미지": person_pil
                                })
                                
                                print(f"✅ 사람 탐지: 프레임 {frame_count}, 신뢰도 {confidence:.2f}")
        
        except Exception as e:
            print(f"⚠️ 프레임 {frame_count} 분석 실패: {e}")
            continue
    
    cap.release()
    print(f"🎉 분석 완료: {len(detected_persons)}명의 사람 탐지")
    
    return detected_persons

def generate_clip_embeddings(detected_persons):
    """탐지된 사람들의 CLIP 임베딩 생성"""
    
    if not clip_model:
        raise Exception("CLIP 모델이 로드되지 않았습니다")
    
    print("🧠 CLIP 임베딩 생성 중...")
    
    embeddings = []
    image_paths = []
    captions = []
    images = []
    
    for i, person in enumerate(detected_persons):
        try:
            # 이미지 전처리
            image = person["이미지"]
            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            
            # CLIP 임베딩 생성
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                image_features = image_features.cpu().numpy()[0]
            
            embeddings.append(image_features)
            image_paths.append(person["파일경로"])
            
            # 기본 캡션 생성
            caption = f"프레임 {person['프레임번호']}에서 탐지된 인물 (신뢰도: {person['신뢰도']:.1%})"
            captions.append(caption)
            
            # Base64 인코딩을 위해 이미지 저장
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_base64)
            
        except Exception as e:
            print(f"⚠️ 임베딩 생성 실패 {i}: {e}")
            continue
    
    print(f"✅ {len(embeddings)}개의 임베딩 생성 완료")
    
    return {
        "embeddings": np.vstack(embeddings) if embeddings else None,
        "image_paths": image_paths,
        "captions": captions,
        "images": images
    }

@app.post("/upload-video")
async def 영상업로드(file: UploadFile = File(...)):
    """CCTV 영상 업로드 및 실제 AI 분석"""
    
    print(f"📹 업로드 시작: {file.filename}")
    
    # 파일 형식 체크
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다")
    
    try:
        # 파일 저장
        video_path = VIDEO_DIR / file.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"💾 파일 저장 완료: {video_path}")
        
        # 실제 AI 분석 수행
        if not yolo_model or not clip_model:
            return {
                "status": "warning",
                "message": "AI 모델이 로드되지 않아 가짜 데이터로 응답합니다",
                "total_crops": 5
            }
        
        # 1. YOLO로 사람 탐지
        detected_persons = detect_persons_in_video(video_path, max_frames=20)
        
        if not detected_persons:
            return {
                "status": "success",
                "message": "영상 분석 완료, 하지만 탐지된 사람이 없습니다",
                "total_crops": 0
            }
        
        # 2. CLIP 임베딩 생성
        embedding_data = generate_clip_embeddings(detected_persons)
        
        # 3. 글로벌 데이터에 저장
        global analyzed_data
        analyzed_data = embedding_data
        
        return {
            "status": "success", 
            "message": f"'{file.filename}' 분석 완료!",
            "total_crops": len(detected_persons),
            "분석결과": f"{len(detected_persons)}명의 실제 인물이 AI로 탐지되었습니다"
        }
        
    except Exception as e:
        print(f"❌ 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def 인물검색(request: SearchRequest):
    """실제 CLIP 기반 자연어 검색"""
    
    print(f"🔍 실제 AI 검색: '{request.query}'")
    
    if not analyzed_data["embeddings"] is not None:
        raise HTTPException(status_code=404, detail="먼저 영상을 업로드하고 분석해주세요")
    
    if not clip_model:
        raise HTTPException(status_code=500, detail="CLIP 모델이 로드되지 않았습니다")
    
    try:
        # 검색어를 CLIP으로 인코딩
        text_input = clip.tokenize([request.query]).to(device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(text_input)
            text_features = text_features.cpu().numpy()[0]
        
        # 코사인 유사도 계산
        image_embeddings = analyzed_data["embeddings"]
        similarities = np.dot(image_embeddings, text_features) / (
            np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(text_features)
        )
        
        # 상위 k개 결과 선택
        top_indices = np.argsort(-similarities)[:request.k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # 최소 유사도 임계값
                results.append(SearchResult(
                    image_path=analyzed_data["image_paths"][idx],
                    caption=analyzed_data["captions"][idx],
                    score=float(similarities[idx]),
                    image_base64=analyzed_data["images"][idx]
                ))
        
        print(f"✅ 실제 AI 검색 완료: {len(results)}개 결과")
        return results
        
    except Exception as e:
        print(f"❌ 검색 오류: {e}")
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@app.get("/image/{filename}")
async def get_image(filename: str):
    """crop 이미지 파일 반환"""
    image_path = CROP_DIR / filename
    if image_path.exists():
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

@app.get("/stats")
async def 통계():
    """서버 통계"""
    return {
        "AI_모델_상태": {
            "YOLO": "로딩됨" if yolo_model else "로딩실패",
            "CLIP": "로딩됨" if clip_model else "로딩실패"
        },
        "분석_통계": {
            "탐지된_인물": len(analyzed_data["image_paths"]),
            "임베딩_생성됨": analyzed_data["embeddings"] is not None
        },
        "시스템_정보": {
            "디바이스": device,
            "GPU_사용가능": torch.cuda.is_available()
        }
    }

# 서버 실행
if __name__ == "__main__":
    print("🚀 실제 AI 분석 서버 시작!")
    print("🤖 YOLO + CLIP 기반 인물 탐지 및 검색")
    print("📍 서버: http://localhost:8001")
    print("📚 API 문서: http://localhost:8001/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)