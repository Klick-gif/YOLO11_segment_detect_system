from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import os
import cv2
import numpy as np
from PIL import Image
import io
import json
from ultralytics import YOLO
import uuid
from typing import List, Dict, Any
import shutil

app = FastAPI(title="积水识别和车辆淹没部位判别系统")

# 创建必要的目录
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# 挂载静态文件
app.mount("/results", StaticFiles(directory="results"), name="results")

# 模板配置
templates = Jinja2Templates(directory="templates")

# 模型配置
MODELS = {
    "detect": {
        "name": "目标检测模型",
        "path": "detect/yolo11_best.pt",
        "type": "detect"
    },
    "segment": {
        "name": "图像分割模型", 
        "path": "segment/yolo11_best.pt",
        "type": "segment"
    }
}

# 车辆淹没部位类别映射
VEHICLE_PARTS = {
    0: "车窗",
    1: "车门",
    2: "车轮",
}

# 请求模型
class PredictRequest(BaseModel):
    file_id: str
    model_type: str = "detect"
    confidence: float = 0.25

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """主页面"""
    return templates.TemplateResponse("index.html", {"request": request, "models": MODELS})

@app.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request):
    """调试页面"""
    return templates.TemplateResponse("debug.html", {"request": request})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """上传图片"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")
    
    # 生成唯一文件名
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{file_id}{file_extension}"
    file_path = os.path.join("uploads", filename)
    
    # 保存文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"file_id": file_id, "filename": filename, "message": "图片上传成功"}

@app.post("/predict")
async def predict_image(request: PredictRequest):
    """图像预测"""
    try:
        # 从请求中获取参数
        file_id = request.file_id
        model_type = request.model_type
        confidence = request.confidence
        
        print(f"收到预测请求: file_id={file_id}, model_type={model_type}, confidence={confidence}")
        
        # 查找上传的文件
        upload_dir = "uploads"
        files = os.listdir(upload_dir)
        input_file = None
        for f in files:
            if f.startswith(file_id):
                input_file = os.path.join(upload_dir, f)
                break
        
        if not input_file or not os.path.exists(input_file):
            raise HTTPException(status_code=404, detail="找不到上传的文件")
        
        # 加载模型
        if model_type not in MODELS:
            raise HTTPException(status_code=400, detail="不支持的模型类型")
        
        model_path = MODELS[model_type]["path"]
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="模型文件不存在")
        
        model = YOLO(model_path)
        
        # 进行预测
        results = model.predict(
            source=input_file,
            conf=confidence,
            save=True,
            project="results",
            name=f"{file_id}_{model_type}",
            exist_ok=True
        )
        
        # 处理结果
        result_data = []
        vehicle_stats = {}
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    result_data.append({
                        "class_id": cls,
                        "class_name": VEHICLE_PARTS.get(cls, f"类别{cls}") if model_type == 'detect' else '水面',
                        "confidence": conf,
                        "bbox": xyxy.tolist()
                    })
                    
                    # 统计车辆部位
                    part_name = VEHICLE_PARTS.get(cls, f"类别{cls}") if model_type == 'detect' else '水面'
                    if part_name not in vehicle_stats:
                        vehicle_stats[part_name] = 0
                    vehicle_stats[part_name] += 1
        
        # 保存结果图片路径
        result_dir = f"results/{file_id}_{model_type}"
        result_image = None
        if os.path.exists(result_dir):
            for file in os.listdir(result_dir):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    result_image = f"/results/{file_id}_{model_type}/{file}"
                    print(result_image)
                    break
        
        return {
            "success": True,
            "file_id": file_id,
            "model_type": model_type,
            "detections": result_data,
            "vehicle_stats": vehicle_stats,
            "result_image": result_image,
            "total_detections": len(result_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/models")
async def get_models():
    """获取可用模型列表"""
    return {"models": MODELS}



@app.get("/check-files")
async def check_files():
    """检查文件状态"""
    try:
        upload_dir = "uploads"
        results_dir = "results"
        
        upload_files = []
        if os.path.exists(upload_dir):
            upload_files = os.listdir(upload_dir)
        
        result_files = []
        if os.path.exists(results_dir):
            result_files = os.listdir(results_dir)
        
        model_status = {}
        for model_name, model_info in MODELS.items():
            model_status[model_name] = {
                "path": model_info["path"],
                "exists": os.path.exists(model_info["path"])
            }
        
        return {
            "upload_dir": upload_dir,
            "upload_files": upload_files,
            "results_dir": results_dir,
            "result_files": result_files,
            "model_status": model_status
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/download/{file_id}/{model_type}")
async def download_result(file_id: str, model_type: str):
    """下载结果图片"""
    result_dir = f"results/{file_id}_{model_type}"
    if not os.path.exists(result_dir):
        raise HTTPException(status_code=404, detail="结果文件不存在")
    
    # 查找结果图片
    result_files = []
    for file in os.listdir(result_dir):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            result_files.append(file)
    
    if not result_files:
        raise HTTPException(status_code=404, detail="没有找到结果图片")
    
    # 返回第一个结果图片
    result_file = os.path.join(result_dir, result_files[0])
    return FileResponse(
        result_file,
        media_type='application/octet-stream',
        filename=f"result_{file_id}_{model_type}.jpg"
    )

@app.get("/download_original/{file_id}")
async def download_original(file_id: str):
    """下载原始图片"""
    upload_dir = "uploads"
    files = os.listdir(upload_dir)
    original_file = None
    
    for f in files:
        if f.startswith(file_id):
            original_file = os.path.join(upload_dir, f)
            break
    
    if not original_file or not os.path.exists(original_file):
        raise HTTPException(status_code=404, detail="原始文件不存在")
    
    return FileResponse(
        original_file,
        media_type='application/octet-stream',
        filename=f"original_{file_id}.jpg"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
