from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import logging

# 使用 mmrotate 的 API
from mmdet.apis import init_detector, inference_detector
import mmrotate

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置模型路径
# ----------------orcnn--------------------
# CONFIG_FILE = '/mnt/bp/backend/configs/orcnn/oriented_rcnn_config.py'  # 替换为你的实际配置文件路径
# CHECKPOINT_FILE = '/mnt/bp/backend/checkpoints/orcnn/oriented_rcnn.pth'  # 替换为你的实际模型文件路径

# ----------------lsknet--------------------
CONFIG_FILE = '/mnt/bp/backend/configs/lsknet/lsk_s_fpn_1x_dota_le90.py'  # 替换为你的实际配置文件路径
CHECKPOINT_FILE = '/mnt/bp/backend/checkpoints/lsknet/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth'  # 替换为你的实际模型文件路径


# ----------------ours--------------------
# CONFIG_FILE = '/mnt/bp/backend/configs/ours/owner-78.19.py'  # 替换为你的实际配置文件路径
# CHECKPOINT_FILE = '/mnt/bp/backend/checkpoints/ours/epoch_13.pth'  # 替换为你的实际模型文件路径

try:
    # 初始化模型
    logger.info(f"正在加载模型... 配置文件: {CONFIG_FILE}, 检查点: {CHECKPOINT_FILE}")
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda')  # 使用 CUDA 设备
    logger.info("模型加载成功！")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    model = None


@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="模型未正确加载")

    logger.info(f"接收到新的检测请求: {file.filename}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            logger.info(f"图片已保存到临时文件: {temp_path}")

        try:
            # 进行推理
            logger.info("开始进行目标检测...")
            result = inference_detector(model, temp_path)
            logger.info("检测完成，开始处理结果...")

            # 可视化结果
            img = model.show_result(
                temp_path,
                result,
                score_thr=0.3,
                show=False,
                wait_time=0
            )

            # 保存结果图片
            output_path = temp_path.replace('.jpg', '_result.jpg')
            cv2.imwrite(output_path, img)
            logger.info(f"结果图片已保存: {output_path}")

            # 处理检测结果为可序列化的格式
            processed_result = []
            if isinstance(result, (list, tuple)):
                for class_results in result:
                    if isinstance(class_results, np.ndarray):
                        # 将每个检测结果转换为列表
                        detections = []
                        for detection in class_results:
                            det_dict = {
                                'bbox': detection[:5].tolist(),  # 前5个值是边界框坐标
                                'score': float(detection[5])     # 第6个值是置信度分数
                            }
                            detections.append(det_dict)
                        processed_result.append(detections)
            else:
                # 如果结果是单个 ndarray
                if isinstance(result, np.ndarray):
                    processed_result = result.tolist()

            return {
                "success": True,
                "image_url": f"/api/results/{Path(output_path).name}",
                "detections": processed_result
            }

        except Exception as e:
            logger.error(f"检测过程发生错误: {str(e)}")
            raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")

        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                logger.info("临时文件已清理")
            except Exception as e:
                logger.error(f"清理临时文件失败: {str(e)}")

    except Exception as e:
        logger.error(f"处理上传文件时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@app.get("/api/results/{image_name}")
async def get_result_image(image_name: str):
    file_path = f"/tmp/{image_name}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="结果图片未找到")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)