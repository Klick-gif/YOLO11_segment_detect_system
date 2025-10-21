import warnings
import multiprocessing
warnings.filterwarnings('ignore')
from ultralytics import YOLO

def evaluate_segmentation(model_path, data_yaml, split_set='val'):
    """
    评价YOLO分割模型
    Args:
        model_path: 模型权重路径
        data_yaml: 数据集配置文件
        split_set: 数据集类型 'train', 'val', 'test'
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 执行分割验证
    metrics = model.val(
        data=data_yaml,
        split=split_set,      # 指定数据集类型
        imgsz=640,
        batch=16,
        iou=0.60,
        conf=0.001,
        workers=0,
        task='segment'        # 重要：指定为分割任务
    )
    print('mAP50  :', metrics.box.map50)
    print('mAP50-95:', metrics.box.map)
    print('Precision:', metrics.box.mp)
    print('Recall   :', metrics.box.mr)
    p, r = metrics.box.mp, metrics.box.mr
    f1 = 2 * p * r / (p + r + 1e-16)
    print(f'F1        : {f1:.3f}')
    
    return metrics


def main():
    # 模型路径 - 替换为你的分割模型路径
    model_path = 'runs/segment/train/weights/best.pt'  # 分割模型通常保存在segment目录
    
    # 数据集配置 - 替换为你的数据配置文件
    data_yaml = 'yolo11n_seg_data.yaml'
    
    # 分别评价三个数据集
    for split_set in ['test', 'train', 'val']:
        print(f"--- 正在评估 {split_set} 数据集 ---")
        
        try:
            evaluate_segmentation(model_path, data_yaml, split_set)
            
        except Exception as e:
            print(f"评估 {split_set} 数据集时出错: {e}\n")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()