import warnings, multiprocessing
warnings.filterwarnings('ignore')
from ultralytics import YOLO

def main():
    model = YOLO(r'runs\\detect\\train4\\weights\\best.pt')
    metrics = model.val(
        data='yolo11n_data.yaml',
        split='test',
        imgsz=640,
        batch=16,
        iou=0.60,
        conf=0.001,
        workers=0          # 先关多进程，调试完再开
    )
    print('mAP50  :', metrics.box.map50)
    print('mAP50-95:', metrics.box.map)
    print('Precision:', metrics.box.mp)
    print('Recall   :', metrics.box.mr)
    p, r = metrics.box.mp, metrics.box.mr
    f1 = 2 * p * r / (p + r + 1e-16)
    print(f'F1        : {f1:.3f}')

if __name__ == '__main__':
    multiprocessing.freeze_support()   # Windows 必加
    main()