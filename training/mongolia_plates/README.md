# `ed-bw6q4/mongolia-plates` локал сургалт

Энэ фолдер нь **зөвхөн** `ed-bw6q4/mongolia-plates` датасетийг локалаар татаж сургах
бие даасан `train.py` скриптийг агуулна.

> Датасет: https://universe.roboflow.com/ed-bw6q4/mongolia-plates
> 9,304 зураг · 36 класс (тэмдэгт бүр) · CC BY 4.0 · бэлэн YOLOv11 модельтэй

Энэ датасет нь **тэмдэгт-түвшний** аннотацитай тул сурсан YOLO модель нь OCR engine
шиг ажиллана: дугаар дээрх тэмдэгт бүрийг илрүүлж, зүүнээс баруун тийш эрэмбэлэн
дугаарын мөрийг сэргээдэг.

## Ашиглах

```bash
# 1. Сангууд
pip install -r requirements.txt

# 2. Roboflow API key (app.roboflow.com -> Settings -> Roboflow API)
export ROBOFLOW_API_KEY="таны_түлхүүр"

# 3. Татаж сургах (анхдагч: 100 epoch, yolov8n)
python train.py

# Сонголтууд:
python train.py --epochs 150 --model yolo11n.pt --device 0   # GPU + YOLOv11 суурь
python train.py --batch -1                                    # AutoBatch
python train.py --skip-download --data ./mongolia-plates-1/data.yaml  # офлайн
```

Дуусахад `best.pt` нь энэ фолдерт болон төслийн `models/mn_plate_ocr_yolo.pt`-д
хуулагдана. Дараа нь `.env`-д:

```env
PLATE_OCR_MODEL_PATH=/app/models/mn_plate_ocr_yolo.pt
```

## Дугаарыг текст болгон унших

```bash
python ../ocr_postprocess.py \
    --weights ../../models/mn_plate_ocr_yolo.pt --source plate_crop.jpg
# -> Detected plate string: '1234УБА'
```

## Гол флагууд

| Флаг | Тайлбар | Анхдагч |
|---|---|---|
| `--model` | Суурь жин (`yolov8n.pt`, `yolo11n.pt` ...) | `yolov8n.pt` |
| `--epochs` | Сургалтын epoch | `100` |
| `--imgsz` | Зургийн хэмжээ | `640` |
| `--batch` | Багц (`-1` = AutoBatch) | `16` |
| `--device` | `0` (GPU) / `cpu` | авто |
| `--export-format` | Roboflow экспорт формат | `yolov8` |
| `--skip-download` | Татахгүй, `--data` ашиглах | — |
| `--no-deploy` | `models/`-д хуулахгүй | — |
