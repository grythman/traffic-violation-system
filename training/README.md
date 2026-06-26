# Монгол дугаар таних YOLOv8 модель сургах

Энэ хавтас нь Roboflow-оос Монгол дугаарын датасет татаж, YOLOv8 детектор сургаж,
гарсан жинг (`best.pt`) системд шууд залгахад зориулсан скриптүүдийг агуулна.

## Файлууд

| Файл | Үүрэг |
|---|---|
| `train.py` | (Stage 2) Дугаарын **байрлал** олох детектор сургана → `models/mn_plate_yolov8.pt`. Анхдагч датасет: `computer-vision-m1xzb/mongolian-plate` (984). |
| `train_ocr.py` | (Stage 3) Дугаарын **текст унших** тэмдэгт-таних модель сургана → `models/mn_plate_ocr_yolo.pt`. Анхдагч датасет: `ed-bw6q4/mongolia-plates` (9.3k). |
| `predict.py` | Stage 2 детекторыг зураг дээр шалгана. |
| `ocr_postprocess.py` | Stage 3 тэмдэгт-илрүүлэлтийг дугаарын **текст мөр** болгон уншина (+CLI). |
| `requirements-train.txt` | Сургалтад шаардлагатай Python сангууд. |

## Урьдчилсан нөхцөл

Сургалт нь **GPU дээр** хийхэд хамаагүй хурдан. CPU дээр ч ажиллана (удаан).
GPU ашиглах бол өөрийн CUDA хувилбарт тохирсон PyTorch суулгана уу
(https://pytorch.org/get-started/locally/).

```bash
pip install -r training/requirements-train.txt
```

## 1-р алхам: Roboflow API key авах

1. https://app.roboflow.com руу нэвтэрнэ (үнэгүй бүртгэл).
2. Settings → **Roboflow API** хэсгээс **Private API Key**-ээ хуулна.
3. Орчны хувьсагч болгон тохируулна:

```bash
export ROBOFLOW_API_KEY="N"
```

## 2-р алхам: Датасет татаж сургах

Анхдагч тохиргоо нь Монгол дугаарын датасет
(`computer-vision-m1xzb/mongolian-plate`, 984 зураг, CC BY 4.0)-руу заасан байгаа:

```bash
python training/train.py --epochs 100 --imgsz 640 --model yolov8n.pt
```

Скрипт автоматаар:
1. Roboflow-оос датасетийг YOLOv8 форматаар татна.
2. YOLOv8-ийг сургана (early-stopping-тэй).
3. Хамгийн сайн checkpoint-ийг validate хийж mAP хэвлэнэ.
4. `best.pt`-г **`models/mn_plate_yolov8.pt`** болгон хуулна.

Хэрэв та аль хэдийн `data.yaml`-тай (офлайн) бол:

```bash
python training/train.py --skip-download --data /зам/data.yaml --epochs 100
```

### Хэрэгтэй сонголтууд

| Флаг | Тайлбар | Анхдагч |
|---|---|---|
| `--model` | Суурь жин (yolov8`n`/`s`/`m`/`l`/`x`.pt) | `yolov8n.pt` |
| `--epochs` | Сургалтын epoch | `100` |
| `--imgsz` | Зургийн хэмжээ | `640` |
| `--batch` | Багц хэмжээ (`-1` = AutoBatch) | `16` |
| `--device` | `0` (GPU) эсвэл `cpu` | авто |
| `--version` | Roboflow датасетийн хувилбар | `2` |

## 3-р алхам: Сургасан жинг шалгах

```bash
python training/predict.py --weights models/mn_plate_yolov8.pt --source зураг.jpg
```

Энэ нь илрүүлсэн дугаарын тоо, итгэлийн оноо, bbox-ийг хэвлэж, тэмдэглэсэн зургийг
`runs/detect/predict/` дотор хадгална.

## 4-р алхам: Системд залгах

`best.pt` нь аль хэдийн `models/mn_plate_yolov8.pt` болж хуулагдсан тул зөвхөн
`.env`-д заана:

```env
PLATE_MODEL_PATH=/app/models/mn_plate_yolov8.pt
OCR_LANGUAGES=mn,en
```

Дараа нь:

```bash
docker-compose up --build
```

Систем одоо: машин илрүүлэх (анхдагч YOLOv8) → **таны сургасан детектороор дугаарын
яг муж олох** → тэр мужид EasyOCR-ийг кирилл дээр ажиллуулна.

## Нарийвчлал сайжруулах зөвлөмж

Датасет харьцангуй жижиг (984 зураг) тул эхний үр дүн дунд зэрэг гарч магадгүй.
Сайжруулах арга: илүү олон Монгол машины зураг нэмж label хийх (Roboflow дээр),
`yolov8s.pt`/`yolov8m.pt` зэрэг том суурь модель ашиглах, epoch нэмэх, augmentation
тохируулах. Текст уншилт (OCR) хангалтгүй бол ирээдүйд тусгай тэмдэгт таних модель
сургах шаардлагатай — энэ нь тэмдэгт бүрийн annotation бүхий нэмэлт өгөгдөл шаардана.


---

## Тэмдэгт-таних (OCR) модель сургах — `train_ocr.py`

`ed-bw6q4/mongolia-plates` датасет нь **тэмдэгт бүрийг** (10 цифр + кирилл үсэг)
тусад нь хайрцаглаж тэмдэглэсэн **9,304 зурагтай** бөгөөд аль хэдийн сургасан
YOLOv11 модельтэй (mAP@50 ~99.5%). Энэ дээр сурсан YOLO модель нь **OCR engine**
шиг ажиллана: дугаар дээрх тэмдэгт бүрийг илрүүлж, зүүнээс баруун тийш эрэмбэлэн
дугаарын мөрийг сэргээдэг. Энэ нь Монгол кирилл дээр сул байдаг EasyOCR-ийг орлох
зорилготой.

### Сургах

```bash
pip install -r training/requirements-train.txt
export ROBOFLOW_API_KEY="таны_түлхүүр"

# Анхдагчаар ed-bw6q4/mongolia-plates татаж YOLOv8-аар сургана
python training/train_ocr.py --epochs 100 --imgsz 640 --model yolov8n.pt

# Эх моделийн адил YOLOv11 суурь ашиглах бол:
python training/train_ocr.py --model yolo11n.pt --epochs 120

# Офлайн (data.yaml бэлэн бол):
python training/train_ocr.py --skip-download --data /зам/data.yaml
```

Скрипт дараах ажлыг автоматаар гүйцэтгэнэ: датасет татах → класс жагсаалт хэвлэх →
сургах → validate (mAP) → `best.pt`-г **`models/mn_plate_ocr_yolo.pt`** болгон хуулах.

### Сургасан моделиор дугаар унших

```bash
python training/ocr_postprocess.py \
    --weights models/mn_plate_ocr_yolo.pt --source plate_crop.jpg
# -> Detected plate string: '1234УБА'
```

`ocr_postprocess.py` дотор байгаа `read_plate()` функцийг ирээдүйд API-ийн OCR
үе шат болгон шууд импортлон ашиглаж болно (EasyOCR-ийн оронд эсвэл fallback-аар).

### Хоёр шатны бүрэн дамжлага (зөвлөмж)

| Шат | Скрипт | Гаралт | Үүрэг |
|---|---|---|---|
| Машин илрүүлэх | (анхдагч YOLOv8) | bbox | COCO, өөрчлөх шаардлагагүй |
| Дугаар байрлал | `train.py` | `mn_plate_yolov8.pt` | дугаарын муж олох |
| Текст унших | `train_ocr.py` | `mn_plate_ocr_yolo.pt` | тэмдэгт→мөр (OCR) |

Хамгийн өндөр нарийвчлалд хоёуланг нь хослуулна: эхлээд дугаарын мужийг олж,
тэр мужид тэмдэгт-таних моделийг ажиллуулна.
