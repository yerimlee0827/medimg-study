"""뇌종양 MRI 4클래스 분류 데모 (교육·연구용, 진단용 아님)
모델: ResNet50 (ImageNet 사전학습 → 뇌 영역 자르기 + 증강으로 fine-tuning)
"""
import os
import numpy as np
import torch
import timm
import gradio as gr
from PIL import Image
from scipy import ndimage
from torchvision.transforms import v2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "local-dir:D:/yerimlee/medimg-study/outputs/p1/hf_model_E2")   # "hf-hub:아이디/저장소" 또는 "local-dir:폴더경로"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
KOR = {"glioma": "신경교종 (glioma)", "meningioma": "수막종 (meningioma)",
       "notumor": "종양 없음 (notumor)", "pituitary": "뇌하수체 종양 (pituitary)"}

model = timm.create_model(MODEL_SOURCE, pretrained=True).eval()
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
resize224 = v2.Resize((224, 224), antialias=True)


def head_mask(a, thr=20):
    m = ndimage.binary_opening(a > thr, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.ones_like(a, bool)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return ndimage.binary_fill_holes(lab == (np.argmax(sizes) + 1))


def brain_crop(a, thr=20, margin=0.04):
    """학습 때와 똑같은 전처리: 머리 영역만 정사각형으로 자르기"""
    ys, xs = np.where(head_mask(a, thr))
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = max(y1 - y0, x1 - x0) * (1 + 2 * margin)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    y0, y1 = int(max(cy - side / 2, 0)), int(min(cy + side / 2, a.shape[0]))
    x0, x1 = int(max(cx - side / 2, 0)), int(min(cx + side / 2, a.shape[1]))
    return a[y0:y1, x0:x1]


def preprocess(img):
    a = np.asarray(Image.fromarray(img).convert("L"))
    a = np.array(Image.fromarray(brain_crop(a)).resize((256, 256), Image.BILINEAR))   # 복사본(쓰기 가능)
    t = resize224(torch.from_numpy(a).unsqueeze(0)).float().div(255)      # 1×224×224, 0~1
    x = (t.repeat(3, 1, 1) - MEAN) / STD
    return x.unsqueeze(0), t[0].numpy()


def predict(img):
    if img is None:
        return None, None
    x, gray = preprocess(img)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].numpy()
    heat = cam(input_tensor=x, targets=None)[0]
    overlay = show_cam_on_image(np.stack([gray] * 3, -1), heat, use_rgb=True)
    return {KOR[c]: float(p[i]) for i, c in enumerate(CLASSES)}, overlay


DESC = """**교육·연구용 데모입니다. 의료 진단에 사용하지 마세요.**

뇌 MRI 사진 한 장을 올리면 4가지(신경교종 / 수막종 / 뇌하수체 종양 / 종양 없음) 중 무엇인지 예측하고,
**Grad-CAM**으로 모델이 판단할 때 중요하게 본 영역(빨간색)을 보여줍니다.

⚠️ **알려진 한계**: 학습 데이터(공개 Kaggle 파생 데이터)에는 클래스별 촬영 방식·단면 방향 차이가 있어,
병변이 거의 안 보이는 16×16 해상도의 단순 모델로도 80%를 맞힐 수 있습니다.
따라서 높은 정확도가 "종양을 잘 본다"는 뜻은 아닙니다. 외부 병원 데이터 검증, 환자 단위 분할은 하지 않았습니다.
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="뇌 MRI 사진 (단면 1장)"),
    outputs=[gr.Label(num_top_classes=4, label="예측 확률"),
             gr.Image(label="Grad-CAM (전처리로 머리 영역만 잘라낸 이미지 위에 표시)")],
    title="뇌종양 MRI 분류 + Grad-CAM (학습용 데모)",
    description=DESC,
    examples=[[os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", f)]
              for f in sorted(os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")))]
             if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")) else None,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
