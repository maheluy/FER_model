"""
face_expression_webcam.py
=========================
Detecção de faces em tempo real via webcam usando YOLOv8-face,
com classificação de expressões faciais pelo modelo AnyNetFER
treinado no notebook notebooks/anynet_fer_optuna.ipynb.

Teclas durante execução:
    Q  — encerra a aplicação
    S  — salva o frame atual em disco (captura_HHMMSS.jpg)
    P  — pausa / retoma o vídeo
"""

# =============================================================
# CONFIGURAÇÃO — ajuste apenas esta seção conforme necessário
# =============================================================

import os

# Pasta onde estão os pesos dos modelos (relativa a este arquivo)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Caminho para os pesos do AnyNetFER treinado.
# Exemplo: "path/anynet_fer_best.pth"
# Se None, o modelo é iniciado com pesos aleatórios (modo teste de pipeline).
ANYNET_WEIGHTS_PATH = os.path.join(MODELS_DIR, "modelo_visao.pth")  # <- altere aqui

# Caminho para os pesos do detector facial YOLOv8n-face
YOLO_FACE_WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov8n-face.pt")

# Índice ou caminho da webcam (0 = câmera padrão do sistema)
WEBCAM_SOURCE = 0

# Limiar mínimo de confiança para aceitar uma detecção facial do YOLO
YOLO_CONF_THRESHOLD = 0.45

# Limiar mínimo de confiança da expressão para exibir o rótulo
EXPR_CONF_THRESHOLD = 0.20

# Hiperparâmetros do AnyNetFER (devem coincidir com o modelo treinado).
# Se ANYNET_WEIGHTS_PATH for None, estes valores são usados como padrão.
ANYNET_HPARAMS = {
    "stem_width"       : 32,
    "depths"           : [1, 2, 4, 2],
    "widths"           : [32, 64, 128, 256],
    "group_width"      : 32,
    "bottleneck_ratio" : 0.85,
    "dropout_rate"     : 0.4,
}

# Resolução de exibição da janela (largura, altura). None = resolução nativa.
DISPLAY_SIZE = None  # ex.: (1280, 720)

# =============================================================
# IMPORTS
# =============================================================

import sys
import time
import os
from datetime import datetime

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERRO] Pacote 'ultralytics' não encontrado.")
    print("       Execute: pip install ultralytics")
    sys.exit(1)

# =============================================================
# MODELO AnyNetFER  (extraído do notebook trabalho2-visao.ipynb)
# =============================================================

NUM_CLASSES = 8
CLASSES = ["Raiva", "Nojo", "Medo", "Felicidade",
           "Tristeza", "Surpresa", "Neutro", "Desprezo"]

# Cores BGR para cada classe (usadas no bounding box)
CLASS_COLORS = {
    "Raiva"      : (0,   0,   220),   # vermelho
    "Nojo"       : (0,   140, 0  ),   # verde escuro
    "Medo"       : (130, 0,   130),   # roxo
    "Felicidade" : (0,   200, 200),   # amarelo esverdeado
    "Tristeza"   : (200, 100, 0  ),   # azul acinzentado
    "Surpresa"   : (0,   165, 255),   # laranja
    "Neutro"     : (180, 180, 180),   # cinza
    "Desprezo"   : (60,  20,  220),   # vermelho escuro
}


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=1, group_width=1, bottleneck_ratio=1):
        super().__init__()
        mid_channels_raw = max(group_width, int(out_channels * bottleneck_ratio))
        num_groups   = max(1, round(mid_channels_raw / group_width))
        mid_channels = num_groups * group_width

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                      padding=1, groups=num_groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.shortcut(x))


class AnyNetStage(nn.Module):
    def __init__(self, in_channels, out_channels,
                 depth, stride=2, group_width=1, bottleneck_ratio=1):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(ResBottleneckBlock(
                in_channels      = in_channels if i == 0 else out_channels,
                out_channels     = out_channels,
                stride           = stride if i == 0 else 1,
                group_width      = group_width,
                bottleneck_ratio = bottleneck_ratio,
            ))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class AnyNetFER(nn.Module):
    """
    AnyNet adaptada para Reconhecimento de Expressões Faciais.
    Entrada : imagem 1×48×48 (escala de cinza)
    Saída   : logits para NUM_CLASSES classes
    """
    def __init__(self, num_classes=NUM_CLASSES, stem_width=32,
                 depths=(1, 2, 4, 2), widths=(32, 64, 128, 256),
                 group_width=16, bottleneck_ratio=0.5, dropout_rate=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_width, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        in_ch = stem_width
        for d, w, s in zip(depths, widths, [1, 2, 2, 2]):
            self.stages.append(AnyNetStage(
                in_channels=in_ch, out_channels=w, depth=d, stride=s,
                group_width=group_width, bottleneck_ratio=bottleneck_ratio,
            ))
            in_ch = w
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(widths[-1], num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


# =============================================================
# PRÉ-PROCESSAMENTO DA FACE PARA O AnyNetFER
# =============================================================

_face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def preprocess_face(face_bgr: np.ndarray) -> torch.Tensor:
    """Converte um crop BGR (OpenCV) para tensor 1×1×48×48 normalizado."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor   = _face_transform(face_rgb)          # [1, 48, 48]
    return tensor.unsqueeze(0)                    # [1, 1, 48, 48]


# =============================================================
# CARREGAMENTO DOS MODELOS
# =============================================================

def load_anynet(weights_path, device, hparams):
    model = AnyNetFER(
        num_classes      = NUM_CLASSES,
        stem_width       = hparams["stem_width"],
        depths           = hparams["depths"],
        widths           = hparams["widths"],
        group_width      = hparams["group_width"],
        bottleneck_ratio = hparams["bottleneck_ratio"],
        dropout_rate     = hparams["dropout_rate"],
    ).to(device)

    if weights_path and os.path.isfile(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        print(f"[INFO] Pesos AnyNetFER carregados de: {weights_path}")
    else:
        if weights_path:
            print(f"[AVISO] Arquivo de pesos não encontrado: {weights_path}")
        print("[AVISO] AnyNetFER iniciado com pesos ALEATÓRIOS — "
              "resultados de expressão não têm significado real.")

    model.eval()
    return model


def load_yolo_face(device):
    """
    Carrega o detector facial YOLOv8n-face a partir de YOLO_FACE_WEIGHTS_PATH.
    Os pesos ficam na pasta models/ do repositório.
    """
    if not os.path.isfile(YOLO_FACE_WEIGHTS_PATH):
        print(f"[ERRO] Pesos do YOLOv8n-face não encontrados: {YOLO_FACE_WEIGHTS_PATH}")
        sys.exit(1)
    print("[INFO] Carregando YOLOv8n-face...")
    model = YOLO(YOLO_FACE_WEIGHTS_PATH)
    print("[INFO] YOLOv8n-face pronto.")
    return model


# =============================================================
# UTILITÁRIOS DE DESENHO
# =============================================================

def draw_face_box(frame, x1, y1, x2, y2, label, conf_expr, color):
    """
    Desenha o bounding box e o rótulo de expressão sobre o frame.
    """
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Texto principal: expressão + confiança
    text      = f"{label}  {conf_expr:.0%}"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    text_thick = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thick)
    pad = 4

    # Fundo semitransparente para legibilidade
    bg_x1 = x1
    bg_y1 = max(0, y1 - th - baseline - pad * 2)
    bg_x2 = x1 + tw + pad * 2
    bg_y2 = y1

    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, text,
                (x1 + pad, y1 - baseline - pad),
                font, font_scale, (255, 255, 255), text_thick,
                cv2.LINE_AA)


def draw_hud(frame, fps, face_count, paused):
    """Informações gerais no canto superior esquerdo."""
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:5.1f}",
        f"Faces: {face_count}",
        "PAUSADO" if paused else "",
    ]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    y     = 24
    for line in lines:
        if not line:
            continue
        color = (0, 60, 255) if line == "PAUSADO" else (200, 200, 200)
        cv2.putText(frame, line, (10, y), font, scale, (0, 0, 0), thick + 2,
                    cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), font, scale, color, thick,
                    cv2.LINE_AA)
        y += 22

    # Legenda de teclas no canto inferior esquerdo
    hint = "[Q] Sair  [S] Salvar  [P] Pausar"
    cv2.putText(frame, hint, (10, h - 10), font, 0.45,
                (0, 0, 0),       2, cv2.LINE_AA)
    cv2.putText(frame, hint, (10, h - 10), font, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)


# =============================================================
# LOOP PRINCIPAL
# =============================================================

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo de inferência: {device}")

    # --- carrega modelos ---
    anynet   = load_anynet(ANYNET_WEIGHTS_PATH, device, ANYNET_HPARAMS)
    yolo_det = load_yolo_face(device)

    # --- abre webcam ---
    cap = cv2.VideoCapture(WEBCAM_SOURCE)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera: {WEBCAM_SOURCE}")
        sys.exit(1)

    print("[INFO] Webcam aberta. Pressione Q para sair, "
          "S para salvar frame, P para pausar.")

    window_name = "Reconhecimento de Expressões Faciais — AnyNetFER + YOLOv8"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if DISPLAY_SIZE:
        cv2.resizeWindow(window_name, *DISPLAY_SIZE)

    fps        = 0.0
    paused     = False
    last_frame = None
    t_prev     = time.perf_counter()

    while True:
        # --- leitura de tecla ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        if key == ord("p") or key == ord("P"):
            paused = not paused
        if key == ord("s") or key == ord("S"):
            if last_frame is not None:
                ts   = datetime.now().strftime("%H%M%S")
                name = f"captura_{ts}.jpg"
                cv2.imwrite(name, last_frame)
                print(f"[INFO] Frame salvo: {name}")

        if paused:
            if last_frame is not None:
                draw_hud(last_frame, fps, 0, paused=True)
                cv2.imshow(window_name, last_frame)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[AVISO] Não foi possível capturar frame. Encerrando.")
            break

        # --- detecção de faces com YOLOv8 ---
        results    = yolo_det.predict(frame, conf=YOLO_CONF_THRESHOLD,
                                      verbose=False, device=device)
        face_count = 0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # coordenadas do bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face_count += 1

                # --- crop e inferência de expressão ---
                face_crop = frame[y1:y2, x1:x2]
                tensor    = preprocess_face(face_crop).to(device)

                with torch.no_grad():
                    logits = anynet(tensor)
                    probs  = torch.softmax(logits, dim=1).squeeze()

                pred_idx  = probs.argmax().item()
                pred_conf = probs[pred_idx].item()
                pred_name = CLASSES[pred_idx]

                # Exibe apenas se a confiança superar o limiar
                if pred_conf >= EXPR_CONF_THRESHOLD:
                    color = CLASS_COLORS.get(pred_name, (200, 200, 200))
                    draw_face_box(frame, x1, y1, x2, y2,
                                  pred_name, pred_conf, color)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (200, 200, 200), 2)

        # --- FPS ---
        t_now = time.perf_counter()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        draw_hud(frame, fps, face_count, paused=False)
        last_frame = frame.copy()
        cv2.imshow(window_name, frame)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Aplicação encerrada.")


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    run()
