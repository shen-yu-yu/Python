import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from model import DetectionModel

# 和训练保持一致
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectionModel()
model.load_state_dict(torch.load("yolo.pth", map_location=device))
model.to(device)
model.eval()

image_path = r"D:\code\yolo\ultralytics-main\datasets\bvn\images\train\10e951c660f1c699ba19d736ccd9cdb4.png"  # 改成实际路径
image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size

inp = transform(image).unsqueeze(0).to(device)  # [1, 3, 448, 448]

with torch.no_grad():
    out = model(inp)[0].cpu()  # [7]

cx, cy, w, h = out[0:4].tolist()
class_id = int(out[4:7].argmax())  # one-hot/logits 取最大的类

# 归一化坐标 -> 原图像素框
x1 = (cx - w / 2) * orig_w
y1 = (cy - h / 2) * orig_h
x2 = (cx + w / 2) * orig_w
y2 = (cy + h / 2) * orig_h

print(f"class={class_id}, box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

draw = ImageDraw.Draw(image)
draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
draw.text((x1, max(0, y1 - 15)), f"class {class_id}", fill="red")
image.save("result.jpg")
image.show()