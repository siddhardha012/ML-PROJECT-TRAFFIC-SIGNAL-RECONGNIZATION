"""
Sample Traffic Signal Dataset Generator
Generates synthetic traffic signs with varied backgrounds, scales, rotations,
and lighting to enable instant end-to-end testing and model training.
"""

import os
import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import config

def get_font(size: int):
    """Attempt to load a crisp TrueType font, falling back to default."""
    try:
        # Standard Windows fonts
        for font_name in ["arialbd.ttf", "arial.ttf", "segoeuib.ttf", "calibrib.ttf"]:
            font_path = Path("C:/Windows/Fonts") / font_name
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size)
    except Exception:
        pass
    return ImageFont.load_default()

def draw_stop_sign(size: int = 128) -> Image.Image:
    """Draw an octagonal red Stop sign with white border and STOP text."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = size * 0.08
    r = (size / 2) - margin
    cx, cy = size / 2, size / 2

    # Octagon vertices
    points = []
    for i in range(8):
        angle = math.radians(22.5 + i * 45)
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append((x, y))

    # Outer white border
    draw.polygon(points, fill=(220, 38, 38, 255), outline=(255, 255, 255, 255), width=int(size * 0.04))

    # Inner red octagon
    inner_r = r * 0.92
    inner_points = [
        (cx + inner_r * math.cos(math.radians(22.5 + i * 45)),
         cy + inner_r * math.sin(math.radians(22.5 + i * 45)))
        for i in range(8)
    ]
    draw.polygon(inner_points, outline=(255, 255, 255, 200), width=int(size * 0.02))

    # Text "STOP"
    font = get_font(int(size * 0.28))
    text = "STOP"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw / 2, cy - th / 2 - size * 0.02), text, font=font, fill=(255, 255, 255, 255))
    return img

def draw_no_entry_sign(size: int = 128) -> Image.Image:
    """Draw a circular red No Entry sign with a white horizontal bar."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = size * 0.08
    cx, cy = size / 2, size / 2
    r = (size / 2) - margin

    # Red circle with thin white rim
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 38, 38, 255), outline=(255, 255, 255, 255), width=int(size * 0.03))

    # White horizontal bar in the middle
    bar_w = r * 1.5
    bar_h = r * 0.38
    draw.rectangle([cx - bar_w / 2, cy - bar_h / 2, cx + bar_w / 2, cy + bar_h / 2], fill=(255, 255, 255, 255))
    return img

def draw_traffic_light_sign(size: int = 128) -> Image.Image:
    """Draw a vertical traffic light housing with illuminated red, yellow, green signals."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    box_w = size * 0.42
    box_h = size * 0.86

    # Housing box (dark rounded rectangle)
    draw.rounded_rectangle(
        [cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2],
        radius=int(size * 0.08),
        fill=(31, 41, 55, 255),
        outline=(107, 114, 128, 255),
        width=int(size * 0.02)
    )

    # 3 circular lenses
    lamp_r = size * 0.10
    colors = [
        (239, 68, 68, 255),   # Red
        (245, 158, 11, 255),  # Yellow
        (34, 197, 94, 255)    # Green
    ]
    offsets = [-box_h * 0.28, 0, box_h * 0.28]

    for offset, color in zip(offsets, colors):
        ly = cy + offset
        draw.ellipse([cx - lamp_r, ly - lamp_r, cx + lamp_r, ly + lamp_r], fill=color, outline=(0, 0, 0, 200), width=1)
    return img

def draw_speed_limit_sign(speed: int = 30, size: int = 128) -> Image.Image:
    """Draw a circular speed limit sign with red border and black speed number."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    r = (size / 2) - size * 0.08

    # White inner circle with red thick border
    border_w = int(size * 0.12)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 250, 250, 255), outline=(220, 38, 38, 255), width=border_w)

    # Number
    font = get_font(int(size * 0.38))
    text = str(speed)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw / 2, cy - th / 2 - size * 0.02), text, font=font, fill=(17, 24, 39, 255))
    return img

def draw_turn_sign(direction: str = "Left", size: int = 128) -> Image.Image:
    """Draw a blue circular mandatory turn sign with white arrow."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    r = (size / 2) - size * 0.08

    # Blue circle with white border
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(37, 99, 235, 255), outline=(255, 255, 255, 255), width=int(size * 0.03))

    # Arrow shaft and arrowhead pointing left or right
    shaft_w = size * 0.12
    if direction == "Left":
        # Shaft coming up and curving left
        draw.line([(cx + size * 0.12, cy + size * 0.28), (cx + size * 0.12, cy - size * 0.05)], fill=(255, 255, 255, 255), width=int(shaft_w))
        draw.line([(cx + size * 0.12, cy - size * 0.05), (cx - size * 0.15, cy - size * 0.05)], fill=(255, 255, 255, 255), width=int(shaft_w))
        # Arrowhead
        ax, ay = cx - size * 0.15, cy - size * 0.05
        draw.polygon([(ax - size * 0.16, ay), (ax, ay - size * 0.15), (ax, ay + size * 0.15)], fill=(255, 255, 255, 255))
    else:
        # Turn Right
        draw.line([(cx - size * 0.12, cy + size * 0.28), (cx - size * 0.12, cy - size * 0.05)], fill=(255, 255, 255, 255), width=int(shaft_w))
        draw.line([(cx - size * 0.12, cy - size * 0.05), (cx + size * 0.15, cy - size * 0.05)], fill=(255, 255, 255, 255), width=int(shaft_w))
        # Arrowhead
        ax, ay = cx + size * 0.15, cy - size * 0.05
        draw.polygon([(ax + size * 0.16, ay), (ax, ay - size * 0.15), (ax, ay + size * 0.15)], fill=(255, 255, 255, 255))

    return img

def render_sample_image(class_name: str, target_size: int = 64) -> Image.Image:
    """
    Renders an augmented realistic sample image:
    1. Generates base sign asset at high resolution (128x128)
    2. Applies realistic environmental background (asphalt, sky, daylight, building wall)
    3. Adds slight scale, translation, rotation, and lighting variations
    4. Resizes to 64x64
    """
    canvas_size = 128
    if class_name == "Stop":
        sign = draw_stop_sign(canvas_size)
    elif class_name == "No_Entry":
        sign = draw_no_entry_sign(canvas_size)
    elif class_name == "Traffic_Light":
        sign = draw_traffic_light_sign(canvas_size)
    elif class_name == "Speed_Limit_30":
        sign = draw_speed_limit_sign(30, canvas_size)
    elif class_name == "Speed_Limit_50":
        sign = draw_speed_limit_sign(50, canvas_size)
    elif class_name == "Turn_Left":
        sign = draw_turn_sign("Left", canvas_size)
    elif class_name == "Turn_Right":
        sign = draw_turn_sign("Right", canvas_size)
    else:
        sign = draw_stop_sign(canvas_size)

    # Random background color simulating outdoor environments
    bg_types = [
        (130, 140, 150),  # Overcast sky
        (90, 100, 110),   # Concrete / Asphalt
        (70, 80, 70),     # Foliage
        (160, 175, 190),  # Bright sky
        (200, 190, 180),  # Building wall
    ]
    base_bg = list(random.choice(bg_types))
    # Add random tint
    bg_color = (
        max(0, min(255, base_bg[0] + random.randint(-15, 15))),
        max(0, min(255, base_bg[1] + random.randint(-15, 15))),
        max(0, min(255, base_bg[2] + random.randint(-15, 15)))
    )

    bg_img = Image.new("RGBA", (canvas_size, canvas_size), bg_color + (255,))

    # Random scale factor (0.80 to 0.98)
    scale = random.uniform(0.80, 0.98)
    new_w = int(canvas_size * scale)
    new_h = int(canvas_size * scale)
    scaled_sign = sign.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Random subtle rotation (-6 to +6 degrees)
    rot_angle = random.uniform(-6, 6)
    rotated_sign = scaled_sign.rotate(rot_angle, resample=Image.Resampling.BICUBIC, expand=False)

    # Position centered with slight jitter
    offset_x = (canvas_size - new_w) // 2 + random.randint(-3, 3)
    offset_y = (canvas_size - new_h) // 2 + random.randint(-3, 3)

    bg_img.paste(rotated_sign, (offset_x, offset_y), rotated_sign)

    # Convert to RGB
    final_img = bg_img.convert("RGB")

    # Downsample to target 64x64
    final_img = final_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return final_img

def generate_dataset(train_per_class: int = 50, test_per_class: int = 15):
    """
    Populates dataset/train/ and dataset/test/ folders with synthetic samples.
    """
    print("=" * 60)
    print("GENERATING SAMPLE TRAFFIC SIGN DATASET")
    print(f"Classes: {config.DEFAULT_CLASSES}")
    print(f"Samples per class: Train={train_per_class}, Test={test_per_class}")
    print("=" * 60)

    for class_name in config.DEFAULT_CLASSES:
        train_folder = config.TRAIN_DIR / class_name
        test_folder = config.TEST_DIR / class_name
        train_folder.mkdir(parents=True, exist_ok=True)
        test_folder.mkdir(parents=True, exist_ok=True)

        # Generate training images
        for i in range(train_per_class):
            img = render_sample_image(class_name, config.IMG_HEIGHT)
            img.save(train_folder / f"{class_name.lower()}_train_{i+1:03d}.jpg", quality=95)

        # Generate test images
        for i in range(test_per_class):
            img = render_sample_image(class_name, config.IMG_HEIGHT)
            img.save(test_folder / f"{class_name.lower()}_test_{i+1:03d}.jpg", quality=95)

        print(f"  [OK] Generated {train_per_class} train & {test_per_class} test samples for '{class_name}'")

    print("\nDataset generation completed successfully!")
    print(f"Train directory: {config.TRAIN_DIR}")
    print(f"Test directory:  {config.TEST_DIR}\n")

if __name__ == "__main__":
    generate_dataset()
