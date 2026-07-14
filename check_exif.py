import sys

from PIL import Image
from PIL.ExifTags import TAGS

paths = sys.argv[1:]
for p in paths:
    img = Image.open(p)
    exif = img.getexif()
    print(f"--- {p} ---")
    if not exif:
        print("  SIN EXIF")
        continue
    found = False
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)
        if "Focal" in str(tag) or "Model" in str(tag):
            print(f"  {tag}: {value}")
            found = True
    if not found:
        print("  EXIF presente pero sin campos de Focal/Model")
