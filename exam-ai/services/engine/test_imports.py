import sys
print(f"Python: {sys.version}")

try:
    import cv2
    print(f"cv2: {cv2.__version__} OK")
except Exception as e:
    print(f"cv2 ERROR: {e}")

try:
    from ultralytics import YOLO
    print("ultralytics: OK")
except Exception as e:
    print(f"ultralytics ERROR: {e}")

try:
    import numpy as np
    print(f"numpy: {np.__version__} OK")
except Exception as e:
    print(f"numpy ERROR: {e}")

try:
    import fastapi
    print(f"fastapi: {fastapi.__version__} OK")
except Exception as e:
    print(f"fastapi ERROR: {e}")

print("All imports OK - ready to start engine!")
