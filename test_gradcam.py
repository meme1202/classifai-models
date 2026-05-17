import tensorflow as tf
from app import get_model, compute_gradcam, MODEL_CONFIGS
import numpy as np

img = np.random.rand(1, 224, 224, 3)

for model_name in MODEL_CONFIGS.keys():
    try:
        model = get_model(model_name)
        hm = compute_gradcam(model, img, 0)
        if hm is None:
            print(f'-> {model_name} returned None!')
        else:
            print(f'-> {model_name} SUCCESS!')
    except Exception as e:
        import traceback
        print(f'-> {model_name} FAILED: {e}\n{traceback.format_exc()}')
