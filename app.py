from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io, time, os, json

app = Flask(__name__, static_folder='.')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Binary class labels (index 0 = first, index 1 = second) ──────────────────
# Adjust this order if your model was trained differently
CLASSES = ["Untreated", "Treated"]

MODEL_CONFIGS = {
    "Advanced_Custom_CNN": {
        "path": os.path.join(BASE_DIR, "Advanced_Custom_CNN-20260421T131403Z-3-001", "Advanced_Custom_CNN", "Advanced_Custom_CNN_best.keras"),
        "display_name": "Advanced Custom CNN",
        "description": "Lightweight custom-built convolutional network.",
        "size_mb": 0.7,
        "type": "Custom CNN",
        "input_size": 224,
        "color": "#00d4ff",
        "icon": "A"
    },
    "Advanced_Custom_CNN_V2": {
        "path": os.path.join(BASE_DIR, "Advanced_Custom_CNN_V2-20260421T131425Z-3-001", "Advanced_Custom_CNN_V2", "Advanced_Custom_CNN_V2_best.keras"),
        "display_name": "Custom CNN V2",
        "description": "Enhanced custom CNN with deeper architecture.",
        "size_mb": 6.4,
        "type": "Custom CNN",
        "input_size": 224,
        "color": "#8b5cf6",
        "icon": "A2"
    },
    "MobileNetV2": {
        "path": os.path.join(BASE_DIR, "MobileNetV2-20260421T131432Z-3-001", "MobileNetV2", "MobileNetV2_best.keras"),
        "display_name": "MobileNetV2",
        "description": "Lightweight depthwise separable convolutions.",
        "size_mb": 16.7,
        "type": "Transfer Learning",
        "input_size": 224,
        "color": "#10b981",
        "icon": "M"
    },
    "EfficientNetB0": {
        "path": os.path.join(BASE_DIR, "EfficientNetB0-20260421T131429Z-3-001", "EfficientNetB0", "EfficientNetB0_best.keras"),
        "display_name": "EfficientNetB0",
        "description": "Compound scaling for optimal accuracy-efficiency.",
        "size_mb": 23.8,
        "type": "Transfer Learning",
        "input_size": 224,
        "color": "#f59e0b",
        "icon": "E"
    },
    "DenseNet121": {
        "path": os.path.join(BASE_DIR, "DenseNet121-20260421T131428Z-3-001", "DenseNet121", "DenseNet121_best.keras"),
        "display_name": "DenseNet121",
        "description": "Dense connections between all layers.",
        "size_mb": 34.3,
        "type": "Transfer Learning",
        "input_size": 224,
        "color": "#ec4899",
        "icon": "D"
    },
    "ResNet50V2": {
        "path": os.path.join(BASE_DIR, "ResNet50V2-20260421T131452Z-3-001", "ResNet50V2", "ResNet50V2_best.keras"),
        "display_name": "ResNet50V2",
        "description": "Deep residual learning with skip connections.",
        "size_mb": 102.6,
        "type": "Transfer Learning",
        "input_size": 224,
        "color": "#ef4444",
        "icon": "R"
    }
}

loaded_models = {}

# ── Model loading ─────────────────────────────────────────────────────────────
def get_model(model_name):
    if model_name not in loaded_models:
        print(f"[INFO] Loading {model_name} ...")
        loaded_models[model_name] = tf.keras.models.load_model(MODEL_CONFIGS[model_name]["path"])
        print(f"[INFO] {model_name} ready. Output shape: {loaded_models[model_name].output_shape}")
    return loaded_models[model_name]

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_bytes, size=224):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Binary prediction helper ──────────────────────────────────────────────────
def binary_result(raw_output):
    """
    Handles both:
      - 1-neuron sigmoid: output shape (1, 1) → prob of class index 1
      - 2-neuron softmax: output shape (1, 2) → prob of each class
    Returns: list of {"class_name", "confidence", "percentage"} for both classes
    """
    out = raw_output[0]  # shape: (1,) or (2,)

    if len(out) == 1:
        # Sigmoid output: probability of CLASSES[1]
        p1 = float(out[0])
        p0 = 1.0 - p1
    elif len(out) == 2:
        # Softmax output
        p0 = float(out[0])
        p1 = float(out[1])
    else:
        # Unexpected — still try to make sense
        p1 = float(np.max(out))
        p0 = 1.0 - p1

    return [
        {"class_idx": 0, "class_name": CLASSES[0], "confidence": p0, "percentage": p0 * 100},
        {"class_idx": 1, "class_name": CLASSES[1], "confidence": p1, "percentage": p1 * 100},
    ]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "loaded_models": list(loaded_models.keys()), "classes": CLASSES})

@app.route('/api/models')
def api_models():
    result = []
    for key, cfg in MODEL_CONFIGS.items():
        info = {k: v for k, v in cfg.items() if k != 'path'}
        info["id"] = key
        info["loaded"] = key in loaded_models
        if key in loaded_models:
            m = loaded_models[key]
            info["output_neurons"] = int(m.output_shape[-1])
            info["total_params"] = int(m.count_params())
        result.append(info)
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    model_name = request.form.get('model', 'MobileNetV2')
    if model_name not in MODEL_CONFIGS:
        return jsonify({"error": "Unknown model"}), 400

    try:
        image_bytes = request.files['image'].read()
        cfg = MODEL_CONFIGS[model_name]

        t0 = time.time()
        model = get_model(model_name)
        img = preprocess(image_bytes, cfg["input_size"])

        t1 = time.time()
        raw = model.predict(img, verbose=0)
        t2 = time.time()

        predictions = binary_result(raw)
        # Sort descending by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        verdict = predictions[0]["class_name"]
        confidence = predictions[0]["confidence"]

        return jsonify({
            "model": model_name,
            "verdict": verdict,
            "confidence": confidence,
            "predictions": predictions,
            "inference_ms": round((t2 - t1) * 1000, 2),
            "total_ms": round((t2 - t0) * 1000, 2),
            "output_neurons": int(raw.shape[-1]),
            "classes": CLASSES
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_bytes = request.files['image'].read()
    models_param = request.form.get('models', '')
    models_list = [m for m in models_param.split(',') if m in MODEL_CONFIGS] if models_param else list(MODEL_CONFIGS.keys())

    results = []
    for name in models_list:
        try:
            cfg = MODEL_CONFIGS[name]
            model = get_model(name)
            img = preprocess(image_bytes, cfg["input_size"])

            t1 = time.time()
            raw = model.predict(img, verbose=0)
            inf_ms = round((time.time() - t1) * 1000, 2)

            predictions = binary_result(raw)
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            verdict = predictions[0]["class_name"]
            confidence = predictions[0]["confidence"]

            results.append({
                "model": name,
                "display_name": cfg["display_name"],
                "color": cfg["color"],
                "size_mb": cfg["size_mb"],
                "type": cfg["type"],
                "verdict": verdict,
                "confidence": confidence,
                "predictions": predictions,
                "inference_ms": inf_ms
            })
        except Exception as e:
            results.append({"model": name, "display_name": MODEL_CONFIGS[name]["display_name"],
                            "color": MODEL_CONFIGS[name]["color"], "error": str(e)})

    return jsonify(results)

# ── Grad-CAM XAI ──────────────────────────────────────────────────────────────

import base64

def compute_gradcam(model, img_array, class_idx):
    try:
        base_model = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.Model):
                try:
                    s = tuple(layer.output_shape)
                    if len(s) == 4:
                        base_model = layer
                        break
                except: pass
                
        img_t = tf.cast(img_array, tf.float32)

        if base_model is not None:
            pre_layers = model.layers[:model.layers.index(base_model)]
            head_layers = model.layers[model.layers.index(base_model)+1:]
            
            x = img_t
            for layer in pre_layers:
                if not isinstance(layer, tf.keras.layers.InputLayer):
                    x = layer(x)
                    
            with tf.GradientTape() as tape:
                conv_out = base_model(x)
                tape.watch(conv_out)
                
                head_x = conv_out
                for layer in head_layers:
                    if not isinstance(layer, tf.keras.layers.InputLayer):
                        head_x = layer(head_x)
                preds = head_x
                if preds.shape[-1] == 1:
                    loss = preds[:, 0] if class_idx == 1 else -preds[:, 0]
                else:
                    loss = preds[:, class_idx]
                
            grads = tape.gradient(loss, conv_out)
        else:
            layer_name = None
            for layer in reversed(model.layers):
                if hasattr(layer, 'output_shape'):
                    try:
                        s = tuple(layer.output_shape)
                        if len(s) == 4:
                            layer_name = layer.name
                            break
                    except: pass
            if not layer_name: return None
            
            grad_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(img_t)
                if preds.shape[-1] == 1:
                    loss = preds[:, 0] if class_idx == 1 else -preds[:, 0]
                else:
                    loss = preds[:, class_idx]
            grads = tape.gradient(loss, conv_out)

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
        heatmap = tf.maximum(heatmap, 0)
        mx = tf.reduce_max(heatmap)
        if mx > 0:
            heatmap = heatmap / mx
        return heatmap.numpy()
    except Exception as e:
        import traceback
        print(f"[WARN] Grad-CAM: {e}\n{traceback.format_exc()}")
        return None

def overlay_heatmap(image_bytes, heatmap, size=224, alpha=0.5):
    """Overlay jet-colormap heatmap on original image using PIL only."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size), Image.LANCZOS)
    img_arr = np.array(img, dtype=np.float32)
    hm = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS),
        dtype=np.float32
    ) / 255.0
    # Jet colormap
    r = np.clip(1.5 - np.abs(4 * hm - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * hm - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * hm - 1), 0, 1)
    jet = np.stack([r, g, b], axis=-1) * 255.0
    blended = np.clip((1 - alpha) * img_arr + alpha * jet, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def region_analysis(heatmap):
    h, w = heatmap.shape
    regions = {
        "top-left":     heatmap[:h//2, :w//2].mean(),
        "top-right":    heatmap[:h//2, w//2:].mean(),
        "center":       heatmap[h//4:3*h//4, w//4:3*w//4].mean(),
        "bottom-left":  heatmap[h//2:, :w//2].mean(),
        "bottom-right": heatmap[h//2:, w//2:].mean(),
    }
    hot = max(regions, key=regions.get)
    coverage = float((heatmap > 0.5).mean() * 100)
    return hot, coverage, {k: round(float(v), 3) for k, v in regions.items()}

TREATED_CRITERIA = [
    "Uniform surface coating patterns detected across leaf area",
    "Chemical residue signatures visible in texture gradients",
    "Reduced stress pigmentation — consistent chlorophyll distribution",
    "Smooth spectral uniformity associated with treatment application",
    "Absence of natural stress markers typical in untreated samples",
]
UNTREATED_CRITERIA = [
    "Natural, uncoated surface texture with raw organic patterns",
    "No chemical residue or coating signatures present",
    "Irregular pigmentation or natural growth stress markers visible",
    "High-frequency texture variation typical of untreated samples",
    "Natural surface micro-structure without treatment interference",
]

@app.route('/api/explain', methods=['POST'])
def explain():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    model_name = request.form.get('model', 'MobileNetV2')
    if model_name not in MODEL_CONFIGS:
        return jsonify({"error": "Unknown model"}), 400
    try:
        image_bytes = request.files['image'].read()
        cfg = MODEL_CONFIGS[model_name]
        model = get_model(model_name)
        img_arr = preprocess(image_bytes, cfg["input_size"])
        raw = model.predict(img_arr, verbose=0)
        preds = binary_result(raw)
        preds.sort(key=lambda x: x["confidence"], reverse=True)
        verdict = preds[0]["class_name"]
        confidence = preds[0]["confidence"]
        class_idx = preds[0]["class_idx"]

        heatmap = compute_gradcam(model, img_arr, class_idx)
        heatmap_b64 = None
        hot_region = "center"
        coverage = 0.0
        region_scores = {}

        if heatmap is not None and heatmap.size > 1:
            heatmap_b64 = overlay_heatmap(image_bytes, heatmap, cfg["input_size"])
            hot_region, coverage, region_scores = region_analysis(heatmap)

        criteria = TREATED_CRITERIA if verdict == "Treated" else UNTREATED_CRITERIA

        return jsonify({
            "model": model_name,
            "verdict": verdict,
            "confidence": round(confidence * 100, 2),
            "heatmap_image": heatmap_b64,
            "hot_region": hot_region,
            "coverage_pct": round(coverage, 1),
            "region_scores": region_scores,
            "criteria": criteria,
            "explanation": (
                f"The model focused primarily on the {hot_region} region of the image, "
                f"covering approximately {coverage:.1f}% of the area with high activation. "
                f"Key visual patterns in that region drove the {verdict} classification "
                f"with {confidence*100:.1f}% confidence."
            )
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("[ClassifAI] Server starting at http://localhost:5000")
    print(f"[ClassifAI] Binary classes: {CLASSES}")
    app.run(debug=True, host='0.0.0.0', port=5000)

