import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tflite_runtime.interpreter as tflite # 💡 FIXED: TensorFlow အစား TFLite သုံးခြင်း

app = Flask(__name__)

MODEL_PATH = 'my_custom_plant_model.tflite'

# TFLite Interpreter အား စတင်ပတ်မောင်းခြင်း (RAM 10MB သာ သုံးပါသည်)
if os.path.exists(MODEL_PATH):
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("🚀 TFLite AI Model Loaded Successfully into Minimal RAM!")
else:
    raise FileNotFoundError(f"Error: {MODEL_PATH} ဖိုင်အား ရှာမတွေ့ပါ။")

class_names = [
    'Alternaria_D', 'Anthracnose - Colletotrichum', 'Bacterialblight', 'Blast', 
    'Botrytis Leaf Blight', 'Brownspot', 'Bulb Rot', 'Bulb_blight-D', 'Caterpillar-P', 
    'Downy mildew', 'Fusarium-D', 'Healthy leaves', 'Iris yellow virus_augment', 
    'Purple blotch', 'Rust', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy', 'Tungro', 'Virosis-D', 'Xanthomonas Leaf Blight', 
    'healthy', 'leaf curl', 'leaf spot', 'non_plant', 'stemphylium Leaf Blight', 
    'whitefly', 'yellowish'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'disease_name': '', 'error': 'No file uploaded.'}), 400
    file = request.files['file']
    
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        
        # Preprocessing
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        
        # TFLite Inference တွက်ချက်ခြင်း
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_name = class_names[predicted_class_idx]

        if predicted_name == 'non_plant':
            return jsonify({'success': False, 'disease_name': 'Unknown', 'error': 'တင်သွင်းထားသော ဓာတ်ပုံသည် အပင်ရွက် မဟုတ်ပါ။ ကျေးဇူးပြု၍ သီးနှံအရွက်ကိုသာ ရိုက်ကူးပါ။'})

        if confidence < 0.55:
            return jsonify({'success': False, 'disease_name': '', 'error': 'AI မှ သေချာစွာ ခွဲခြားမရပါ။'})

        return jsonify({
            'success': True,
            'disease_name': predicted_name, 
            'confidence': round(confidence * 100, 2),
            'error': ''
        })
    except Exception as e:
        return jsonify({'success': False, 'disease_name': '', 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
