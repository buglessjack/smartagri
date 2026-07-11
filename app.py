import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# Render ပေါ်တွင် ဖြစ်တတ်သော TensorFlow Warning စာတန်းများကို ပိတ်ထားခြင်း (Optimization)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K # 💡 FIXED: RAM စနစ်ကို ထိန်းချုပ်ရန် ပေါင်းထည့်ခြင်း

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# AI MODEL & CLASSES CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = 'my_custom_plant_model.h5'

# မော်ဒယ်ဖိုင် တကယ်ရှိမရှိ ကြိုတင်စစ်ဆေးခြင်း
if os.path.exists(MODEL_PATH):
    print(f"Loading AI Model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("AI Model Loaded Successfully!")
else:
    raise FileNotFoundError(f"Error: {MODEL_PATH} ဖိုင်အား ရှာမတွေ့ပါ။ Folder ထဲတွင် သေချာထည့်ပေးပါ။")

# 💡 FIXED: Colab Cell 2 က ထွက်လာတဲ့ စာလုံးအကြီးအသေး ၃၆ ခုစာရင်းအတိုင်း ကွက်တိပြင်ဆင်ထားခြင်း ဖြစ်ပါသည်
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

# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINT FOR IMAGE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    # 1. ဖိုင်ပါမပါ စစ်ဆေးခြင်း
    if 'file' not in request.files:
        return jsonify({
            'success': False, 
            'disease_name': '', 
            'error': 'ဓာတ်ပုံဖိုင် ပေးပို့မှု မရှိပါ (No file uploaded).'
        }), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False, 
            'disease_name': '', 
            'error': 'ဖိုင်အမည် ဗလာဖြစ်နေပါသည် (Empty filename).'
        }), 400

    try:
        # 2. ဓာတ်ပုံအား ဖတ်ရှုပြီး RGB သို့ ပြောင်းလဲခြင်း
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # 3. MobileNetV2 standard size (224x224) ဖြစ်အောင် အရွယ်အစားပြန်ညှိခြင်း
        img = img.resize((224, 224))
        
        # 4. Image Preprocessing (Normalize to 0-1)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Rescale values
        
        # 5. AI မော်ဒယ်အား ခန့်မှန်းခိုင်းခြင်း (Inference)
        predictions = model.predict(x)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_name = class_names[predicted_class_idx]

        # 💡 💡 FIXED: အကယ်၍ အပင်ပုံမဟုတ်ဘဲ လူမျက်နှာ/တခြားအရာများ တင်ပါက တိုက်ရိုက်သိရှိစေမည့် Filter Logic
        if predicted_name == 'non_plant':
            return jsonify({
                'success': False,
                'disease_name': 'Unknown',
                'error': 'စစ်ဆေးမှု မအောင်မြင်ပါ။ ဓာတ်ပုံသည် စိုက်ပျိုးရေးသီးနှံ သို့မဟုတ် အပင်ပုံ မဟုတ်ပါဗျာ။'
            })

        # 💡 Threshold Option: စိတ်ချရမှု 55% ထက်နည်းရင် ရောဂါရှာမတွေ့ဟု သတ်မှတ်မည်
        if confidence < 0.55:
            return jsonify({
                'success': False, 
                'disease_name': '', 
                'error': 'AI မှ သေချာစွာ ခွဲခြားမရပါ။ ပုံကို ပိုမိုနီးကပ်ပြတ်သားစွာ ထပ်မံရိုက်ကူးပေးပါ။'
            })

        # 6. အောင်မြင်သော ရလဒ်အား PHP ဝဘ်ဆိုက်ဆီ JSON ဖြင့် ပြန်လည်ပေးပို့ခြင်း
        return jsonify({
            'success': True,
            'disease_name': predicted_name, 
            'confidence': round(confidence * 100, 2),
            'error': ''
        })

    except Exception as e:
        return jsonify({
            'success': False, 
            'disease_name': '', 
            'error': f'ဆာဗာအတွင်းပိုင်း Error ဖြစ်ပွားပါသည်: {str(e)}'
        }), 500
        
    finally:
        # 💡 FIXED FOR RENDER: ဓာတ်ပုံခန့်မှန်းပြီးတိုင်း ခေါင်းထဲက ယာယီ Cache RAM များကို 
        # ချက်ချင်းပြန်ရှင်းပစ်ခြင်းဖြင့် 512MB RAM Limit ကြောင့် Server Crash ဖြစ်ခြင်းကို ကာကွယ်ပေးပါသည်
        K.clear_session()

# ─────────────────────────────────────────────────────────────────────────────
# SERVER ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
