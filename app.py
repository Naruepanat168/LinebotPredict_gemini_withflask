from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img
from dotenv import load_dotenv
import os
import numpy as np
import google.generativeai as genai # API Gemini

def create_generative_model():
       
    api_key = os.getenv('GOOGLE_API_KE')
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    return genai.GenerativeModel(model_name="gemini-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)

app = Flask(__name__)

load_dotenv()
channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN",None)
channel_secret = os.getenv("LINE_CHANNEL_SECRET",None)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

model_generative = create_generative_model()
model = load_model("model.h5")

response = ""
# โหลด Model Gemini
model_generative = create_generative_model()
convo = model_generative.start_chat(history=[])

# Webhook route for LINE Messaging API

@app.route('/webhook', methods=['GET','POST'])
def webhook():
    # Get X-Line-Signature header and request body
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # Handle webhook events
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)  

    return 'Connection'  

# Event handler for text messages
# Test
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # Extract message text from the event
    message_text = event.message.text
    convo.send_message(message_text)
    response=convo.last.text

    # Reply to the user with the same message
    reply_message = TextSendMessage(text=response)
    line_bot_api.reply_message(event.reply_token, reply_message)

# Event handler for images messages
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):

    # โหลดโมเดล Process Img
    model = load_model("model.h5")

    # รับข้อมูลรูปภาพ
    message_content = line_bot_api.get_message_content(event.message.id)

    # ตั้งชื่อไฟล์
    filename = f"image_{event.message.id}.jpg"

    # บันทึกไฟล์
    with open(filename, "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)

    # อ่านรูปภาพ
    img1 = cv2.imread(filename)

    # แปลงสี
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # ปรับขนาด
    img1 = cv2.resize(img1, (224, 224))

    # แปลงเป็น array
    img1 = np.array(img1) / 255.0

    # เปลี่ยนมิติ
    img1 = np.reshape(img1, (1, 224, 224, 3))

    # ทำนาย
    prediction = model.predict(img1)

    # ดึงคลาสและความน่าจะเป็น
    label = ['Anthracnose', 'Healthy']
    result = label[np.argmax(prediction)] # output ชนิดใบ เช่น ="แอนแทรคโนส"
    percen = prediction[0][np.argmax(prediction)]*100

    # กำหนดข้อความตอบกลับ
    if result == label[1]:  # 'Healthy'
        response = "ใบมะม่วงของคุณดูมีสุขภาพดี "
    else:
        # ประมวลผลข้อความด้วย model_generative
        convo.send_message(f"อาการและวิธีอารักขาของใบ{result}ของมะม่วงทำยังไง")
        response = convo.last.text

    # ส่งผลลัพธ์กลับไปยังผู้ใช้
    line_bot_api.reply_message(
        event.reply_token,
        [
            TextSendMessage(text=f"เป็นใบ {result}"),
            TextSendMessage(text=f"โอกาส {round(percen,2)}%"),
            TextSendMessage(text=response)
        ],
    )
if __name__ == '__main__':
    app.run(port=8000,debug=True)
    
