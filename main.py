# SMART RETAIL SIMPLE DEMO PROJECT (YOLO DL VERSION)

import cv2
from ultralytics import YOLO
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

print("SMART RETAIL MONITORING SYSTEM DEMO")

# -------------------------------
# MODULE 1: SHELF IMAGE ANALYZER (YOLO - DEEP LEARNING)
# -------------------------------

print("\nRunning Shelf Image Analyzer...")

try:
    # Load YOLOv8 pretrained model
    model = YOLO("yolov8n.pt")
    model.to("cpu")   # Force CPU mode (IMPORTANT)

    img = cv2.imread("shelf.jpg")

    if img is None:
        raise Exception("shelf.jpg not found")

    # Run YOLO detection
    results = model.predict(source=img, device="cpu")

    # Get annotated image with bounding boxes
    annotated_frame = results[0].plot()

    cv2.imshow("Shelf YOLO Detection", annotated_frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    print("Shelf Image Processed Successfully Using YOLO")

except Exception as e:
    print("Shelf Image or YOLO Model Error:", e)


# -------------------------------
# MODULE 2: NLP REVIEW ANALYZER
# -------------------------------

print("\nRunning Review Sentiment Analyzer...")

review = input("Enter Customer Review: ")
analysis = TextBlob(review)

print("Sentiment Polarity:", analysis.sentiment.polarity)

if analysis.sentiment.polarity > 0:
    print("Positive Review")
elif analysis.sentiment.polarity < 0:
    print("Negative Review")
else:
    print("Neutral Review")


# -------------------------------
# MODULE 3: DEMAND PREDICTION
# -------------------------------

print("\nRunning Demand Prediction...")

days = np.array([1,2,3,4,5,6,7]).reshape(-1,1)
sales = np.array([100,120,130,150,170,200,220])

model_lr = LinearRegression()
model_lr.fit(days, sales)

future_day = int(input("Enter Future Day Number: "))
prediction = model_lr.predict([[future_day]])

print("Predicted Sales:", int(prediction[0]))


# -------------------------------
# GRAPH VISUALIZATION
# -------------------------------

plt.plot(days, sales)
plt.title("Sales Data")
plt.xlabel("Days")
plt.ylabel("Sales")
plt.show()