# SMART RETAIL SIMPLE DEMO PROJECT

import cv2
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

print("SMART RETAIL MONITORING SYSTEM DEMO")

# -------------------------------
# MODULE 1: SHELF IMAGE ANALYZER
# -------------------------------

print("\nRunning Shelf Image Analyzer...")

try:
    img = cv2.imread("shelf.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    cv2.imshow("Shelf Image", img)
    cv2.imshow("Shelf Edges Detection", edges)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("Shelf Image Processed Successfully")
except:
    print("Shelf Image Not Found (Skip if no image)")

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

model = LinearRegression()
model.fit(days, sales)

future_day = int(input("Enter Future Day Number: "))
prediction = model.predict([[future_day]])

print("Predicted Sales:", int(prediction[0]))

# -------------------------------
# GRAPH VISUALIZATION
# -------------------------------

plt.plot(days, sales)
plt.title("Sales Data")
plt.xlabel("Days")
plt.ylabel("Sales")
plt.show()

print("\nDemo Completed Successfully")
