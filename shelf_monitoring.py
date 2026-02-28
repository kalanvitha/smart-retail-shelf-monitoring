# =============================== SMART RETAIL SYSTEM (PYTHON 3) ===============================
import streamlit as st
import cv2
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from PIL import Image
import sqlite3
from datetime import datetime
import pandas as pd
import os
import uuid

# =============================== PAGE CONFIG ===============================
st.set_page_config(page_title="Smart Retail System", layout="centered")

# =============================== DARK THEME ===============================
st.markdown("""
<style>
.stApp {background-color: #0f172a; color: #e2e8f0;}
div.stButton>button {background-color: #2563eb; color: white; border-radius: 10px; height: 3em;}
div[data-testid="metric-container"] {background-color: #1e293b; padding: 15px; border-radius: 15px; box-shadow: 0px 4px 15px rgba(0,0,0,0.4);}
h1, h2, h3 {color: #f8fafc;}
hr {border: 1px solid #334155;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================== DATABASE ===============================
conn = sqlite3.connect("retail_data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS shelf_data(
id INTEGER PRIMARY KEY AUTOINCREMENT,
timestamp TEXT,
note TEXT,
image_path TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS review_data(
id INTEGER PRIMARY KEY AUTOINCREMENT,
review TEXT,
polarity REAL,
shelf_id INTEGER,
timestamp TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_data(
id INTEGER PRIMARY KEY AUTOINCREMENT,
day INTEGER,
prediction INTEGER,
timestamp TEXT
)
""")
conn.commit()

# =============================== IMAGE FOLDER ===============================
folder_shelves = os.path.join(os.getcwd(), "shelves")
os.makedirs(folder_shelves, exist_ok=True)

# =============================== HELPER FUNCTIONS ===============================
def insert_shelf(note, img_path):
    cursor.execute(
        "INSERT INTO shelf_data (timestamp,note,image_path) VALUES (?,?,?)", 
        (str(datetime.now()), note, img_path)
    )
    conn.commit()
    return cursor.lastrowid

def insert_review(review, polarity, shelf_id):
    cursor.execute(
        "INSERT INTO review_data (review,polarity,shelf_id,timestamp) VALUES (?,?,?,?)",
        (review, polarity, shelf_id, str(datetime.now()))
    )
    conn.commit()

def insert_prediction(day, pred):
    cursor.execute(
        "INSERT INTO prediction_data (day,prediction,timestamp) VALUES (?,?,?)",
        (day, pred, str(datetime.now()))
    )
    conn.commit()

# =============================== ANALYZE SHELF (IMAGE PROCESSING) ===============================
def analyze_shelf(img, expected_slots=5):
    """
    Analyze shelf image using image processing:
    - Detect products based on contours
    - Highlight empty slots
    - Determine shelf status
    """
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    product_count = len(contours)

    annotated = img_np.copy()
    y_positions = [cv2.boundingRect(c)[1] for c in contours]
    alignment_std = np.std(y_positions) if y_positions else 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        color = (0,255,0) if abs(y - np.mean(y_positions)) < 20 else (255,0,0)
        cv2.rectangle(annotated, (x,y), (x+w, y+h), color, 2)

    # Highlight empty slots
    img_width = img_np.shape[1]
    slot_width = img_width // expected_slots
    occupied_slots = [0]*expected_slots
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        slot_idx = min(x // slot_width, expected_slots-1)
        occupied_slots[slot_idx] = 1
    for i, filled in enumerate(occupied_slots):
        if not filled:
            x_start = i * slot_width
            cv2.rectangle(annotated, (x_start,0), (x_start+slot_width,img_np.shape[0]), (0,255,255),2)

    # Shelf status note
    if product_count == 0:
        note = "Empty Shelf"
    elif product_count < expected_slots:
        note = "Partially Filled Shelf"
    elif alignment_std > 20:
        note = "Misarranged Shelf"
    else:
        note = "Well Stocked Shelf"

    annotated_img = Image.fromarray(annotated).convert("RGB")
    return note, annotated_img

def notify_shelf_status(note):
    if note == "Empty Shelf":
        st.warning("⚠️ ALERT: Shelf is EMPTY! Restock immediately.")
    elif note == "Partially Filled Shelf":
        st.info("ℹ️ NOTICE: Shelf is partially filled. Monitor stock levels.")
    elif note == "Misarranged Shelf":
        st.warning("⚠️ ALERT: Shelf items are MISARRANGED! Reorganize stock.")

# =============================== MANAGER ACCOUNTS ===============================
MANAGERS = {
    "kalanvitha_29": "kalanvitha@29",
    "imthiyaz_49": "imthiyaz@49"
}

# =============================== SESSION STATE ===============================
for key in ["login","page","role","reviewed","submitted","current_shelf"]:
    if key not in st.session_state:
        st.session_state[key] = None if key=="role" else False if key in ["login","reviewed","submitted"] else "home"

# =============================== LOGIN PAGE ===============================
if not st.session_state.login:
    st.title("🏪 Smart Retail Monitoring System")
    st.subheader("Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in MANAGERS and MANAGERS[user] == pwd:
            st.session_state.login = True
            st.session_state.role = "manager"
            st.session_state.page = "home"
            st.success("Manager Login Successful ✅")
            st.rerun()
        elif user.strip() != "" and pwd.strip() != "":
            st.session_state.login = True
            st.session_state.role = "customer"
            st.session_state.page = "reviews"
            st.success("Customer Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")

# =============================== MAIN SYSTEM ===============================
if st.session_state.login:
    allowed_pages = ["reviews"] if st.session_state.role=="customer" else ["home","shelf","reviews","prediction","history"]
    if st.session_state.page not in allowed_pages:
        st.session_state.page = allowed_pages[0]

    st.title("🛒 Smart Retail Monitoring Platform")

    # ---------------- HOME (Manager Only) ----------------
    if st.session_state.page=="home":
        st.success("🟢 System Status: All Modules Operational")
        total_shelf = cursor.execute("SELECT COUNT(*) FROM shelf_data").fetchone()[0]
        total_reviews = cursor.execute("SELECT COUNT(*) FROM review_data").fetchone()[0]
        total_predictions = cursor.execute("SELECT COUNT(*) FROM prediction_data").fetchone()[0]
        pos = cursor.execute("SELECT COUNT(*) FROM review_data WHERE polarity>0").fetchone()[0]
        neg = cursor.execute("SELECT COUNT(*) FROM review_data WHERE polarity<0").fetchone()[0]

        col1,col2,col3 = st.columns(3)
        col1.metric("Shelf Records", total_shelf)
        col2.metric("Total Reviews", total_reviews)
        col3.metric("Predictions", total_predictions)
        col1,col2 = st.columns(2)
        col1.metric("Positive Reviews", pos)
        col2.metric("Negative Reviews", neg)

        st.divider()
        if st.button("Start Process ▶"):
            st.session_state.page="shelf"
            st.rerun()
        if st.button("🚪 Logout"):
            st.session_state.login=False
            st.session_state.role=None
            st.rerun()

    # ---------------- SHELF (Manager Only) ----------------
    elif st.session_state.page=="shelf" and st.session_state.role=="manager":
        st.subheader("📷 AI Shelf Classification with Empty Slot Highlight")
        uploaded = st.file_uploader("Upload Shelf Image", type=["jpg","png","jpeg"])
        if uploaded:
            with st.spinner("Analyzing Shelf Image..."):
                img = Image.open(uploaded).convert("RGB")
                col1,col2,col3 = st.columns([1,2,1])
                with col2: st.image(img, width=300)
                note, annotated_img = analyze_shelf(img, expected_slots=5)
                
                img_filename = f"shelf_{uuid.uuid4().hex}.png"
                img_path = os.path.join(folder_shelves, img_filename)
                annotated_img.save(img_path)
                
                shelf_id = insert_shelf(note, img_path)
                notify_shelf_status(note)
                
                st.subheader("Annotated Shelf Image 🔍")
                st.image(Image.open(img_path), width=400)
                st.info(f"Shelf Status: {note}")

        col1,col2 = st.columns(2)
        with col1: 
            if st.button("⬅ Back to Dashboard"):
                st.session_state.page="home"
                st.rerun()
        with col2:
            if st.button("Next ➜ Customer Reviews"):
                st.session_state.page="reviews"
                st.rerun()

    # ---------------- REVIEWS (Customer + Manager) ----------------
    elif st.session_state.page=="reviews":
        st.subheader("💬 Customer Review")

        # ---------- CUSTOMER ----------
        if st.session_state.role == "customer":
            if st.session_state.reviewed:
                st.success("✅ You have already submitted your review.")
            else:
                df_shelves = pd.read_sql(
                    "SELECT * FROM shelf_data WHERE image_path IS NOT NULL AND image_path != ''",
                    conn
                )
                df_shelves = df_shelves[df_shelves['image_path'].apply(os.path.exists)].copy()
                if df_shelves.empty:
                    st.info("No shelf images available for review yet.")
                else:
                    if not isinstance(st.session_state.current_shelf, dict):
                        st.session_state.current_shelf = None
                    if st.session_state.current_shelf is None:
                        row = df_shelves.sample(1).iloc[0]
                        st.session_state.current_shelf = row.to_dict()
                    else:
                        row = st.session_state.current_shelf

                    shelf_id = row['id']
                    note = row['note']
                    img_path = row['image_path']

                    st.markdown(f"**Shelf ID:** {shelf_id}")
                    st.image(Image.open(img_path), width=400)

                    options = ["Empty Shelf", "Partially Filled Shelf", "Well Stocked Shelf", "Misarranged Shelf"]
                    default_index = options.index(note) if note in options else 0

                    review_text = st.selectbox(
                        "Select Shelf Status",
                        options,
                        index=default_index,
                        key=f"review_{shelf_id}"
                    )

                    comment = st.text_area("Optional Comment", key=f"comment_{shelf_id}")

                    if st.button("Submit Review", key=f"submit_{shelf_id}"):
                        full_review = review_text
                        if comment.strip() != "":
                            full_review += " | " + comment
                        polarity = TextBlob(full_review).sentiment.polarity
                        insert_review(full_review, polarity, shelf_id)

                        st.session_state.reviewed = True
                        st.success("✅ Review submitted successfully! Logging out...")
                        st.session_state.login = False
                        st.session_state.role = None
                        st.session_state.page = "home"
                        st.session_state.current_shelf = None
                        st.rerun()

            if st.button("⬅ Logout"):
                st.session_state.login = False
                st.session_state.role = None
                st.session_state.page = "home"
                st.session_state.current_shelf = None
                st.session_state.submitted = False
                st.session_state.reviewed = False
                st.rerun()

        # ---------- MANAGER ----------
        elif st.session_state.role=="manager":
            df_review = pd.read_sql("SELECT * FROM review_data", conn)
            for col in df_review.columns:
                df_review[col] = df_review[col].apply(lambda x: str(x) if x is not None else "")

            if not df_review.empty:
                st.subheader("📊 Sentiment Distribution")
                df_review["polarity"] = pd.to_numeric(df_review["polarity"], errors="coerce").fillna(0)
                pos = len(df_review[df_review["polarity"]>0])
                neg = len(df_review[df_review["polarity"]<0])
                fig_bar, ax_bar = plt.subplots(figsize=(6,2))
                ax_bar.barh(["Positive","Negative"], [pos,neg], color=["#22c55e","#ef4444"])
                for i,v in enumerate([pos,neg]):
                    ax_bar.text(v+0.5,i,str(v),va='center',fontweight='bold')
                st.pyplot(fig_bar)

                st.markdown("**All Reviews:**")
                st.dataframe(df_review)

            col1,col2 = st.columns(2)
            with col1: 
                if st.button("⬅ Back to Shelf"):
                    st.session_state.page="shelf"
                    st.rerun()
            with col2: 
                if st.button("Next ➜ Sales Prediction"):
                    st.session_state.page="prediction"
                    st.rerun()

    # ---------------- PREDICTION (Manager Only) ----------------
    elif st.session_state.page=="prediction" and st.session_state.role=="manager":
        st.subheader("📈 Sales Prediction")
        days=np.array([1,2,3,4,5,6,7]).reshape(-1,1)
        sales=np.array([100,120,130,150,170,200,220])
        model=LinearRegression().fit(days,sales)
        future=st.number_input("Enter Future Day", min_value=1, step=1)
        if st.button("Predict"):
            pred=int(model.predict([[future]])[0])
            insert_prediction(future,pred)
            st.success(f"Predicted Sales: {pred}")
        fig, ax = plt.subplots(figsize=(4,2))
        ax.plot(days,sales,marker='o',color="#f97316")
        ax.set_title("Sales Trend Analysis")
        ax.set_xlabel("Day")
        ax.set_ylabel("Sales")
        ax.grid(True)
        st.pyplot(fig)
        col1,col2 = st.columns(2)
        with col1: 
            if st.button("⬅ Back to Reviews"):
                st.session_state.page="reviews"
                st.rerun()
        with col2: 
            if st.button("Next ➜ Data History"):
                st.session_state.page="history"
                st.rerun()

    # ---------------- HISTORY (Manager Only) ----------------
    elif st.session_state.page=="history" and st.session_state.role=="manager":
        st.subheader("📊 Data History Dashboard")
        df_shelf=pd.read_sql("SELECT * FROM shelf_data",conn)
        df_review=pd.read_sql("SELECT * FROM review_data",conn)
        df_pred=pd.read_sql("SELECT * FROM prediction_data",conn)

        for col in df_review.columns:
            df_review[col] = df_review[col].apply(lambda x: str(x) if x is not None else "")

        col1,col2,col3=st.columns(3)
        col1.metric("Shelf Entries", len(df_shelf))
        col2.metric("Total Reviews", len(df_review))
        col3.metric("Predictions", len(df_pred))
        st.divider()

        if not df_review.empty:
            st.subheader("📊 Sentiment Distribution")
            df_review["polarity"] = pd.to_numeric(df_review["polarity"], errors="coerce").fillna(0)
            pos=len(df_review[df_review["polarity"]>0])
            neg=len(df_review[df_review["polarity"]<0])
            fig_bar, ax_bar = plt.subplots(figsize=(6,2))
            ax_bar.barh(["Positive","Negative"], [pos,neg], color=["#22c55e","#ef4444"])
            for i,v in enumerate([pos,neg]):
                ax_bar.text(v+0.5,i,str(v),va='center',fontweight='bold')
            st.pyplot(fig_bar)

        if not df_shelf.empty:
            st.subheader("📦 Shelf Status Summary")
            shelf_counts = df_shelf["note"].value_counts()
            fig_shelf, ax_shelf = plt.subplots(figsize=(6,2))
            ax_shelf.barh(shelf_counts.index, shelf_counts.values, color="#3b82f6")
            for i,v in enumerate(shelf_counts.values):
                ax_shelf.text(v+0.5,i,str(v),va='center',fontweight='bold')
            st.pyplot(fig_shelf)

        if not df_pred.empty:
            st.subheader("📈 Prediction History Trend")
            fig_pred, ax_pred = plt.subplots(figsize=(4,2))
            ax_pred.plot(df_pred["day"], df_pred["prediction"], marker='o', color="#f97316")
            ax_pred.set_xlabel("Day")
            ax_pred.set_ylabel("Predicted Sales")
            ax_pred.grid(True)
            st.pyplot(fig_pred)

        st.subheader("📋 Full Data Records")
        if not df_shelf.empty:
            st.markdown("**Shelf Records:**")
            st.dataframe(df_shelf)
        if not df_review.empty:
            st.markdown("**Review Records:**")
            st.dataframe(df_review)
        if not df_pred.empty:
            st.markdown("**Prediction Records:**")
            st.dataframe(df_pred)

        if st.button("⬅ Back to Dashboard"):
            st.session_state.page="home"
            st.rerun()

# =============================== FOOTER ===============================
st.markdown("""
<hr style="margin-top:50px">
<center style='color:gray; font-size:13px;'>
Smart Retail Monitoring System © 2026 <br>
Developed by Kalanvitha & Imthiyaz
</center>
""", unsafe_allow_html=True)