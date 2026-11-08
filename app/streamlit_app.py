from datetime import datetime, timezone

import requests
import streamlit as st
from pymongo import MongoClient

from src.config import API_URL, MONGO_URI

db = MongoClient(MONGO_URI).idverify

st.title("ID document verification")
tab1, tab2 = st.tabs(["Verify", "Review queue"])

with tab1:
    f = st.file_uploader("Document photo", type=["jpg", "jpeg", "png"])
    if f:
        st.image(f, width=400)
        if st.session_state.get("file") != (f.name, f.size):    # call the API once per file
            files = {"file": (f.name, f.getvalue(), f.type)}
            st.session_state.result = requests.post(f"{API_URL}/verify", files=files).json()
            st.session_state.file = (f.name, f.size)
        r = st.session_state.result
        st.metric("Decision", r["decision"])
        st.write(f"Spoof score: {r['spoof_score']:.2f}")
        st.json(r["quality"])

with tab2:
    done = {d["pid"] for d in db.review_queue.find()}
    todo = [p for p in db.predictions.find({"decision": "review"}) if p["id"] not in done]
    st.write(f"{len(todo)} pending")
    if todo:
        p = todo[0]
        st.image(p["image"], width=400)
        st.write(f"Model score: {p['spoof_score']:.2f}")
        for col, name, label in zip(st.columns(2), ["Genuine", "Fake"], [0, 1]):
            if col.button(name):
                db.review_queue.insert_one({"pid": p["id"], "label": label,
                                            "image": p["image"], "ts": datetime.now(timezone.utc)})
                st.rerun()