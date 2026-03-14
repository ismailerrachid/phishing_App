
import sqlite3
from pathlib import Path
import hashlib
import time

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


APP_TITLE = "Phishing Detection • Demo (Phishing / Legit)"
DATA_PATH = Path(__file__).with_name("phishing_dataset.csv")

DB_PATH = Path(__file__).with_name("users.db")

def _hash_pwd(pwd: str, salt: str = "yanecode-salt") -> str:
    return hashlib.sha256((salt + pwd).encode("utf-8")).hexdigest()

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
        """
    )
    con.commit()

    defaults = [
        ("admin", _hash_pwd("admin"), "admin"),
        ("user", _hash_pwd("user"), "user"),
    ]
    for u, p, r in defaults:
        try:
            cur.execute("INSERT INTO users(username, password_hash, role) VALUES(?,?,?)", (u, p, r))
        except sqlite3.IntegrityError:
            pass
    con.commit()
    con.close()

def check_credentials(username: str, password: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    con.close()
    if not row:
        return False, None
    stored_hash, role = row
    return stored_hash == _hash_pwd(password), role

def logout():
    for k in ["auth", "role", "username"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["label"].astype(int)
    return df

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    if not SKLEARN_OK:
        return None, None, None

    X = df["email_text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=35000)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_size": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
    }
    return pipe, metrics, (X_train, X_test, y_train, y_test)

def test_interface(model):
    st.subheader("🧪 Test interface (paste an email text)")

    samples = {
        "Phishing sample": "URGENT: Your Microsoft 365 account will be suspended. Verify now: http://security-verify-login.com/update",
        "Legit sample": "Meeting reminder: cybersecurity at 14:00. Location: Teams. More info: https://support.google.com",
    }
    sample_key = st.selectbox("Load sample", list(samples.keys()))
    email_text = st.text_area("Email text", value=samples[sample_key], height=160)

    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)

    if st.button("Detect", type="primary"):
        if model is None:
            st.error("Model is not available because scikit-learn could not be imported.")
            return

        proba = float(model.predict_proba([email_text])[0][1])
        pred = int(proba >= threshold)

        if pred == 1:
            st.error(f"⚠️ Result: **PHISHING** — probability = **{proba:.2f}** (threshold={threshold:.2f})")
        else:
            st.success(f"✅ Result: **LEGIT** — probability = **{proba:.2f}** (threshold={threshold:.2f})")

        st.caption("Educational demo on a synthetic dataset. For real deployment, train on real data and evaluate carefully.")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_db()

    st.title(APP_TITLE)
    st.caption("Educational demo: login + dashboard + dataset preview + ML model + test interface.")

    if "auth" not in st.session_state:
        st.session_state["auth"] = False

    if not st.session_state["auth"]:
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username", placeholder="admin or user")
            password = st.text_input("Password", type="password", placeholder="admin or user")
            submitted = st.form_submit_button("Sign in")
            if submitted:
                ok, role = check_credentials(username, password)
                if ok:
                    st.session_state["auth"] = True
                    st.session_state["role"] = role
                    st.session_state["username"] = username
                    st.success("Logged in successfully.")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        st.info("Demo accounts: **admin/admin** and **user/user**")
        return

    df = load_data()

    if not SKLEARN_OK:
        st.warning("scikit-learn is not installed. Install it with: `pip install scikit-learn`")
        st.stop()

    model, metrics, _ = train_model(df)

    with st.sidebar:
        st.write(f"👤 User: **{st.session_state.get('username','')}**")
        st.write(f"🔐 Role: **{st.session_state.get('role','')}**")
        st.divider()
        page = st.radio("Navigation", ["Dashboard", "Test", "Dataset"], index=0)
        st.divider()
        if st.button("Logout"):
            logout()

    if page == "Dashboard":
        st.subheader("📊 Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df)}")
        c2.metric("Phishing rate", f"{df['label'].mean():.2%}")
        c3.metric("Test size", f"{metrics['test_size']}")
        c4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

        st.markdown("**Model metrics (on test split):**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{metrics['precision']:.3f}")
        m3.metric("Recall", f"{metrics['recall']:.3f}")
        m4.metric("F1", f"{metrics['f1']:.3f}")

        st.markdown("**Confusion matrix** (rows = true, cols = predicted):")
        st.write(np.array(metrics["confusion_matrix"]))

        st.divider()
        st.subheader("Dataset balance")
        st.bar_chart(df["label"].value_counts().sort_index().rename({0:"Legit", 1:"Phishing"}))

    elif page == "Test":
        test_interface(model)

    else:
        st.subheader("🧾 Dataset")
        st.write("Preview (top 50 rows):")
        st.dataframe(df.head(50), use_container_width=True)

        st.download_button(
            label="Download CSV dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="phishing_dataset.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
