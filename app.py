import os
import io
import re
import csv
import json
import random
from datetime import datetime
from functools import wraps

# Prevent background thread locking on Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    Response
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from pypdf import PdfReader


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)
app.secret_key = "jobshield_secure_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "jobshield_v2.db")

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ============================================================
# MODERNBERT MODEL INITIALIZATION
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "jobshield-modernbert-final")

print("=" * 60)
print("JOBSHIELD AI - MODEL INITIALIZATION")
print("=" * 60)
print("Model path:", MODEL_PATH)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA available: True | GPU:", torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")
    print("CUDA available: False | Using CPU")

tokenizer = None
model = None

# Class 0: Legitimate, Class 1: Fake
LEGITIMATE_INDEX = 0
FAKE_INDEX = 1

if os.path.isdir(MODEL_PATH):
    try:
        print("\nLoading JobShield ModernBERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        print("Tokenizer loaded successfully!")

        print("Loading fine-tuned JobShield ModernBERT model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            low_cpu_mem_usage=True,
            dtype=torch.float32
        )
        model.to(DEVICE)
        model.eval()

        print("Model loaded successfully!")
        print("Number of labels:", model.config.num_labels)
        print("Label mapping: {0: 'Legitimate', 1: 'Fake'}")

    except Exception as e:
        print("\nERROR: Could not initialize JobShield AI model.")
        print("Reason:", str(e))
        tokenizer = None
        model = None
else:
    print("\nWARNING: Model directory was not found at:", MODEL_PATH)

print("=" * 60)
print(f"MODEL READY: {model is not None} | Device: {DEVICE}")
print("=" * 60)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    return response


# ============================================================
# DATABASE MODELS
# ============================================================

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    scans = db.relationship(
        "Scan",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )


class Scan(db.Model):
    __tablename__ = "scans"

    id = db.Column(db.Integer, primary_key=True)
    scan_code = db.Column(db.String(20), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    job_title = db.Column(db.String(150), nullable=False)
    company_name = db.Column(db.String(150), nullable=False)
    job_text = db.Column(db.Text, nullable=True)
    prediction = db.Column(db.String(20), nullable=True, default="Legitimate")
    fake_prob = db.Column(db.Float, nullable=True, default=0.0)
    legit_prob = db.Column(db.Float, nullable=True, default=100.0)
    confidence = db.Column(db.Float, nullable=True, default=100.0)
    risk_score = db.Column(db.Integer, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    indicators = db.Column(db.Text, nullable=True)
    scan_category = db.Column(db.String(100), nullable=True, default="General Job Posting")
    date_analyzed = db.Column(db.DateTime, default=datetime.utcnow)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session or not session.get("is_admin", False):
            flash("Access denied. Administrator privileges required.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# COMPREHENSIVE RULE-BASED SCAM INDICATOR ENGINE
# ============================================================

def evaluate_risk_indicators(text):
    if not text:
        return [], 0, "General Job Posting"

    text_lower = text.lower()
    sentences = re.split(r'[\r\n.!?]+', text_lower)
    indicators = []
    rule_risk = 0
    category = "General Job Posting"

    # 1. UPFRONT PAYMENT / REGISTRATION FEE EXTORTION (WITH NEGATION AWARENESS)
    fee_negation_patterns = [
        r'\bno\s+(?:registration|application|security|training|processing|account|activation|hidden)?\s*(?:fees?|charges?|deposit|payment)\b',
        r'\b(?:do|does|did)\s+not\s+(?:charge|ask|require|collect|demand)\b',
        r'\b(?:never|not)\s+(?:ask|charge|require|request|collect|demand)\s+(?:for\s+)?(?:any\s+)?(?:fees?|payment|money|deposit)\b',
        r'\b(?:fee|charge|deposit|payment)\s+(?:is|are)?\s*(?:not|never)\s*(?:required|collected|asked)\b',
        r'\bwithout\s+(?:any\s+)?(?:fee|charge|payment|deposit)\b',
        r'\bfree\s+of\s+(?:cost|charge)\b'
    ]

    fee_extortion_patterns = [
        r'\b(?:registration|application|processing|activation|equipment|training|security|uniform|onboarding)\s+fees?\b',
        r'\b(?:pay|send|transfer|deposit)\s+(?:a\s+)?(?:refundable\s+)?(?:fee|amount|sum|money|deposit|\₹|rs\.?|\$)\b',
        r'\b(?:refundable|non[- ]refundable)\s+(?:registration|application|security|account)?\s*fees?\b',
        r'\baccount\s+activation\s+fee\b',
        r'\bpay\s+to\s+activate\b'
    ]

    fee_detected = False
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if any(re.search(np, s) for np in fee_negation_patterns):
            continue
        if any(re.search(fp, s) for fp in fee_extortion_patterns):
            fee_detected = True
            break

    if fee_detected:
        indicators.append("Upfront payment or recruitment fee requested (Definite scam pattern)")
        rule_risk += 85
        category = "Application / Registration Fee Scam"

    # 2. CRYPTOCURRENCY / USDT / BINANCE / CRYPTO WALLETS
    crypto_patterns = [
        r'\b(?:usdt|crypto|cryptocurrency|bitcoin|binance|coinbase|trust\s*wallet|crypto\s*wallet)\b',
        r'\b(?:payout\s+in\s+usdt|commission\s+settlement|crypto\s+settlement)\b',
        r'\b(?:trial\s+bonus|trial\s+bonus\s+upon\s+registration)\b',
        r'\breferral\s+code\b'
    ]
    if any(re.search(pat, text_lower) for pat in crypto_patterns):
        indicators.append("Request for crypto wallet, USDT daily payout, or registration bonus detected (Task Scam Pattern)")
        rule_risk += 80
        if category == "General Job Posting":
            category = "Cryptocurrency / Task Scam"

    # 3. OFF-PLATFORM MESSAGING APPS (WHATSAPP / TELEGRAM RECRUITMENT)
    chat_patterns = [
        r'\b(?:whatsapp|telegram|signal|viber)\b',
        r'\bcontact\s+(?:us\s+)?on\s+(?:whatsapp|telegram)\b',
        r'\bmessage\s+(?:our\s+)?(?:onboarding\s+mentor|hr|mentor|recruiter|manager)\b',
        r'whatsapp\s*[:\-]?\s*[\+\d\s\-]+',
        r'telegram\s*[:\-]?\s*@[\w\d_]+'
    ]
    if any(re.search(pat, text_lower) for pat in chat_patterns):
        indicators.append("Recruiter requests communication through unofficial messaging app (WhatsApp/Telegram)")
        rule_risk += 75
        if category == "General Job Posting":
            category = "Off-Platform Communication"

    # 4. TASK OPTIMIZATION / APP RATING SCAMS
    task_patterns = [
        r'\b(?:tasks?\s+per\s+set|data\s+submission\s+tasks|optimize\s*/\s*submit|simulate\s+real-user\s+engagement)\b',
        r'\b(?:app\s+reviewer|rating\s+apps|store\s+rankings|cloud\s+workbench)\b',
        r'\b(?:30[–-]60\s+minutes\s+daily|no\s+typing.*required)\b'
    ]
    if any(re.search(pat, text_lower) for pat in task_patterns):
        indicators.append("App rating / Data submission task manipulation pattern detected")
        rule_risk += 75
        if category == "General Job Posting":
            category = "App Optimization / Task Scam"

    # 5. UNREALISTIC DAILY INCOME / INSTANT DAILY PAYOUTS
    income_patterns = [
        r'\b(?:guaranteed\s+(?:income|job|salary|placement))\b',
        r'\b(?:easy\s+money|earn\s+daily|unlimited\s+income|huge\s+salary)\b',
        r'\b\$\s*[\d,]+\s*[–-]\s*\$\s*[\d,]+\s*(?:per\s+day|daily)\b',
        r'\binstant\s+daily\s+payout\b',
        r'\bearn\s+(?:₹|rs\.?|\$)?\s*[\d,]+\s*(?:per\s+day|daily)\b'
    ]
    if any(re.search(pat, text_lower) for pat in income_patterns):
        indicators.append("Unrealistic daily income claim or instant daily payout detected")
        rule_risk += 40
        if category == "General Job Posting":
            category = "Unrealistic Income Claim"

    # 6. SENSITIVE FINANCIAL DATA / OTP / PASSBOOK
    credential_patterns = [
        r'\bbank\s+(?:account|details|passbook)\b',
        r'\botp\b',
        r'\bgift\s*cards?\b',
        r'\bwire\s+transfer\b'
    ]
    if any(re.search(pat, text_lower) for pat in credential_patterns):
        indicators.append("Request for sensitive banking credentials, passbook, or wire transfer detected")
        rule_risk += 50
        if category == "General Job Posting":
            category = "Financial Data Harvesting"

    # 7. GENERIC RECRUITER EMAIL PROVIDER
    email_match = re.search(r'[\w\.-]+@([\w\.-]+\.\w+)', text_lower)
    if email_match:
        domain = email_match.group(1).lower()
        if domain in ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "rediffmail.com"]:
            indicators.append(f"Official corporate email missing; generic provider used (@{domain})")
            rule_risk += 20

    # 8. DIRECT JOINING / PROCESS BYPASS
    bypass_patterns = [
        r'\bno\s+interview\b',
        r'\bno\s+written\s+test\b',
        r'\bno\s+technical\s+test\b',
        r'\bdirect\s+(?:joining|hiring|selection|placement)\b',
        r'\binstant\s+(?:joining|hiring|appointment)\b',
        r'\bno\s+experience\s+required\b'
    ]
    if any(re.search(pat, text_lower) for pat in bypass_patterns):
        indicators.append("Standard interview and qualification verification bypassed")
        rule_risk += 25

    rule_risk = min(100, rule_risk)

    if not indicators:
        indicators = [
            "No common recruitment scam indicators detected",
            "Job posting reflects standard recruitment language"
        ]

    return indicators, rule_risk, category


def extract_job_details(text):
    if not text:
        return {"salary": "Not specified", "location": "Not specified", "email": "Not specified"}

    original_text = text
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', original_text, re.IGNORECASE)
    email = email_match.group(0) if email_match else "Not specified"

    salary_patterns = [
        r'(?:salary|pay|ctc|package|compensation)\s*[:\-]?\s*((?:₹|rs\.?|inr|\$)\s*[\d,.]+(?:\s*(?:lpa|lakhs?|per\s+month|monthly|pa|per\s+day|daily))?(?:\s*[-–—to]+\s*(?:₹|rs\.?|inr|\$)?\s*[\d,.]+(?:\s*(?:lpa|lakhs?|per\s+month|monthly|pa|per\s+day|daily))?)?)',
        r'(\$[\d,.]+\s*[-–—to]+\s*\$[\d,.]+\s*(?:per\s+day|daily|per\s+month|monthly|annually|per\s+year)?)',
        r'((?:₹|rs\.?|inr|\$)\s*[\d,.]+\s*(?:lpa|lakhs?|per\s+month|monthly|pa|per\s+day|daily))'
    ]
    salary = "Not specified"
    for pattern in salary_patterns:
        match = re.search(pattern, original_text, re.IGNORECASE)
        if match:
            salary = match.group(1).strip()
            break

    location_patterns = [
        r'(?:location|work\s+location|work\s+mode)\s*[:\-]\s*([^\n\r]+)',
        r'\b(work\s+from\s+home)\b',
        r'\b(remote\s*\([^\)]+\))\b',
        r'\b(remote)\b',
        r'\b(hybrid)\b',
        r'\b(on[-\s]?site)\b'
    ]
    location = "Not specified"
    for pattern in location_patterns:
        match = re.search(pattern, original_text, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            if len(location) > 80:
                location = location[:80].strip()
            break

    return {"salary": salary, "location": location, "email": email}


# ============================================================
# SYNCHRONIZED DUAL-LAYER PREDICTION & RISK ANALYSIS
# ============================================================

def predict_with_modernbert(text):
    """
    Synchronized Dual-Layer Assessment:
    - Layer 1: ModernBERT Contextual NLP Classifier
    - Layer 2: Rule-Based Security Engine
    - Decision: Combines NLP and heuristic indicators to eliminate false negatives.
    """
    if model is None or tokenizer is None:
        raise RuntimeError("ModernBERT model is not loaded.")

    if not text or not text.strip():
        raise ValueError("Job posting text cannot be empty.")

    # 1. Tokenize (Max sequence length 512)
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 2. ModernBERT Raw Softmax Probabilities
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()

    model_legit_raw = round(probabilities[LEGITIMATE_INDEX] * 100, 2)
    model_fake_raw = round(probabilities[FAKE_INDEX] * 100, 2)

    # 3. Rule-Based Security Assessment
    indicators, rule_risk_score, scam_category = evaluate_risk_indicators(text)

    # 4. Composite Risk Score (Takes highest danger signal from either AI or Rules)
    composite_fake_score = max(model_fake_raw, float(rule_risk_score))
    composite_fake_score = max(0.0, min(100.0, composite_fake_score))
    composite_legit_score = round(100.0 - composite_fake_score, 2)

    # 5. Unified Decision (>= 50% is FAKE)
    if composite_fake_score >= 50.0:
        final_prediction = "Fake"
        risk_level = "High Risk"
    elif composite_fake_score >= 25.0:
        final_prediction = "Suspicious"
        risk_level = "Moderate Risk"
    else:
        final_prediction = "Legitimate"
        risk_level = "Low Risk"

    confidence = max(composite_fake_score, composite_legit_score)

    return {
        "prediction": final_prediction,
        "legitimate_probability": composite_legit_score,
        "fake_probability": round(composite_fake_score, 2),
        "confidence": round(confidence, 2),
        "risk_score": int(round(composite_fake_score)),
        "trust_score": composite_legit_score,
        "risk_level": risk_level,
        "indicators": indicators,
        "category": scam_category,
        "model_raw_fake": model_fake_raw,
        "rule_risk_raw": rule_risk_score
    }


def analyze_job_posting(text):
    result = predict_with_modernbert(text)
    return (
        result["risk_score"],
        result["risk_level"],
        " | ".join(result["indicators"]),
        result
    )


def infer_scan_category(scan_or_text):
    if hasattr(scan_or_text, "scan_category") and scan_or_text.scan_category:
        return scan_or_text.scan_category
    if hasattr(scan_or_text, "job_text") and scan_or_text.job_text:
        _, _, category = evaluate_risk_indicators(scan_or_text.job_text)
        return category
    if isinstance(scan_or_text, str) and scan_or_text.strip():
        _, _, category = evaluate_risk_indicators(scan_or_text)
        return category
    return "General Job Posting"


def build_scam_category_data(scans):
    category_counts = {}

    for scan in scans:
        label = infer_scan_category(scan)
        if not label:
            label = "General Job Posting"
        category_counts[label] = category_counts.get(label, 0) + 1

    ranked = sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"label": label, "count": count}
        for label, count in ranked[:7]
    ]


# ============================================================
# USER & PUBLIC ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/auth", methods=["GET", "POST"])
def auth():
    mode = request.args.get("mode", "login")

    if request.method == "POST":
        action = request.form.get("action")

        if action == "register":
            fullname = request.form.get("fullname", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

            if not fullname:
                flash("Please enter your full name.", "error")
                return redirect(url_for("auth", mode="register"))

            if not is_valid_email(email):
                flash("Please enter a valid email address.", "error")
                return redirect(url_for("auth", mode="register"))

            if password != confirm_password:
                flash("Passwords do not match. Please try again.", "error")
                return redirect(url_for("auth", mode="register"))

            if len(password) < 6:
                flash("Password must contain at least 6 characters.", "error")
                return redirect(url_for("auth", mode="register"))

            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash("Email is already registered. Please sign in.", "error")
                return redirect(url_for("auth", mode="login"))

            hashed_pwd = generate_password_hash(password, method="pbkdf2:sha256")
            new_user = User(
                fullname=fullname,
                email=email,
                password_hash=hashed_pwd,
                is_admin=False,
                is_active=True
            )
            db.session.add(new_user)
            db.session.commit()

            session["user_id"] = new_user.id
            session["user_name"] = new_user.fullname
            session["user_email"] = new_user.email
            session["is_admin"] = False
            session["is_first_visit"] = True

            return redirect(url_for("dashboard"))

        elif action == "login":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            user = User.query.filter_by(email=email).first()

            if user and check_password_hash(user.password_hash, password):
                if not user.is_active:
                    flash("Your account has been suspended by an administrator.", "error")
                    return redirect(url_for("auth", mode="login"))

                session["user_id"] = user.id
                session["user_name"] = user.fullname
                session["user_email"] = user.email
                session["is_admin"] = user.is_admin
                session["is_first_visit"] = False

                if user.is_admin:
                    return redirect(url_for("admin_dashboard"))
                return redirect(url_for("dashboard"))

            flash("Invalid email or password.", "error")
            return redirect(url_for("auth", mode="login"))

    return render_template("auth.html", mode=mode)


# ============================================================
# USER DASHBOARD
# ============================================================

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("auth", mode="login"))

    user_id = session["user_id"]
    user = User.query.get(user_id)

    if not user:
        session.clear()
        return redirect(url_for("auth", mode="login"))

    # POST: NEW JOB ANALYSIS SUBMISSION
    if request.method == "POST":
        job_title = request.form.get("job_title", "Analyzed Job Posting").strip()
        company = request.form.get("company_name", "Unverified Employer").strip()
        job_text = request.form.get("job_text", "").strip()

        # PDF Text Extraction
        if "pdf_file" in request.files:
            file = request.files["pdf_file"]
            if file and file.filename and file.filename.lower().endswith(".pdf"):
                try:
                    pdf_reader = PdfReader(io.BytesIO(file.read()))
                    extracted_text = ""
                    for page in pdf_reader.pages:
                        page_str = page.extract_text()
                        if page_str:
                            extracted_text += page_str + "\n"

                    if extracted_text.strip():
                        job_text = extracted_text.strip()
                    else:
                        flash("Could not extract text from the PDF. It may contain scanned images.", "error")
                        return redirect(url_for("dashboard"))
                except Exception as e:
                    flash("Error parsing PDF file.", "error")
                    return redirect(url_for("dashboard"))

        if not job_text:
            flash("Please enter job details or upload a valid PDF.", "error")
            return redirect(url_for("dashboard"))

        # Dual-Layer Hybrid Inference
        try:
            score, level, flags, prediction_result = analyze_job_posting(job_text)
        except Exception as e:
            print("\nMODEL PREDICTION ERROR:", str(e))
            flash("AI model could not analyze this job posting. Please check model files.", "error")
            return redirect(url_for("dashboard"))

        # Generate unique scan code
        scan_code = f"SCAN-{random.randint(1000, 9999)}"
        while Scan.query.filter_by(scan_code=scan_code).first():
            scan_code = f"SCAN-{random.randint(1000, 9999)}"

        # Save scan to database with exact synchronized results
        category_label = prediction_result.get("category") or infer_scan_category(job_text)
        new_scan = Scan(
            scan_code=scan_code,
            user_id=user_id,
            job_title=job_title,
            company_name=company,
            job_text=job_text,
            prediction=prediction_result["prediction"],
            fake_prob=prediction_result["fake_probability"],
            legit_prob=prediction_result["legitimate_probability"],
            confidence=prediction_result["confidence"],
            risk_score=score,
            risk_level=level,
            indicators=flags,
            scan_category=category_label
        )
        db.session.add(new_scan)
        db.session.commit()

        print("\n" + "=" * 60)
        print(f"ANALYSIS COMPLETE: {scan_code} | Title: {job_title}")
        print(f"Prediction: {prediction_result['prediction']} | Risk Score: {score}% | Level: {level}")
        print(f"ModernBERT Raw Fake: {prediction_result.get('model_raw_fake', 0)}% | Rule Security Risk: {prediction_result.get('rule_risk_raw', 0)}%")
        print(f"Composite Fake Prob: {prediction_result['fake_probability']}% | Legit Prob: {prediction_result['legitimate_probability']}%")
        print(f"Indicators: {flags}")
        print("=" * 60 + "\n")

        return redirect(url_for("dashboard", scan=scan_code) + "#resultOverview")

    # GET: RETRIEVE USER STATS & SCANS
    user_scans = Scan.query.filter_by(user_id=user_id).order_by(Scan.date_analyzed.desc()).all()
    total_scans = len(user_scans)
    high_risk_count = sum(1 for s in user_scans if s.risk_score >= 50)
    safe_count = sum(1 for s in user_scans if s.risk_score < 25)
    suspicious_count = sum(1 for s in user_scans if 25 <= s.risk_score < 50)

    # RETRIEVE SCAN TO DISPLAY
    latest_analysis = None
    target_scan_code = request.args.get("scan")

    scan_to_display = None
    if target_scan_code:
        scan_to_display = Scan.query.filter_by(scan_code=target_scan_code, user_id=user_id).first()

    if scan_to_display:
        fake_prob = scan_to_display.fake_prob if scan_to_display.fake_prob is not None else float(scan_to_display.risk_score)
        legit_prob = scan_to_display.legit_prob if scan_to_display.legit_prob is not None else round(100.0 - fake_prob, 2)
        confidence = scan_to_display.confidence if scan_to_display.confidence is not None else round(max(fake_prob, legit_prob), 2)
        pred_label = scan_to_display.prediction if scan_to_display.prediction else ("Fake" if fake_prob >= 50 else ("Suspicious" if fake_prob >= 25 else "Legitimate"))

        raw_indicators = (scan_to_display.indicators or "").split(" | ")
        clean_indicators = [i.strip() for i in raw_indicators if i.strip()]

        extracted_info = extract_job_details(scan_to_display.job_text or "")
        _, _, scam_category = evaluate_risk_indicators(scan_to_display.job_text or "")

        latest_analysis = {
            "scan_code": scan_to_display.scan_code,
            "job_title": scan_to_display.job_title,
            "company_name": scan_to_display.company_name,
            "prediction": pred_label,
            "legitimate_probability": round(legit_prob, 2),
            "fake_probability": round(fake_prob, 2),
            "confidence": round(confidence, 2),
            "risk_score": scan_to_display.risk_score,
            "trust_score": round(legit_prob, 2),
            "risk_level": scan_to_display.risk_level,
            "indicators": clean_indicators,
            "job_text": scan_to_display.job_text or "",
            "salary": extracted_info["salary"],
            "location": extracted_info["location"],
            "email": extracted_info["email"],
            "category": scam_category
        }

    is_first_visit = session.pop("is_first_visit", False)

    return render_template(
        "dashboard.html",
        user=user,
        scans=user_scans,
        total_scans=total_scans,
        high_risk=high_risk_count,
        safe_jobs=safe_count,
        suspicious_jobs=suspicious_count,
        latest_analysis=latest_analysis,
        is_first_visit=is_first_visit
    )


# ============================================================
# HISTORY & PROFILE (WITH USER SCAN DELETION)
# ============================================================

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("auth", mode="login"))

    user_scans = Scan.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Scan.date_analyzed.desc()).all()

    scans_dict = {}
    for s in user_scans:
        fake_prob = s.fake_prob if s.fake_prob is not None else float(s.risk_score)
        legit_prob = s.legit_prob if s.legit_prob is not None else round(100.0 - fake_prob, 2)
        confidence = s.confidence if s.confidence is not None else round(max(fake_prob, legit_prob), 2)
        pred = s.prediction if s.prediction else ("Fake" if fake_prob >= 50 else "Legitimate")
        scans_dict[str(s.id)] = {
            "id": s.id,
            "scan_code": s.scan_code,
            "job_title": s.job_title,
            "company_name": s.company_name,
            "date_analyzed": s.date_analyzed.strftime('%b %d, %Y') if s.date_analyzed else 'Recent',
            "risk_score": s.risk_score,
            "risk_level": s.risk_level,
            "prediction": pred,
            "fake_prob": fake_prob,
            "legit_prob": legit_prob,
            "confidence": confidence,
            "indicators": s.indicators or '',
            "job_text": s.job_text or ''
        }

    return render_template("history.html", scans=user_scans, scans_json=json.dumps(scans_dict))


@app.route("/delete_scan/<int:scan_id>", methods=["GET", "POST"])
def user_delete_scan(scan_id):
    if "user_id" not in session:
        return redirect(url_for("auth", mode="login"))

    # Security check: users can only delete their OWN scans
    scan_to_delete = Scan.query.filter_by(
        id=scan_id,
        user_id=session["user_id"]
    ).first_or_404()

    db.session.delete(scan_to_delete)
    db.session.commit()
    flash(f"Scan record {scan_to_delete.scan_code} was successfully deleted.", "success")
    return redirect(url_for("history"))


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("auth", mode="login"))

    user = User.query.get(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("auth", mode="login"))

    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email = request.form.get("email", "").strip().lower()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not fullname:
            flash("Full name cannot be empty.", "error")
            return redirect(url_for("profile"))

        if not is_valid_email(email):
            flash("Please enter a valid email address.", "error")
            return redirect(url_for("profile"))

        existing_user = User.query.filter(User.email == email, User.id != user.id).first()
        if existing_user:
            flash("This email address is already in use.", "error")
            return redirect(url_for("profile"))

        if new_password:
            if len(new_password) < 6:
                flash("Password must contain at least 6 characters.", "error")
                return redirect(url_for("profile"))
            if new_password != confirm_password:
                flash("New passwords do not match.", "error")
                return redirect(url_for("profile"))

            user.password_hash = generate_password_hash(new_password, method="pbkdf2:sha256")

        user.fullname = fullname
        user.email = email
        db.session.commit()

        session["user_name"] = user.fullname
        session["user_email"] = user.email
        flash("Profile settings updated successfully!", "success")
        return redirect(url_for("profile"))

    return render_template("profile.html", user=user)


# ============================================================
# ADMIN MODULE
# ============================================================

@app.route("/admin")
@admin_required
def admin_dashboard():
    users = User.query.order_by(User.created_at.desc()).all()
    all_scans = Scan.query.order_by(Scan.date_analyzed.desc()).all()

    total_users = len(users)
    total_scans = len(all_scans)
    fake_scans = sum(1 for s in all_scans if s.risk_score >= 50)
    suspicious_scans = sum(1 for s in all_scans if 25 <= s.risk_score < 50)
    genuine_scans = sum(1 for s in all_scans if s.risk_score < 25)
    scam_category_data = build_scam_category_data(all_scans)

    return render_template(
        "admin.html",
        users=users,
        scans=all_scans,
        total_users=total_users,
        total_scans=total_scans,
        fake_scans=fake_scans,
        suspicious_scans=suspicious_scans,
        genuine_scans=genuine_scans,
        scam_category_data=json.dumps(scam_category_data)
    )


@app.route("/admin/profile", methods=["GET", "POST"])
@admin_required
def admin_profile():
    admin_user = User.query.get(session["user_id"])

    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email = request.form.get("email", "").strip().lower()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not fullname:
            flash("Full name cannot be empty.", "error")
            return redirect(url_for("admin_profile"))

        if not is_valid_email(email):
            flash("Please enter a valid email address.", "error")
            return redirect(url_for("admin_profile"))

        existing_user = User.query.filter(User.email == email, User.id != admin_user.id).first()
        if existing_user:
            flash("This email address is already in use.", "error")
            return redirect(url_for("admin_profile"))

        if new_password:
            if len(new_password) < 6:
                flash("Password must contain at least 6 characters.", "error")
                return redirect(url_for("admin_profile"))
            if new_password != confirm_password:
                flash("New passwords do not match.", "error")
                return redirect(url_for("admin_profile"))

            admin_user.password_hash = generate_password_hash(new_password, method="pbkdf2:sha256")

        admin_user.fullname = fullname
        admin_user.email = email
        db.session.commit()

        session["user_name"] = admin_user.fullname
        session["user_email"] = admin_user.email
        flash("Admin profile updated successfully!", "success")
        return redirect(url_for("admin_profile"))

    return render_template("admin_profile.html", admin=admin_user)


@app.route("/admin/toggle_user/<int:user_id>", methods=["POST"])
@admin_required
def toggle_user_status(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == session.get("user_id"):
        flash("You cannot suspend your own admin account.", "error")
        return redirect(url_for("admin_dashboard"))

    user.is_active = not user.is_active
    db.session.commit()
    status_str = "activated" if user.is_active else "suspended"
    flash(f"User account {user.email} has been {status_str}.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    if user_id == session.get("user_id"):
        flash("You cannot delete your own admin account while logged in.", "error")
        return redirect(url_for("admin_dashboard"))

    user_to_delete = User.query.get_or_404(user_id)
    db.session.delete(user_to_delete)
    db.session.commit()
    flash("User account deleted successfully.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete_scan/<int:scan_id>", methods=["POST"])
@admin_required
def delete_scan(scan_id):
    scan_to_delete = Scan.query.get_or_404(scan_id)
    db.session.delete(scan_to_delete)
    db.session.commit()
    flash("Scan entry removed from system logs.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/export_scans")
@admin_required
def export_scans_csv():
    scans = Scan.query.order_by(Scan.date_analyzed.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Scan Code", "User Name", "User Email", "Job Title",
        "Company", "Risk Score", "Risk Level", "Date Analyzed"
    ])

    for s in scans:
        writer.writerow([
            s.scan_code,
            s.user.fullname,
            s.user.email,
            s.job_title,
            s.company_name,
            f"{s.risk_score}%",
            s.risk_level,
            s.date_analyzed.strftime("%Y-%m-%d %H:%M:%S")
        ])

    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=jobshield_scans_report.csv"
    return response


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():
    with app.app_context():
        db.create_all()

        with db.engine.connect() as conn:
            from sqlalchemy import text
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN prediction VARCHAR(20)"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN fake_prob FLOAT"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN legit_prob FLOAT"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN confidence FLOAT"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN scan_category VARCHAR(100)"))
            except Exception:
                pass

        admin_user = User.query.filter_by(email="admin@jobshield.ai").first()
        if not admin_user:
            admin_user = User(
                fullname="System Admin",
                email="admin@jobshield.ai",
                password_hash=generate_password_hash("admin123", method="pbkdf2:sha256"),
                is_admin=True,
                is_active=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Default admin verified: admin@jobshield.ai / admin123")

init_db()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)