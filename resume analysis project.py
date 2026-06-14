from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    except Exception as e:
        print("PDF read error:", e)
    return text

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text.lower()

def get_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])
        return round(similarity[0][0] * 100, 2)
    except Exception as e:
        print("Similarity error:", e)
        return 0.0

def find_missing_skills(resume_text, jd_text):
    resume_words = set(resume_text.split())
    jd_words = set(jd_text.split())
    missing = jd_words - resume_words
    # filter very short words
    return sorted([w for w in missing if len(w) > 3])

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    missing_skills = []
    if request.method == "POST":
        # check file
        if "resume" not in request.files:
            return redirect(request.url)
        file = request.files["resume"]
        jd_text_raw = request.form.get("jobdesc", "")
        if file.filename == "" or not allowed_file(file.filename):
            result = "Please upload a PDF resume."
            return render_template("index.html", result=result)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # extract and process text
        resume_text_raw = extract_text_from_pdf(filepath)
        resume_text = clean_text(resume_text_raw)
        jd_text = clean_text(jd_text_raw)

        if not resume_text.strip():
            result = "Could not extract text from the uploaded PDF. Try a different resume."
            return render_template("index.html", result=result)

        score = get_similarity(resume_text, jd_text)
        missing_skills = find_missing_skills(resume_text, jd_text)
        result = f"{score} %"

        # optionally remove uploaded file after processing
        try:
            os.remove(filepath)
        except:
            pass

    return render_template("index.html", result=result, missing_skills=missing_skills)

if __name__ == "__main__":
    app.run(debug=True)
