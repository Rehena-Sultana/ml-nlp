import streamlit as st
import requests

BASE_URL = " https://ml-nlp-9r7t.onrender.com"  # change after deploying to Render
SINGLE_URL = f"{BASE_URL}/preprocess/single"
FULL_URL = f"{BASE_URL}/preprocess"

st.set_page_config(page_title="Text Preprocessing Tool",  page_icon="📝")

st.markdown(
    "<h1 style='text-align:center; font-family:\"Times New Roman\", serif; color:blue'> 📝 Text Preprocessing Tool</h1>",
    unsafe_allow_html=True,
)
st.markdown("<br> ", unsafe_allow_html=True)

st.markdown("Enter text below, then click a specific technique **or** run the full pipeline.")

text_input = st.text_area("Enter your text here:", height=150, placeholder="Type or paste your text...")


def call_api(url, payload):
    try:
        with st.spinner("Processing... please wait ⏳"):
            response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            st.error(f"❌ API Error: {response.status_code}")
            st.write(response.json())
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API server. Please try again in a minute.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ Server is waking up from sleep. Please click the button again in 1-2 minutes.")
        return None


@st.dialog("Result")
def show_single_result(title, output):
    st.subheader(title)
    st.write(output)


ACCENT_COLOR = "#2F02AC"  # change to any hex color you like


def technique_header(label):
    st.markdown(
        f"<h3 style='color:{ACCENT_COLOR}; font-size:20px; margin-top:18px;'>{label}</h3>",
        unsafe_allow_html=True,
    )


@st.dialog("Full Pipeline Result", width="large")
def show_full_result(data):
    technique_header("No HTML Tags")
    st.write(data["no_html"])

    technique_header("No URLs")
    st.write(data["no_urls"])

    technique_header("Lowercased")
    st.write(data["lowercased"])

    technique_header("No Punctuation")
    st.write(data["no_punctuation"])

    technique_header("No Numbers")
    st.write(data["no_numbers"])

    technique_header("Cleaned Text (extra spaces removed)")
    st.write(data["cleaned_text"])

    technique_header("Tokens")
    st.write(data["tokens"])

    technique_header("Tokens Without Stopwords")
    st.write(data["tokens_without_stopwords"])

    technique_header("Stemmed Words")
    st.write(data["stemmed_words"])

    technique_header("Lemmatized Words")
    st.write(data["lemmatized_words"])

    technique_header("POS Tags")
    st.write(data["pos_tags"])


def run_single(technique_key, label):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
        return
    result = call_api(SINGLE_URL, {"text": text_input, "technique": technique_key})
    if result:
        show_single_result(label, result["output"])


def run_full():
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
        return
    result = call_api(FULL_URL, {"text": text_input})
    if result:
        show_full_result(result)


st.markdown("### 🧹 Cleaning Steps")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🔡 Lowercase"):
        run_single("lowercase", "Lowercased Text")
    if st.button("❌ Remove Punctuation"):
        run_single("remove_punctuation", "Text Without Punctuation")
with c2:
    if st.button("🔢 Remove Numbers"):
        run_single("remove_numbers", "Text Without Numbers")
    if st.button("␣ Remove Extra Spaces"):
        run_single("remove_extra_spaces", "Text Without Extra Spaces")
with c3:
    if st.button("🏷️ Remove HTML Tags"):
        run_single("remove_html_tags", "Text Without HTML Tags")
    if st.button("🔗 Remove URLs"):
        run_single("remove_urls", "Text Without URLs")

st.markdown("### 🔍 NLP Techniques")
n1, n2, n3 = st.columns(3)
with n1:
    if st.button("🔠 Tokenize"):
        run_single("tokenize", "Tokens")
    if st.button("🚫 Remove Stopwords"):
        run_single("remove_stopwords", "Tokens Without Stopwords")
with n2:
    if st.button("🌱 Stem"):
        run_single("stem", "Stemmed Words")
    if st.button("📖 Lemmatize"):
        run_single("lemmatize", "Lemmatized Words")
with n3:
    if st.button("🏷️ POS Tag"):
        run_single("pos_tag", "POS Tags")

st.markdown("---")
if st.button("🚀 Run All (Full Pipeline)", type="primary"):
    run_full()