import streamlit as st
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. تحميل أدوات NLTK الضرورية لمعالجة النصوص
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')

download_nltk_data()

# 2. تحميل البيانات (FAQs)
@st.cache_data
def load_faqs(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

# جلب الأسئلة والأجوبة
faqs = load_faqs('faqs.json')
questions = [faq['question'] for faq in faqs]
answers = [faq['answer'] for faq in faqs]

# 3. معالجة النصوص وحساب TF-IDF
# استعملنا NLTK word_tokenize كـ tokenizer
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english', token_pattern=None)
X = vectorizer.fit_transform(questions)

# 4. دالة المطابقة (Intent Matching)
def get_best_response(user_question):
    # تحويل سؤال المستخدم لـ Vector
    user_vec = vectorizer.transform([user_question])
    
    # حساب نسبة التشابه (Cosine Similarity)
    similarities = cosine_similarity(user_vec, X)
    best_match_idx = np.argmax(similarities)
    
    # إذا كانت نسبة التشابه أقل من 0.2، يعني السؤال مامفهومش
    if similarities[0][best_match_idx] < 0.2:
        return "I'm sorry, I don't have an answer for that in my FAQs. Can you rephrase?"
    
    # إرجاع الجواب الأقرب
    return answers[best_match_idx]

# ==========================================
# 5. واجهة المستخدم (Streamlit Chat UI)
# ==========================================
st.title("🤖 Smart FAQ Chatbot")
st.write("Welcome! Ask me anything about our services.")

# تهيئة سجل المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثات السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# التفاعل مع المستخدم
if prompt := st.chat_input("Type your question here..."):
    # عرض سؤال المستخدم وحفظه
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # توليد الجواب وعرضه
    response = get_best_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)