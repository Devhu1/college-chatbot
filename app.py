# from flask import Flask, request, render_template
# from fuzzywuzzy import process
# import json
#
# # Load FAQ data
# with open("faqs.json", "r") as f:
#     faqs = json.load(f)
# questions = [item["question"] for item in faqs]
#
# app = Flask(__name__)
#
# # Store chat history (in-memory)
# chat_history = []
#
# @app.route("/", methods=["GET", "POST"])
# def chatbot():
#     if request.method == "POST":
#         if "clear" in request.form:
#             chat_history.clear()
#         else:
#             user_input = request.form["query"]
#             match_result = process.extractOne(user_input, questions)
#
#             if match_result:
#                 match, score = match_result
#                 if score >= 70:
#                     bot_response = next(item["answer"] for item in faqs if item["question"] == match)
#                 else:
#                     bot_response = "Sorry, I couldn't understand your question. Please try rephrasing it."
#             else:
#                 bot_response = "Sorry, I couldn't process your question."
#
#             chat_history.append({"question": user_input, "answer": bot_response})
#
#     return render_template("chat.html", chat_history=chat_history)
#
# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, render_template, request, jsonify, redirect, url_for
# import json
# import difflib
#
# app = Flask(__name__)
# FAQ_FILE = 'faqs.json'
#
# # Load FAQs from JSON
# def load_faqs():
#     with open(FAQ_FILE, 'r') as f:
#         return json.load(f)
#
# # Save FAQs to JSON
# def save_faqs(faqs):
#     with open(FAQ_FILE, 'w') as f:
#         json.dump(faqs, f, indent=4)
#
# # Home route - chatbot interface
# @app.route("/")
# def index():
#     return render_template("chat.html")
#
# # Chatbot query route
# @app.route("/get", methods=["GET", "POST"])
# def get_bot_response():
#     print("Method:", request.method)
#     print("Headers:", request.headers)
#
#     if request.method == "POST":
#         try:
#             data = request.get_json()
#             user_query = data['msg'].lower()
#         except Exception as e:
#             print("Error parsing JSON:", e)
#             return "Invalid data sent.", 400
#     else:
#         user_query = request.args.get("msg").lower()
#
#     if not user_query:
#         return "No query received.", 400
#
#     faqs = load_faqs()
#
#     # Try finding the best matching question
#     questions = [faq["question"].lower() for faq in faqs]
#     match = difflib.get_close_matches(user_query, questions, n=1, cutoff=0.6)
#
#     if match:
#         for faq in faqs:
#             if faq["question"].lower() == match[0]:
#                 return faq["answer"]
#
#     return "Sorry, I didn't understand that. Please ask something related to admission or courses."
#
# # Admin panel to view & add FAQs
# @app.route("/admin")
# def admin_panel():
#     faqs = load_faqs()
#     return render_template("admin.html", faqs=faqs)
#
# # Route to add new FAQ
# @app.route("/add_faq", methods=["POST"])
# def add_faq():
#     question = request.form["question"]
#     answer = request.form["answer"]
#
#     faqs = load_faqs()
#     faqs.append({"question": question, "answer": answer})
#     save_faqs(faqs)
#
#     return redirect(url_for("admin_panel"))
#
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import json
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
FAQ_FILE = 'faqs.json'

# Load BERT model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAQs and compute embeddings once
def load_faqs_and_embeddings():
    with open(FAQ_FILE, 'r') as f:
        faqs = json.load(f)
    questions = [faq["question"] for faq in faqs]
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    return faqs, questions, question_embeddings

faqs, questions, question_embeddings = load_faqs_and_embeddings()

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    try:
        data = request.get_json()
        user_query = data.get("msg", "").strip()
        if not user_query:
            return "No query received.", 400
    except Exception as e:
        return f"Error parsing request: {str(e)}", 400

    # Encode user query
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Compute cosine similarity with FAQ questions
    scores = util.cos_sim(query_embedding, question_embeddings)[0]

    # Find best match
    best_match_idx = scores.argmax().item()
    best_score = scores[best_match_idx].item()

    if best_score >= 0.6:
        response = faqs[best_match_idx]["answer"]
    else:
        response = "Sorry, I didn't understand that. Please ask something related to admission or courses."

    return response

@app.route("/admin")
def admin_panel():
    return render_template("admin.html", faqs=faqs)

@app.route("/add_faq", methods=["POST"])
def add_faq():
    question = request.form["question"]
    answer = request.form["answer"]

    faqs.append({"question": question, "answer": answer})

    # Save updated FAQs
    with open(FAQ_FILE, 'w') as f:
        json.dump(faqs, f, indent=4)

    # Recompute embeddings
    global questions, question_embeddings
    questions = [faq["question"] for faq in faqs]
    question_embeddings = model.encode(questions, convert_to_tensor=True)

    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)



# from flask import Flask, request, render_template, redirect, url_for
# import json
# from transformers import BertTokenizer, BertModel
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
#
# app = Flask(__name__)
# FAQ_FILE = 'faqs.json'
#
# # Load FAQs
# def load_faqs():
#     with open(FAQ_FILE, 'r') as f:
#         return json.load(f)
#
# # BERT setup
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")
#
# # Encode a sentence using BERT
# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state[:, 0, :].numpy()  # Use CLS token
#
#
# @app.route("/")
# def index():
#     return render_template("chat.html")
#
#
# # Chatbot route
# @app.route("/get", methods=["GET", "POST"])
# def get_bot_response():
#     if request.method == "POST":
#         data = request.get_json()
#         user_query = data['msg']
#     else:
#         user_query = request.args.get("msg")
#
#     if not user_query:
#         return "No query received", 400
#
#     faqs = load_faqs()
#     questions = [faq["question"] for faq in faqs]
#     question_embeddings = [get_embedding(q) for q in questions]
#     user_embedding = get_embedding(user_query)
#
#     # Calculate cosine similarity
#     similarities = [cosine_similarity(user_embedding, q_emb)[0][0] for q_emb in question_embeddings]
#     best_index = similarities.index(max(similarities))
#
#     if similarities[best_index] > 0.7:  # similarity threshold
#         return faqs[best_index]["answer"]
#     else:
#         return "Sorry, I didn't understand that."
#
# if __name__ == "__main__":
#     app.run(debug=True)
#
