import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# === Load LDA Topic Vectors ===
with open("lda_topic_vectors.pkl", "rb") as f:
    lda_topic_vectors = pickle.load(f)

lda_matrix = np.array(lda_topic_vectors)
similarity_matrix = cosine_similarity(lda_matrix)

# === Load Metadata and Abstracts ===
df = pd.read_excel("JIA_LDA_BERTopic_Combined.xlsx")
similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

# === App Title ===
st.title("📚 LDA Abstract Recommender")
st.markdown("Use topic-based filtering to explore similar abstracts.")

# === Filter: Select Abstract ===
doc_index = st.number_input("🔎 Select Abstract Index", min_value=0, max_value=len(df) - 1, value=0)
lda_topic_options = sorted(df["LDA_Topic"].unique().tolist())
selected_topic = st.selectbox("📌 Filter by LDA Topic", options=["All"] + lda_topic_options)
min_similarity = st.slider("📈 Minimum Similarity Score", 0.0, 1.0, 0.75)

# === Abstract Preview ===
st.markdown("### 📄 Selected Abstract")
st.markdown(f"**{df.iloc[doc_index, 0]}**")
st.markdown(f"**LDA Topic:** {df.iloc[doc_index]['LDA_Topic']}")
st.markdown(f"_Keywords:_ {df.iloc[doc_index]['LDA_Keywords']}")

# === Recommender Logic ===
def recommend_filtered(doc_index, top_n=20, topic_filter=None, min_sim=0.75):
    sims = similarity_matrix[doc_index]
    top_indices = sims.argsort()[::-1]
    top_indices = [i for i in top_indices if i != doc_index]
    
    results = []
    for i in top_indices:
        if sims[i] >= min_sim:
            if topic_filter == "All" or df.iloc[i]["LDA_Topic"] == topic_filter:
                results.append((i, sims[i], df.iloc[i, 0], df.iloc[i]["LDA_Keywords"]))
        if len(results) >= top_n:
            break
    return results

# === Results ===
st.markdown("### 🧠 Top Similar Abstracts")
results = recommend_filtered(doc_index, topic_filter=selected_topic, min_sim=min_similarity)

if results:
    for idx, sim, title, keywords in results:
        st.markdown(f"**{title}**  \n*Similarity: {sim:.2f}*  \n_Keywords: {keywords}_")
        st.markdown("---")
else:
    st.warning("No matching abstracts found with current filters.")

# === Export Option ===
if st.button("📥 Export Results to Excel"):
    rec_df = pd.DataFrame(results, columns=["Index", "Similarity", "APA Reference", "LDA Keywords"])
    filename = f"lda_filtered_recommendations_{doc_index}.xlsx"
    rec_df.to_excel(filename, index=False)
    st.success(f"Exported to {filename}")
