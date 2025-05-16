from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(text):
    embeddings = model.encode([text], convert_to_tensor=True)
    return embeddings

def getScore(original_text,summary_text):
  embed1 = get_embeddings(original_text)
  embed2 = get_embeddings(summary_text)
  score = cosine_similarity(embed1,embed2)
  return score[0][0]



# text1 = "Kidney disease is an asymptomatic disease, which leads to severe complications or even mortality if not diagnosed early. Routine diagnostic methods, such as serum-based tests and biopsies, are either less effective in the early stages of the disease. This paper proposes an automatic detection of kidney disease using CNNs applied to medical imaging data. Our model is designed to analyze computed tomography (CT) images for the identification of kidney disease, classifying normal and tumors. The proposed CNN architecture leverages deep learning techniques to extract features from these images and classify them with high accuracy. This paper aims to build a system for detection of kidney disease using CNN, based on a public dataset sourced from Kaggle."
# text2 = "Kidney disease is an asymptomatic condition that can lead to serious complications or even death if not diagnosed early. This paper proposes automatic kidney disease detection using CNN applied to medical image data to detect kidney disease in the early stages of the disease. the aim of this article is to develop a kidney disease detection system using CNN based on a public dataset from Kaggle. the proposed architecture uses deep learning techniques to extract features from computed tomography CT images and classify them with high accuracy to detect normal and tumor kidney disease."
# getScore(text1,text2)