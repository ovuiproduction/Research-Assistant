# import ollama

# def paraphrase_text(text):
#     prompt = f"""
# You are a human-like paraphraser model.
# Your task is to rewrite the following text to sound natural, human-written, and undetectable as AI-generated.

# Guidelines:
# - Rephrase using natural language and common human expressions.
# - Vary sentence length and structure to avoid robotic patterns.
# - Use creative synonyms or rewording that still preserve the original meaning.
# - Introduce subtle imperfections (like informal transitions, filler words, or mild redundancy) that mimic human writing.
# - Avoid overly precise or formulaic wording; write with a touch of personality or tone.
# - Make the flow sound spontaneous, like how a thoughtful human might write—not like a polished AI response.
# - Use contractions (e.g., "don't", "it's") where appropriate for a conversational tone.
# - Add small rhetorical touches (e.g., "you know," "actually," "well," etc.) if it fits naturally.

# ### Output format:
# - Only output the paraphrased version.
# - Do not include explanations, disclaimers, or formatting instructions.

# Text to paraphrase:
# {text}
# """

#     response = ollama.chat(
#         model='mistral',
#         messages=[{'role': 'user', 'content': prompt}]
#     )
    
#     content = response['message']['content'].strip()
#     print(content)
#     return content




# paraphrase_text(''' Citations in the legal field relate to earlier rulings 
# cited in support of the current case. Attorneys use citations to 
# create compelling arguments and ensure uniformity in rulings. 
# However, for attorneys, the process is difficult and time
# consuming because it's like needle-hunting to identify pertinent 
# quotations from a vast number of judgments. This procedure is 
# greatly improved by Legal Citation Recommendation Systems 
# (LCRS), which rapidly find the most pertinent citations. LCRS 
# typically evaluates the pairwise similarity between judgments, 
# however, problems occur because of the judgments' uneven 
# lengths and information overload. The similarity score is 
# directly impacted by these difficulties, which also result in 
# additional noise, semantic dilution effects, size-induced 
# similarity degradation, and dimensional inconsistencies. 
# Research suggests a technique to deal with similarity 
# deterioration in which assessments are divided into different 
# pieces using regular expressions. The sections are chosen after 
# consulting subject-matter experts. Because a judgment has 
# several portions, summarization and semantic chunking are 
# used to construct sections of the right size while addressing 
# dimensional inconsistencies and noise. This method 
# concentrates on discovering similarities between matching 
# portions rather than similarities between full judgments. A 
# more accurate estimate of similarity is then obtained by 
# calculating the average of these section-wise similarities. The 
# preference or precedence of parts based on user requirements is 
# also incorporated into this strategy. The LCRS becomes more 
# dynamic and more in line with user needs when parts are given 
# weighted similarity values. 
#  ''')

import nltk
import random
import spacy
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")

def get_synonym(word, pos_tag):
    """Return a synonym of the word using WordNet, or the original word if none found."""
    pos_map = {
        'NOUN': wn.NOUN,
        'VERB': wn.VERB,
        'ADJ': wn.ADJ,
        'ADV': wn.ADV
    }
    wn_pos = pos_map.get(pos_tag, None)
    if wn_pos is None:
        return word

    synonyms = wn.synsets(word, pos=wn_pos)
    lemmas = set()
    for syn in synonyms:
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                lemmas.add(lemma.name().replace("_", " "))
    if lemmas:
        return random.choice(list(lemmas))
    return word

def synonym_paraphraser(sentence, replacement_rate=0.3):
    """Replace some words in the sentence with synonyms based on replacement_rate."""
    doc = nlp(sentence)
    new_tokens = []

    for token in doc:
        if (
            token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and 
            not token.is_stop and 
            token.is_alpha and 
            random.random() < replacement_rate
        ):
            synonym = get_synonym(token.text, token.pos_)
            new_tokens.append(synonym)
        else:
            new_tokens.append(token.text)

    return " ".join(new_tokens)


text = """
Kidney disease is an asymptomatic disease, which leads to severe complications or even mortality if not diagnosed early. Routine diagnostic methods, such as serum-based tests and biopsies, are either less effective in the early stages of the disease. This paper proposes an automatic detection of kidney disease using CNNs applied to medical imaging data. Our model is designed to analyze computed tomography (CT) images for the identification of kidney disease, classifying normal and tumors. The proposed CNN architecture leverages deep learning techniques to extract features from these images and classify them with high accuracy. This paper aims to build a system for detection of kidney disease using CNN, based on a public dataset sourced from Kaggle."
"""
print(synonym_paraphraser(text, replacement_rate=0.3))
