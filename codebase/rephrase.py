from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
import spacy
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize


nlp = spacy.load("en_core_web_sm")

# model_name = "vennify/t5-base-grammar-correction"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# def correct_grammer(sentence, prefix="Correct the grammer "):
#     input_text = f"{prefix}: {sentence}"
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
#     outputs = model.generate(**inputs, max_length=100)
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if decoded.lower().startswith(prefix.lower()):
#         decoded = decoded[len(prefix):].lstrip(": ").strip()
#     return decoded

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

def synonym_paraphraser(sentence, replacement_rate):
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

def split_text(text):
    return sent_tokenize(text)

def convert_text(text):
    # sentences = split_text(text)
    sentences = text
    ans = ""
    for sentence in sentences:
        sentence = synonym_paraphraser(sentence, replacement_rate=0.3)
        # sentence = correct_grammer(sentence)
        ans += sentence.strip()
        if not sentence.strip().endswith((".", "!", "?")):
            ans += ". "
        else:
            ans += " "
    return ans.strip()
