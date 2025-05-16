from transformers import BartTokenizer, BartForConditionalGeneration
import pdfplumber
import re
import math
import random
import string
import requests
from dotenv import load_dotenv
import os

load_dotenv()

rewriter_apikey = os.getenv("rewriter_apikey")

url = "https://rewriter-paraphraser-text-changer-multi-language.p.rapidapi.com/rewrite"

from codebase.rephrase import convert_text



# def introduce_human_errors(text, error_rate):
#     def random_typo(word):
#         if len(word) < 3 or random.random() > 0.4:
#             return word
        
#         typo_type = random.choice(['duplicate', 'delete', 'swap'])

#         if typo_type == 'duplicate':
#             pos = random.randint(1, len(word) - 2)
#             return word[:pos] + word[pos] + word[pos:]  # Duplicate a character

#         elif typo_type == 'delete':
#             pos = random.randint(0, len(word) - 1)
#             return word[:pos] + word[pos + 1:]  # Remove a character

#         elif typo_type == 'swap':
#             pos = random.randint(0, len(word) - 2)
#             return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]  # Swap adjacent characters

#         return word

#     def random_case_error(word):
#         if len(word) < 4 or random.random() > 0.4:  
#             return word
        
#         pos = random.randint(0, len(word) - 1)
#         letter = word[pos]
        
#         if letter.isalpha():
#             new_letter = letter.upper() if letter.islower() else letter.lower()
#             return word[:pos] + new_letter + word[pos + 1:]
        
#         return word

#     def random_spacing():
#         return "  " if random.random() < 0.6 else " "

#     def random_punctuation(word):
#         if len(word) < 5 or random.random() < 0.6:
#             return word

#         if random.random() < 0.5:
#             return word.rstrip(string.punctuation)  # Remove punctuation
#         else:
#             return word + random.choice([',', '.', '!'])  # Add punctuation

#     words = text.split()
#     for i in range(len(words)):
#         if random.random() < error_rate:
#             error_type = random.choices(
#                 ['typo', 'case', 'space', 'punctuation'], 
#                 weights=[0.4, 0.5, 0.8, 0.6]  # Reduced typo frequency
#             )[0]
            
#             if error_type == 'typo':
#                 words[i] = random_typo(words[i])
#             elif error_type == 'case':
#                 words[i] = random_case_error(words[i])
#             elif error_type == 'space' and i > 0:
#                 words[i] = random_spacing() + words[i]
#             elif error_type == 'punctuation':
#                 words[i] = random_punctuation(words[i])

#     return ' '.join(words)

def introduce_human_errors(text, error_rate):
    keyboard_neighbors = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx', 'e': 'wsdr',
        'f': 'rtgdvc', 'g': 'tyhfvb', 'h': 'yugjnb', 'i': 'ujko', 'j': 'uikmnh',
        'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
        'z': 'asx'
    }

    common_grammar_confusions = {
        'their': 'there', 'there': 'their', 'your': 'you\'re', 'you\'re': 'your',
        'its': 'it\'s', 'it\'s': 'its', 'then': 'than', 'than': 'then'
    }

    def random_typo(word):
        if len(word) < 3:
            return word
        typo_type = random.choice(['duplicate', 'delete', 'swap', 'substitute'])

        if typo_type == 'duplicate':
            pos = random.randint(1, len(word) - 2)
            return word[:pos] + word[pos] + word[pos:]

        elif typo_type == 'delete':
            pos = random.randint(0, len(word) - 1)
            return word[:pos] + word[pos + 1:]

        elif typo_type == 'swap':
            pos = random.randint(0, len(word) - 2)
            return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]

        elif typo_type == 'substitute':
            pos = random.randint(0, len(word) - 1)
            char = word[pos].lower()
            if char in keyboard_neighbors:
                new_char = random.choice(keyboard_neighbors[char])
                return word[:pos] + new_char + word[pos + 1:]
        
        return word

    def random_case_error(word):
        if not any(c.isalpha() for c in word):
            return word
        pos = random.randint(0, len(word) - 1)
        c = word[pos]
        if c.islower():
            return word[:pos] + c.upper() + word[pos + 1:]
        else:
            return word[:pos] + c.lower() + word[pos + 1:]

    def random_spacing(word):
        return "  " + word if random.random() < 0.5 else word + "  "

    def random_punctuation(word):
        if word[-1] in string.punctuation:
            word = word[:-1]
        if random.random() < 0.5:
            return word
        return word + random.choice([',', '.', '!', '...'])

    def grammar_confuse(word):
        return common_grammar_confusions.get(word.lower(), word)

    words = text.split()
    modified_words = []

    for word in words:
        modified_word = word
        if random.random() < error_rate:
            error_type = random.choices(
                ['typo', 'case', 'spacing', 'punctuation', 'grammar'],
                weights=[0.3, 0.2, 0.15, 0.2, 0.15],  # Balanced weights
                k=1
            )[0]

            if error_type == 'typo':
                modified_word = random_typo(modified_word)
            elif error_type == 'case':
                modified_word = random_case_error(modified_word)
            elif error_type == 'spacing':
                modified_word = random_spacing(modified_word)
            elif error_type == 'punctuation':
                modified_word = random_punctuation(modified_word)
            elif error_type == 'grammar':
                modified_word = grammar_confuse(modified_word)

        modified_words.append(modified_word)

    return ' '.join(modified_words)



def paraphrase_text(text) :
    payload = {
	"language": "en",
	"strength": 3,
	"text": text
    }
    headers = {
        "x-rapidapi-key": rewriter_apikey,
        "x-rapidapi-host": "rewriter-paraphraser-text-changer-multi-language.p.rapidapi.com",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    return data["rewrite"]

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9.,!?\'"%\-\s]', '', text)
    text = re.sub(r'\s([?.!,"\'-])', r'\1', text)
    text = re.sub(r'([?.!,"\'-])([^\s])', r'\1 \2', text)
    text = re.sub(r'\b(\d+\.){2,}\d+\b', '', text)
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()

    if text and text[0].islower():
        text = text[0].upper() + text[1:]
        
    if text and text[-1] not in {'.', '!', '?'}:
        text += '.'
    return text

def generate_summary(text, max_length, min_length, num_beams,model_type,length_penalty):
    
    tokenizer = BartTokenizer.from_pretrained(f'./Models/{model_type}')
    model = BartForConditionalGeneration.from_pretrained(f'./Models/{model_type}')
    
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(
        
        inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_pdftext(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    text = clean_text(text)
    return text

def split_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    return sentence_endings.split(text)

def get_paraphrased_sentences(text):
    sentences = split_into_sentences(text)
    paraphrased_sentences = []
    
    for sentence in sentences:
        if sentence:
            paraphrased_sentence = paraphrase_text(sentence)
            if paraphrased_sentence:
                paraphrased_sentences.append(paraphrased_sentence)
 
    return paraphrased_sentences

def get_summary(paraphrased_sentences,min_limit,max_limit,qualityIndex, model_type):
    
    N = len(paraphrased_sentences)
    no_of_block=0
    if N > 0:
        block_size = math.ceil(math.sqrt(N))
        no_of_block = math.ceil(N/block_size)
    
    
    print("\nNo of blocks : ",N)
    Summary_sentences = []

    lengths_box = [0.9,0.5,0.8,0.4,0.6,0.7,0.3]

    if no_of_block > 0:
        for i in range(0, N, block_size):
            
            sentence_block = paraphrased_sentences[i:i + block_size]
            combined_sentence = ' '.join(sentence_block)
            
            print("\nSentence Block :\n")
            print(combined_sentence)

            word_count = len(combined_sentence.split())
            print("\nWord count : ",word_count)
            
            min_word_count = int(min_limit/no_of_block)
            max_word_count = int(max_limit/no_of_block)
            
            if(min_word_count > word_count):
                min_word_count = word_count-15
        
            print("Min Word : ",min_word_count)
            print("Max word : ",max_word_count)
            length_penalty = random.choice(lengths_box)
            summary_text = generate_summary(combined_sentence,max_word_count,min_word_count, qualityIndex, model_type,length_penalty)
            
            print("\nSummary Text :\n")
            print(summary_text)
        
            Summary_sentences.append(summary_text)
            
        combined_summary = ' '.join(Summary_sentences)
        
        return combined_summary
    
    return "Error in paraphresing text."




def get_humanized_text(paraphrased_sentences):
    
    N = len(paraphrased_sentences)
    no_of_block=0
    if N > 0:
        block_size = math.ceil(math.sqrt(N))
        no_of_block = math.ceil(N/block_size)
    
    
    print("\nNo of blocks : ",N)
    Summary_sentences = []

    if no_of_block > 0:
        for i in range(0, N, block_size):
            
            sentence_block = paraphrased_sentences[i:i + block_size]
            combined_sentence = ' '.join(sentence_block)

            summary_text = convert_text(combined_sentence)
        
            Summary_sentences.append(summary_text)
            
        combined_summary = ' '.join(Summary_sentences)
        
        return combined_summary
    
    return "Error in paraphresing text."


def humanize_text(text,qualityIndex,max_limit,min_limit,model_type):
    if not text:
        return "Error"
    
    paraphrased_sentences = get_paraphrased_sentences(text)
    
    print("\ncombined_paraphrased_text\n")
    print(paraphrased_sentences)
    
    summary = get_summary(paraphrased_sentences,min_limit,max_limit,qualityIndex, model_type)
    # summary = get_humanized_text(paraphrased_sentences)
    summary = clean_text(summary)

    generated_summary = convert_text(paraphrased_sentences)

    print("\nGenerated Summary")
    print(generated_summary)
    
    generated_summary = introduce_human_errors(generated_summary,error_rate=0.3)
    return generated_summary