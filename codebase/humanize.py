from transformers import BartTokenizer, BartForConditionalGeneration
import pdfplumber
import re
import math
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
import random
import nltk
import random
import string
import requests

url = "https://rewriter-paraphraser-text-changer-multi-language.p.rapidapi.com/rewrite"

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Function to get simple synonyms of a word
def introduce_human_errors(text, error_rate=0.6):
    def random_typo(word):
        if len(word) < 3 or random.random() > 0.4:
            return word
        
        typo_type = random.choice(['duplicate', 'delete', 'swap'])

        if typo_type == 'duplicate':
            pos = random.randint(1, len(word) - 2)
            return word[:pos] + word[pos] + word[pos:]  # Duplicate a character

        elif typo_type == 'delete':
            pos = random.randint(0, len(word) - 1)
            return word[:pos] + word[pos + 1:]  # Remove a character

        elif typo_type == 'swap':
            pos = random.randint(0, len(word) - 2)
            return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]  # Swap adjacent characters

        return word

    def random_case_error(word):
        if len(word) < 4 or random.random() > 0.4:  
            return word
        
        pos = random.randint(0, len(word) - 1)
        letter = word[pos]
        
        if letter.isalpha():
            new_letter = letter.upper() if letter.islower() else letter.lower()
            return word[:pos] + new_letter + word[pos + 1:]
        
        return word

    def random_spacing():
        return "  " if random.random() < 0.6 else " "

    def random_punctuation(word):
        if len(word) < 5 or random.random() < 0.6:
            return word

        if random.random() < 0.5:
            return word.rstrip(string.punctuation)  # Remove punctuation
        else:
            return word + random.choice([',', '.', '!'])  # Add punctuation

    words = text.split()
    for i in range(len(words)):
        if random.random() < error_rate:
            error_type = random.choices(
                ['typo', 'case', 'space', 'punctuation'], 
                weights=[0.4, 0.5, 0.8, 0.6]  # Reduced typo frequency
            )[0]
            
            if error_type == 'typo':
                words[i] = random_typo(words[i])
            elif error_type == 'case':
                words[i] = random_case_error(words[i])
            elif error_type == 'space' and i > 0:
                words[i] = random_spacing() + words[i]
            elif error_type == 'punctuation':
                words[i] = random_punctuation(words[i])

    return ' '.join(words)


def paraphrase_text(text) :
    payload = {
	"language": "en",
	"strength": 3,
	"text": text
    }
    headers = {
        "x-rapidapi-key": "d5a7b3bc98msh178cdee21b4d606p188169jsncc7fa65cefe8",
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

def generate_summary(text, max_length, min_length, num_beams,model_type):
    
    tokenizer = BartTokenizer.from_pretrained(f'./Models/{model_type}')
    model = BartForConditionalGeneration.from_pretrained(f'./Models/{model_type}')
    
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(
        
        inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=0.9,
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

# def get_paraphrased_sentences(text):
#     sentences = split_into_sentences(text)
#     paraphrased_sentences = []

#     # Group sentences in chunks of three
#     for i in range(0, len(sentences), 3):
#         sentence_group = ' '.join(sentences[i:i + 3])
#         if sentence_group:
#             paraphrased_group = paraphrase_text(sentence_group)
#             if paraphrased_group:
#                 paraphrased_sentences.append(paraphrased_group)

#     return paraphrased_sentences

def get_paraphrased_sentences(text):
    sentences = split_into_sentences(text)
    paraphrased_sentences = []
    
    for sentence in sentences:
        if sentence:
            paraphrased_sentence = paraphrase_text(sentence)
            if paraphrased_sentence:
                paraphrased_sentences.append(paraphrased_sentence)
                
    # combined_paraphrased_text = ' '.join(paraphrased_sentences)
    
    return paraphrased_sentences

# def get_summary(paraphrased_sentences,min_limit,max_limit,qualityIndex, model_type):
#     N = len(paraphrased_sentences)
#     no_of_block=N
   
#     print("\nNo of blocks : ",N)
#     Summary_sentences = []

#     if no_of_block > 0:
#         for i in range(0, N):
        
#             combined_sentence = paraphrased_sentences[i]
#             # combined_sentence = ' '.join(sentence_block)
            
#             print("\nSentence Block :\n")
#             print(combined_sentence)

#             word_count = len(combined_sentence.split())
#             print("\nWord count : ",word_count)
            
#             min_word_count = int(min_limit/no_of_block)
#             max_word_count = int(max_limit/no_of_block)
            
#             if(min_word_count > word_count):
#                 min_word_count = word_count-15
        
#             print("Min Word : ",min_word_count)
#             print("Max word : ",max_word_count)
            
#             summary_text = generate_summary(combined_sentence,max_word_count,min_word_count, qualityIndex, model_type)
            
#             print("\nSummary Text :\n")
#             print(summary_text)
        
#             Summary_sentences.append(summary_text)
            
#         combined_summary = ' '.join(Summary_sentences)
        
#         return combined_summary
    
#     return "Error in paraphresing text."


def get_summary(paraphrased_sentences,min_limit,max_limit,qualityIndex, model_type):
    
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
            
            summary_text = generate_summary(combined_sentence,max_word_count,min_word_count, qualityIndex, model_type)
            
            print("\nSummary Text :\n")
            print(summary_text)
        
            Summary_sentences.append(summary_text)
            
        combined_summary = ' '.join(Summary_sentences)
        
        return combined_summary
    
    return "Error in paraphresing text."


# def paraphrase_text(text):
#     prompt = f"""
#     You are a human-like paraphraser model. 
#     Your job is to rephrase the given text to make it sound natural, human-written, and coherent, ensuring it cannot be detected as AI-generated content.

#     Guidelines:
#     - Use human writing style, including natural transitions and varied sentence structures.
#     - Maintain the original meaning while improving fluency and readability.
#     - Avoid using overly technical or mechanical phrasing common in AI-generated outputs.

#     ### Output format:
#     - Provide only the paraphrased text.
#     - No explanations, disclaimers, or additional statements.

#     Examples:

#     AI Generated:  
#     Kidney disease is an asymptomatic disease, which leads to severe complications or even mortality if not diagnosed early. Routine diagnostic methods, such as serum-based tests and biopsies, are either less effective in the early stages of the disease. This paper proposes an automatic detection of kidney disease using CNNs applied to medical imaging data. Our model is designed to analyze computed tomography (CT) images for the identification of kidney disease, classifying normal and tumors. The proposed CNN architecture leverages deep learning techniques to extract features from these images and classify them with high accuracy. This paper aims to build a system for detection of kidney disease using CNN, based on a public dataset sourced from Kaggle.

#     Humanized:  
#     Kidney disease is an asymptomatic illness, and hence kidney disease may result in serious complications or even death if not detected in the early stages. Standard diagnostic techniques, like serum tests and biopsies, are either not efficient in the early stages of the disease. It is in this context that this paper suggests an automatic diagnosis of kidney disease through CNNs on medical imaging data. Our model is trained to examine computed tomography (CT) scans for kidney disease identification, categorizing normal and tumors. The suggested CNN architecture utilizes deep learning methods to extract features from the images and classify them accurately. This work focuses on constructing a system for kidney disease detection using CNN based on a public dataset obtained from Kaggle.

#     AI Generated:  
#     In recent years, advancements in machine learning and deep learning have led way for more sophisticated diagnostic tools. Convolutional Neural Network, a subset of deep learning models designed for image data processing, have shown exceptional performance in medical imaging tasks, such as tumor detection, retinal disease classification, and lung disease identification. By leveraging CNNs for kidney disease detection, we can potentially improve diagnostic accuracy and reduce the need for invasive procedures. In this paper, we propose a CNN-based framework for the detection and classification of kidney diseases using medical image data, such as computed tomography scans. The proposed model is trained to identify patterns in kidney images that are indicative of either normal kidney or tumor.

#     Humanized:  
#     With advances in deep learning and machine learning in the recent past, there has been way for more complex diagnostic technologies. Convolutional Neural Network, a branch of deep learning architectures that is geared towards processing image data, has demonstrated superior performance in medical image analysis applications like tumor detection, retinal disease classification, and lung disease identification. By using CNNs to detect kidney disease, we are potentially improving diagnostic accuracy and minimizing the necessity for invasive testing. In this work, we introduce a CNN-based model for detection and classification of kidney diseases from medical image data, such as computed tomography scans. The model proposed is trained to detect patterns in images of kidneys that predict normal kidney or tumor.

#     Text: {text}
#     """
#     paraphrased_text = gemini_response(prompt)
  
#     return paraphrased_text

def humanize_text(text,qualityIndex,max_limit,min_limit,model_type):
    if not text:
        return "Error"
    
    paraphrased_sentences = get_paraphrased_sentences(text)
    
    print("\ncombined_paraphrased_text\n")
    print(paraphrased_sentences)
    
    summary = get_summary(paraphrased_sentences,min_limit,max_limit,qualityIndex, model_type)

    generated_summary = clean_text(summary)
    
    print("\nGenerated Summary")
    print(generated_summary)
    
    generated_summary = introduce_human_errors(generated_summary)
    return generated_summary