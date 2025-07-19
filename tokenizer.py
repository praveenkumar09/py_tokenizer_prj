import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import spacy
from transformers import BertTokenizer
from transformers import XLNetTokenizer

def warn(*args,**kwargs):
    pass
import warnings
warnings.filterwarnings("ignore")
warnings.warn = warn

def word_based_tokenizer_nltk():
    text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
    tokens = word_tokenize(text)
    return tokens

def word_based_tokenizer_spacy():
    text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
    nlp  = spacy.load('en_core_web_sm')
    doc = nlp(text)
    token_list = [token.text for token in doc]
    return token_list

def subword_wordpiece_based_tokenizer_bert():
    text = "IBM taught me tokenization"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    return tokens

def subword_wordpiece_based_tokenizer_xlnet():
    text = "IBM taught me tokenization"
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    tokens = tokenizer.tokenize(text)
    return tokens

if __name__ == '__main__':
    print("NLKT tokens:",word_based_tokenizer_nltk())
    print("Spacy tokens:",word_based_tokenizer_spacy())
    print("Subword Bert Wordpiece tokens:",subword_wordpiece_based_tokenizer_bert())
    print("Subword XLNet Wordpiece tokens:",subword_wordpiece_based_tokenizer_xlnet())

