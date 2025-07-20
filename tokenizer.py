import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import spacy
from transformers import BertTokenizer
from transformers import XLNetTokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

dataset = [
    (1,"Introduction to NLP"),
    (2,"Basics of PyTorch"),
    (1,"NLP Techniques for Text Classification"),
    (3,"Named Entity Recognition with PyTorch"),
    (3,"Sentiment Analysis using PyTorch"),
    (3,"Machine Translation with PyTorch"),
    (1," NLP Named Entity,Sentiment Analysis,Machine Translation "),
    (1," Machine Translation with NLP "),
    (1," Named Entity vs Sentiment Analysis  NLP ")]


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


def yield_tokens(datasetList):
    tokenizer = get_tokenizer("basic_english")
    for _,data in datasetList:
     yield tokenizer(data)

if __name__ == '__main__':
    print("NLKT tokens:",word_based_tokenizer_nltk())
    print("Spacy tokens:",word_based_tokenizer_spacy())
    print("Subword Bert Wordpiece tokens:",subword_wordpiece_based_tokenizer_bert())
    print("Subword XLNet Wordpiece tokens:",subword_wordpiece_based_tokenizer_xlnet())
    tokenlist = yield_tokens(dataset)
    vocab = build_vocab_from_iterator(yield_tokens(dataset),specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    tokenIndices = []
    for data in dataset:
        tokenWord = next(tokenlist)
        token_indices = [vocab[token] for token in tokenWord]
        print("token_indices:",token_indices)
        print("token_word:",tokenWord)


