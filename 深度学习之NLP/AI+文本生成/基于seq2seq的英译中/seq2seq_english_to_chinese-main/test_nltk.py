"""
离线下载：http://www.nltk.org/nltk_data/
"""
import nltk
# nltk.download('punkt')
from nltk import word_tokenize


sen = "Tom didn't know how to translate the word computer because the people he was talking to had never seen one."
res = word_tokenize(sen.lower())
print(res)