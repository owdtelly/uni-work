from urllib.request import urlopen
from nltk.tokenize import word_tokenize

target_url0 = "http://www.gutenberg.org/files/135/135-0.txt"
book_raw = urlopen(target_url0).read().decode("utf-8-sig")

print(book_raw[1:250])

print("hello")
