from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

example_sentence = "I'm the boy in your other phone.. Lighting up inside your drawer at home. All alone"
stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)

# filtered_sentence = []
# for w in words:
#     if w not in stop_words:
#         filtered_sentence.append(w)

filtered_sentence = [w for w in words if not w in stop_words]
# print(filtered_sentence)

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)
print(type(train_text))
