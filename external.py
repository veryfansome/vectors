import nltk

nltk.download('words')

from nltk.corpus import words as nltk_words

nltk_word_set = set()
nltk_word_set.update(nltk_words.words())


if __name__ == '__main__':
    print(len(nltk_word_set), "unique words total")
    print(list(nltk_word_set)[:10])
