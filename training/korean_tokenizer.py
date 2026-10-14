from kiwipiepy import Kiwi

kiwi = Kiwi()

def tokenize_korean(text):
    return [token.form for token in kiwi.tokenize(text)]