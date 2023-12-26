import nltk

def install_averaged_perceptron_tagger():
    nltk.download('averaged_perceptron_tagger')

def install():
    install_averaged_perceptron_tagger()

if __name__ == '__main__':
    install()