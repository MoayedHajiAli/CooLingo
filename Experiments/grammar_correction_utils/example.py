from grammar_correction import *

def main():
    text = "I eated humbarger tommorow"

    gec = GrammarCorrector()

    corrected_sentence = gec.correct_grammar(text)

    print(f'Original Sentence: {text}\nCorrected Sentence: {corrected_sentence}')

if __name__ == "__main__":
    main()