

import tamilinayavaani as ti

def check_tamil_spelling_and_grammar(text):
    """Checks Tamil spelling and grammar using tamilinayavaani."""
    try:
        # Spell check
        spell_check_results = ti.SpellChecker.REST_interface(text)
        print("Spell Checking Results:")
        print(spell_check_results)

        # Grammar check (if supported by the library, add functionality here)
        # Currently, tamilinayavaani might focus on spell checking. Grammar checking
        # could be implemented if grammar tools are available.

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    tamil_text = input("Enter Tamil text to check spelling and grammar: ")
    check_tamil_spelling_and_grammar(tamil_text)
