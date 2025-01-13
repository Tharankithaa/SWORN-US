# -*- coding: utf-8 -*-
# இந்த நிரல் GPLv3 உரிமத்தில் வெளியிடப்படுகிறது.
import argparse
from pprint import pprint
import sys
from tamilinayavaani import SpellChecker, SpellCheckerResult

parser = argparse.ArgumentParser()
parser.add_argument('file',help='filename for spellcheck')

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    sorkal = [0,0]
    for result in SpellChecker(args.file).run():
        sorkal[0]+=1
        if result.Flag:
            print(f"சரியான சொல்: {result.Userword}")
            continue
        sorkal[1]+=1
        print(f"தவறான சொல்: {result.Userword} : " + (f"{', '.join(result.Suggestions)}" if len(result.Suggestions) > 0 else "[பரிந்துரை இல்லை]"))
    print("#"*80)
    print(f"மொத்த சொற்கள் : {sorkal[0]} , தவறான சொற்கள் {sorkal[1]}")
