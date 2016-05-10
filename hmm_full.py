import pandas as pd
import string
import itertools
from scipy.misc import comb
import nltk
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.probability import DictionaryProbDist
from collections import Counter


#
#
# Given a letter or string, outputs the total possible scramblings for the last letter
#
#
def letter_to_scramblings(letter_or_string):
    global BLANK_STATE

    letter = letter_or_string
    if len(letter_or_string) > 1:
        letter = letter[-1]

    binary_letter = '0' + format(ord(letter), 'b')

    if letter_or_string[-1] == BLANK_STATE or letter_or_string==BLANK_STATE:
        binary_letter = '00000000'


    scramblings = [binary_letter]

    for i in range(1, N_BITS + 1):
        for blank_bits in itertools.combinations(range(N_BITS), i):
            binary_with_blanks = list(binary_letter)
            for b in blank_bits:
                binary_with_blanks[b] = 'x'
            binary_with_blanks = ''.join(binary_with_blanks)
            scramblings += [binary_with_blanks]

    return scramblings




female_first = pd.read_csv('data/census-names/census-dist-female-first.csv',header=None)
male_first = pd.read_csv('data/census-names/census-dist-male-first.csv',header=None)
NAMES = male_first[0].tolist() + female_first[0].tolist()

PREV_WINDOW_SIZE = 1
PREV_WINDOW_ALL = 'ALL'
START_STATE = 'START'
BLANK_STATE = 'B'
END_STATE = 'END'
N_BITS = 8
N_CHARACTERS = 9
N_SCRAMBLINGS = 2**N_BITS-1
# N_SCRAMBLINGS = 0
# for i in range(1,N_BITS+1):
#     N_SCRAMBLINGS += comb(N_BITS,i)
names = [n.lower() for n in set(NAMES) if type(n)==str and len(n) <= N_CHARACTERS]
names = sorted(list(names))

# Cache total possible observations for each letter
LETTER_SCRAMBLINGS = dict()
for letter in string.ascii_lowercase + BLANK_STATE:
    LETTER_SCRAMBLINGS[letter] = letter_to_scramblings(letter)




# for (first,second) in itertools.product(list(string.ascii_lowercase),repeat=2):
#     for i in range(N_CHARACTERS):
#         transition_counts[(first+str(i),second+str(i+1))] = 0
#
# for letter in string.ascii_lowercase:
#     for number in range(N_CHARACTERS):
#         transition_counts[(letter+str(number),BLANK_STATE)] = 0
#
#     transition_counts[(letter + str(range(N_CHARACTERS)[-1]), END_STATE)] = 0
#
# for character in list(string.ascii_lowercase):
#     transition_counts[(START_STATE, character+'0')] = 0
#     transition_counts[(BLANK_STATE,END_STATE)] = 0
# transition_counts[(BLANK_STATE,BLANK_STATE)] = 0

transition_counts = dict()
outputs = []
states = list(string.ascii_lowercase + BLANK_STATE)


for name in names:
    name = list(name)
    print "NAME",name
    if len(name) < N_CHARACTERS:
        name += [BLANK_STATE]*(N_CHARACTERS-len(name))

    assert len(name) == N_CHARACTERS
    for i in range(len(name)-1):
        first = None
        second = None
        if PREV_WINDOW_SIZE == 'ALL' or PREV_WINDOW_SIZE > i:
            first = name[:i+1]
        else:
            first = name[i+1-PREV_WINDOW_SIZE:i+1]


        first = ''.join(first)
        # second = name[i+1]
        second = first+name[i+1]
        if PREV_WINDOW_SIZE != 'ALL' and len(second) > PREV_WINDOW_SIZE:
            second = second[-PREV_WINDOW_SIZE:]

        print "FIRST","SECOND",first,second


        if transition_counts.get((first,second),None) == None:
            transition_counts[(first, second)] = 0
        transition_counts[(first,second)]+= 1
        output_scramblings = LETTER_SCRAMBLINGS[second[-1]]
        outputs += [(second,o) for o in output_scramblings]
        states += [second]

states = list(set(states))



outputs = nltk.ConditionalFreqDist(outputs)
outputs = nltk.ConditionalProbDist(outputs, nltk.MLEProbDist)




# END state probabilities affected by
# - Whether the end state is in the middle
# - WHether the end state is at the end
# symbols
# states
# transitions
# outputs
# priors
# transform


symbols = list()
for letter in LETTER_SCRAMBLINGS:
    symbols += LETTER_SCRAMBLINGS[letter]
symbols = list(set(symbols))



transitions = []
for (k,v) in transition_counts:
    print "K,V",k,v
    transitions += [(k,v)] * transition_counts[(k,v)]
transitions = nltk.ConditionalFreqDist(transitions)
transitions = nltk.ConditionalProbDist(transitions, nltk.MLEProbDist)





# Priors
priors = dict()
first_letters_counter = Counter([n[0] for n in names])
first_letters_total = sum([first_letters_counter[l] for l in first_letters_counter.keys()])
for letter in first_letters_counter.keys():
    priors[letter] = first_letters_counter[letter]/float(first_letters_total)
print priors
print states
print len(states),transitions,outputs,sum(priors.values())
priors = DictionaryProbDist(priors)



print "TRAINING"
tagger = HiddenMarkovModelTagger(symbols,states,transitions,outputs,priors)







observations = None
with open('data/input-data/leakedBits.txt','r') as inf:
    observations = inf.read().split('\n')


labels = None
with open('data/input-data/names.txt','r') as inf:
    labels = inf.read().split('\n')


assert len(observations) == len(labels)

for (x,y) in zip(observations,labels):
    x = x.split()
    x = ['x' + d for d in x]
    y = list(y)


    print "TAGGING..."
    y_predict = tagger.tag(x)

    y_predict = ''.join([v[-1] for (k,v) in y_predict])
    # y_predict = y_predict[0::2]
    print "PRED",y_predict,"TRUE",''.join(y)