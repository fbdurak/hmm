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
# FUNCTIONS
#
#


#
#
# Outputs all possible 8-bit strings that could encode the given letters
# Input: Lowercase letter or lowercase alphabetic string
# Output: All possible strings of occluded bits (8-bit)
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


#
#
# Transforms letters in a name to states for training
# Input: original states, transformation method
# Output: Transformed states to be used for model training
# Kwarg: ngram_length - number of consecutive states on sliding window method
#
#
def transform_states(states,transforms=None,**kwargs):
    global BLANK_STATE
    global VALID_TRANSFORM_METHODS
    global N_CHARACTERS

    assert transforms is None or (set(transforms) <= set(VALID_TRANSFORM_METHODS))

    if len(states) < N_CHARACTERS:
        states += [BLANK_STATE] * (N_CHARACTERS - len(states))

    if transforms is None:
        return states

    if 'POSITION' in transforms:
        states = [s+str(i) for (s,i) in zip(states,range(len(states)))]

    if 'NGRAM' in transforms:
        assert kwargs.get('ngram_length',None) is not None
        ngram_length = kwargs.get('ngram_length')
        new_states = []
        # print "LEN",len(states)
        for i in range(len(states)-1):
            first = None
            second = None
            if ngram_length == 'ALL' or ngram_length > i:
                first = states[:i + 1]
            else:
                first = states[i + 1 - ngram_length:i + 1]
            first = ''.join(first)
            second = first + states[i + 1]
            if ngram_length != 'ALL' and len(second) > ngram_length:
                second = second[-ngram_length:]

            if i==0:
                new_states += [first]
            new_states += [second]
        states = new_states

    return states




#
#
# Converse transformed states to original states.  To be used to convert states to human-readable letters in a name.
# Input: original states, transformation method
# Output: Original states/letters
#
#
def inverse_transform_states(states,transforms=None):
    global BLANK_STATE
    global VALID_TRANSFORM_METHODS
    global N_CHARACTERS

    assert transforms is None or (set(transforms) <= set(VALID_TRANSFORM_METHODS))

    if transforms is None:
        return states


    if 'POSITION' in transforms:
        states = [s[0::2] for s in states]

    if 'NGRAM' in transforms:
        states = [s[-1] for s in states]

    return states




#
#
# Globals
#
#

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


TRANSFORM_METHOD = ['NGRAM']
NGRAM_LENGTH = 3

VALID_TRANSFORM_METHODS = ['POSITION','NGRAM']
names = [n.lower() for n in set(NAMES) if type(n)==str and len(n) <= N_CHARACTERS]
names = sorted(list(names))


#
#
# Small Test Cases
#
#
# print transform_states(['l','i','v','i','a'],transforms=['POSITION'])
# print inverse_transform_states(transform_states(['l','i','v','i','a'],transforms=['POSITION']),transforms=['POSITION'])
# print "\n"
# print transform_states(['l','i','v','i','a'],transforms=['NGRAM'],{'ngram_length':3})
# print inverse_transform_states(transform_states(['l','i','v','i','a'],transforms=['NGRAM'],{'ngram_length':3}),transforms=['NGRAM'])


#
#
# Cache total possible observations for each letter
#
#
LETTER_SCRAMBLINGS = dict()
for letter in string.ascii_lowercase + BLANK_STATE:
    LETTER_SCRAMBLINGS[letter] = letter_to_scramblings(letter)

transition_counts = dict()
outputs = []
states = list(string.ascii_lowercase + BLANK_STATE)






#
# Transitions between states and emission probabilities
#
for name in names:
    name = list(name)
    print "NAME",name

    name = transform_states(name,transforms=None)
    # transformed_name = transform_states(name,transforms=TRANSFORM_METHOD)
    transformed_name = transform_states(name,transforms=TRANSFORM_METHOD,**{'ngram_length':NGRAM_LENGTH})

    print "NAME AFTER",transformed_name
    for i in range(len(name)-1):
        first_state = transformed_name[i]
        second_state = transformed_name[i+1]
        first_letter = name[i]
        second_letter = name[i+1]


        if transition_counts.get((first_state, second_state), None) == None:
            transition_counts[(first_state, second_state)] = 0
        transition_counts[(first_state, second_state)] += 1

        if i == 0:
            outputs += [(first_state, o) for o in LETTER_SCRAMBLINGS[name[i]]]
            states += [first_state]
        outputs += [(second_state, o) for o in LETTER_SCRAMBLINGS[name[i + 1]]]
        states += [second_state]



states = list(set(states))
outputs = nltk.ConditionalFreqDist(outputs)
outputs = nltk.ConditionalProbDist(outputs, nltk.MLEProbDist)




#
# Symbols: output
#
symbols = list()
for letter in LETTER_SCRAMBLINGS:
    symbols += LETTER_SCRAMBLINGS[letter]
symbols = list(set(symbols))



transitions = []
for (k,v) in transition_counts:
    transitions += [(k,v)] * transition_counts[(k,v)]
transitions = nltk.ConditionalFreqDist(transitions)
transitions = nltk.ConditionalProbDist(transitions, nltk.MLEProbDist)





# Priors
priors = dict()
# first_letters_counter = Counter([transform_states(list(n),transforms=TRANSFORM_METHOD)[0] for n in names])
first_letters_counter = Counter([transform_states(list(n),transforms=TRANSFORM_METHOD,**{'ngram_length':NGRAM_LENGTH})[0] for n in names])
first_letters_total = sum([first_letters_counter[l] for l in first_letters_counter.keys()])
for letter in first_letters_counter.keys():
    priors[letter] = first_letters_counter[letter]/float(first_letters_total)
print priors
print states
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
    x = ['0' + d for d in x]
    y = list(y)


    print "TAGGING..."
    y_predict = tagger.tag(x)
    y_predict = [v for (k,v) in y_predict]
    y_predict = inverse_transform_states(y_predict,transforms=TRANSFORM_METHOD)

    print "Y PREDICT INVERSE",y_predict

    y_predict = ''.join(y_predict)
    print "PRED",y_predict,"TRUE",''.join(y)