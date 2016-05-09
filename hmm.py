import pandas as pd
import string
import itertools
from scipy.misc import comb
import nltk
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.probability import DictionaryProbDist


# Idea: split "zelda" into "z1,e2,l3,d4,a5,B"

female_first = pd.read_csv('data/census-names/census-dist-female-first.csv',header=None)
male_first = pd.read_csv('data/census-names/census-dist-male-first.csv',header=None)
names = male_first[0].tolist() + female_first[0].tolist()





# 
# Training
# 

START_STATE = 'START'
BLANK_STATE = 'B'
END_STATE = 'END'
N_BITS = 8
N_CHARACTERS = 9

N_SCRAMBLINGS = 0
for i in range(1,N_BITS+1):
    N_SCRAMBLINGS += comb(N_BITS,i)



names = [n.lower() for n in set(names) if type(n)==str and len(n) <= N_CHARACTERS]
names = sorted(list(names))



transition_counts = dict()
for (prev_char,next_char) in itertools.product(list(string.ascii_lowercase),repeat=2):
    transition_counts[(prev_char,next_char)] = 0
for character in list(string.ascii_lowercase):
    transition_counts[(START_STATE, character)] = 0
    transition_counts[(character, BLANK_STATE)] = 0
    transition_counts[(character, END_STATE)] = 0
    transition_counts[(BLANK_STATE,END_STATE)] = 0
transition_counts[(BLANK_STATE,BLANK_STATE)] = 0



for name in names:
    name = list(name)
    if len(name) < N_CHARACTERS:
        name += [BLANK_STATE]*(N_CHARACTERS-len(name))
    name = [START_STATE]+name+[END_STATE]
    print "NAME",name
    for i in range(len(name)-1):
        transition_counts[(name[i],name[i+1])]+= 1


transitions = []
for (prev_char,next_char) in transition_counts:
    if prev_char == START_STATE:
        continue
    # if k == END_STATE or v==END_STATE:
    #     print "BLAH!"
    #     exit()
    transitions += [(prev_char,next_char)] * transition_counts[(prev_char,next_char)]
transitions = nltk.ConditionalFreqDist(transitions)
transitions = nltk.ConditionalProbDist(transitions, nltk.MLEProbDist)

# emission_probabilities
# for character in list()

# END state probabilities affected by
# - Whether the end state is in the middle
# - WHether the end state is at the end
# symbols 
# states
# transitions
# outputs
# priors
# transform

outputs = []
symbols = list()
for letter in string.ascii_lowercase:
    binary_letter = '0' + format(ord(letter), 'b')
    outputs += [(letter,binary_letter)]
    for i in range(1, N_BITS+1):
        for blank_bits in itertools.combinations(range(N_BITS),i):
            binary_with_blanks = list(binary_letter)
            for b in blank_bits:
                binary_with_blanks[b] = 'x'
            binary_with_blanks = ''.join(binary_with_blanks)
            symbols += [binary_with_blanks]
            outputs += [(letter,binary_with_blanks)]

outputs = nltk.ConditionalFreqDist(outputs)
outputs = nltk.ConditionalProbDist(outputs, nltk.MLEProbDist)






# outputs



priors = dict()
for letter in string.ascii_lowercase:
    priors[letter] = transition_counts[(START_STATE,letter)]
total = sum([priors[l] for l in priors])
for letter in priors:
    priors[letter] /= float(total)
priors = DictionaryProbDist(priors)



symbols = list(set(symbols))
states = list(string.ascii_lowercase) + [BLANK_STATE]






# Create HMM
tagger = HiddenMarkovModelTagger(symbols,states,transitions,outputs,priors)
# symbols= obs


# Test
# x data
observations = None
with open('data/input-data/leakedBits.txt','r') as inf:
    observations = inf.read().split('\n')

# y data
labels = None
with open('data/input-data/names.txt','r') as inf:
    labels = inf.read().split('\n')


for (x,y) in zip(observations,labels):
    x = x.split() 
    x = ['x' + d for d in x] #list of 8-bit binaries
    y = list(y) #list of letters

    y_predict = tagger.tag(x)
    y_predict = ''.join([v for (k,v) in y_predict])
    print "PRED",y_predict,"TRUE",''.join(y)