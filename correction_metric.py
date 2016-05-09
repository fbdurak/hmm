
#finding the longest common prefix between two strings
def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer

# returns the list of names who has a fixed prefix that we find between two strings
# names is a list of names we get from census dataset.
def correction_metrix (prediction, true_value, names):
    answer=list()
    prefix= longestSubstringFinder(prediction, true_value)
    for name in names:
        s= longestSubstringFinder(prefix,name)
        if (s=prefix):
            answer.append(name)
    return answer
