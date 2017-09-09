import csv
# valid_list is the list of valid graphemes containing a totoal of 27 graphemes
# lower case letters a to z and a null character represented by a underscore.
valid_list = [chr(letter) for letter in range(ord('a'), ord('z') + 1)] + ['_']

def get_stats(train_file, valid_pfile="cmu-phonemes.txt",
              valid_graphemes=valid_list):
    '''
    get_stats takes three arguments:
    train_file, a string with the name of the CSV file to read in the
    alignment data from,
    valid_pfile, a string with the name of the CSV file to read in the valid
    phoneme data from, set to "cmu_phonemes.txt" by default,
    and valid_graphemes, a set of strings, containing the full inventory of
    valid graphemes, set to valid_list by default.
    get_stats then returns a tuple of numbers containing the number of invalid
    rows in train_file (an integer), the average number of phonemes per word,
    (a float), the average number of graphemes per word(a float) and the
    proportion of grapheme sequences containing one or more underscores
    (a float).
    '''

    # use csvreader to read trainfile to get a list of data after opening it
    phoneme2grapheme = open(train_file)
    data = csv.reader(phoneme2grapheme)
    next(data)   # get rid of the header in the train_file
    valid_phonemes = open(valid_pfile).read()  # get a string of valid phonemes

    # count the number of invalid rows
    # where the number of elements does not match between two sequences
    # or there is/are grapheme(s)/phoneme(s) not contained in the dictionary
    mylist = []
    count_invalid = 0     # count of valid rows initialised to 0
    for list_p2g in data:
        validality = True  # validality indicates if the row is valid

        # check if lengthes of graphemes and phonemes are equal
        if len(list_p2g[0].split()) != len(list_p2g[1].split()):
            validality = False

        # check if phoneme is in the corresponding dictionary of phonemes
        for phonemes in list_p2g[0].split():
            if phonemes not in valid_phonemes:
                validality = False

        # check if grapheme is in the corresponding dictionary of graphemes
        for graphemes in list_p2g[1].split():
            for each_grapheme in graphemes:
                if each_grapheme not in valid_graphemes:
                    validality = False

        # append the row to mylist if it is a valid row
        # otherwise, the number of invalid rows incremented by one
        if validality is True:
            mylist.append(list_p2g)
        else:
            count_invalid += 1

    # append valid phonemes and graphemes respectively to two empty lists
    list_phoneme = []
    list_grapheme = []
    for valid_list_p2g in mylist:
        list_phoneme.append(valid_list_p2g[0])  # contains all valid phonemes
        list_grapheme.append(valid_list_p2g[1])  # contains all valid graphemes

    # count average number of phonemes per word
    # return None if the denominator, length of the word is zero
    if len(list_phoneme) != 0:
        count_phoneme = 0
        # iterate over the list of phonemes to count the number of phonemes
        for phonemes in list_phoneme:
            for every_phoneme in phonemes.split():
                count_phoneme += 1
        average_phoneme = float(count_phoneme/len(list_phoneme))
    else:
        average_phoneme = None

    # calculate average number of grapheme per word
    # calculate proportion of graphemes sequences containing one of more
    # underscore, return None if the denominator, length of the word is zero
    if len(list_grapheme) != 0:
        count_grapheme = 0
        count_underscore = 0

        # iterate over the list of graphemes to count the number of graphemes
        # and the number of underscores
        for graphemes in list_grapheme:
            for every_grapheme in graphemes:
                # get rid of NULL grapheme, the underscore and whitespace
                if not every_grapheme == ' ' and not every_grapheme == '_':
                    count_grapheme += 1

                # count the number of underscores
                if every_grapheme == '_':
                    count_underscore += 1
        average_grapheme = float(count_grapheme/len(list_grapheme))
        proportion_underscore = float(count_underscore/len(list_grapheme))
    else:
        average_grapheme = None
        proportion_underscore = None

    phoneme2grapheme.close()

    # return the tuple
    return (count_invalid, average_phoneme, average_grapheme,
            proportion_underscore)







from collections import defaultdict
def train_ngrams(train_file):
    '''
    train_ngrams takes the single argument train_file which is a string
    identifying the name of alignment file to read the alignment data in from)
    and returns 2-tuple (bigram, trigram).
    '''
    # read the file with csv reader turning the string into rows of lists
    # and next function gets rid of the header from content.
    content = csv.reader(open(train_file))
    next(content)
    
    # freq_bigram and freq_trigram take dictionaries as their values.
    freq_bigram = defaultdict(dict)
    freq_trigram = defaultdict(dict)
    
    # a default dictionary of bigram frequency is created first.
    # the default dictionary, freq_bigram takes a phoneme as the key and
    # a dictionary as the value. Inside the dictionary, the matching grapheme
    # is the key and frequency that it appears is the value.
    for mylist in content:
        for n in range(len(mylist[0].split())):
            phoneme = mylist[0].split()[n]
            grapheme = mylist[1].split()[n]
            if grapheme in freq_bigram[phoneme].keys():
                freq_bigram[phoneme][grapheme] += 1
            else:
                freq_bigram[phoneme][grapheme] = 1
            
            # a default dictionary of trigram frequency is created then
            # the default dictionary, trigram_grapheme takes a tuple of a
            # phoneme and a grapheme that is preceding to the matching grapheme
            # as the key and a dictionary as the value.
            # Inside the dictionary, the matching grapheme is the key and
            # frequency that it appears is the value.
            if n > 0:
                trigram_grapheme = mylist[1].split()[n-1]
            else:
                trigram_grapheme = '^'
            if grapheme in freq_trigram[(phoneme, trigram_grapheme)].keys():
                freq_trigram[(phoneme, trigram_grapheme)][grapheme] += 1
            else:
                freq_trigram[(phoneme, trigram_grapheme)][grapheme] = 1

    return (freq_bigram, freq_trigram)



def normalise(ngrams):
    '''
    normalise takes the single argument ngrams which is a default dictionary
    of the form returned by train_ngrams for each of bigrams and trigrams in
    the return tuple, and converts each of the frequency to probabilities in
    place and returns None.
    '''
    # To get total count, increment frequencies of graphemes in the dictionary,
    # which is the value of the default dictionary, ngrams.
    for mydict in ngrams.values():
        total = 0
        for freq in mydict.values():
            total += freq

        # using total count divided by frequency of each grapheme to calculate
        # the probability of each grapheme corresponding to a given phoneme.
        for grapheme in mydict.keys():
            probability = mydict[grapheme]/total
            mydict[grapheme] = probability

    return None




def speech2text(phonemes, bigrams, trigrams, alpha=1, topn=10):
    '''
    speech2text takes 5 arguments:
    phonemes, a sequence of phonemes, in the form of a string,
    bigrams, a default dictionary of dictionaries of floats containing the
    bigram alignment probabilities,
    trigrams, a default dicitonary of dictionaries of floats, containing the
    trigram alignment probabilities,
    alpha, an optional argument used in calculation, set to 1.0 by default,
    and topn, an optional argument to set the size of the beam, set to 10 by
    default.
    speech2text then returns a list of 2-tuples, each containing a grapheme
    sequence in the form a string, and the probability of that seuence in the
    form of a float, in descending order of probability, sub-sorted
    orthographically in the case of ties.
    '''

    # initialise the beam to the single grapheme with the probability of 1.0
    # beam is the list containing tuple(s)
    # the list in the tuple(s) contains grapheme(s)
    beam = [(['^'], 1.0)]

    # iterate over the phonemes provided in the argument
    for phoneme in phonemes.split():
        # every time python iterates over one phoneme, mylist is initialised
        # as an empty list, functions as a temporary list stores the tuple(s)
        # containing lisst of grapheme(s) and related probability of the
        # grapheme(s) after calculation. Eventually mylist will be processed
        # and appended to beam.
        mylist = []

        # iterate over beam to find the list of grapheme(s)
        for tuple_grapheme_prob in beam:
            grapheme_list = tuple_grapheme_prob[0]

            # access the graphemes that are the keys of a dictionary by
            # iterating over value of the default dictionary bigrams
            # that corresponds to each phoneme which is the key.
            # in order to get the values of the dictionary corresponding to
            # each grapheme which is the key of the dictionary.
            for grapheme in bigrams[phoneme].keys():
                bi_prob = bigrams[phoneme][grapheme]

                # see whether the grapheme corresponding to the phoneme has
                # probability related to the preceding grapheme.
                # if it is not in the trigrams dictionary, the probability
                # will be 0
                if grapheme in trigrams[(phoneme, grapheme_list[-1])].keys():
                    tri_prob = trigrams[(phoneme, grapheme_list[-1])][grapheme]
                else:
                    tri_prob = 0

                # calculate the probability using given for formula.
                probability = tuple_grapheme_prob[-1] * (alpha * bi_prob +
                                                         (1-alpha) * tri_prob)

                # append the tuple containing negative value of probability and
                # the concatenation of lists of existing grapheme(s) in the
                # beam and new grapheme.
                # the probability is put in front of the list of graphemes and
                # set to negative in order to be sorted in order.
                mylist.append((-probability, grapheme_list+[grapheme]))

        # finallist is the sorted list containing topn, reticted number of
        # graphemes candidates.
        finallist = sorted(mylist)[:topn]

        # temp_list, a temporary list is used in order to change the order of
        # probability and final_grapheme_list and also the get rid of the minus
        # sign in front of the value of probability.
        temp_list = []
        for (probability, final_grapheme_list) in finallist:
            temp_list.append((final_grapheme_list, -probability))

        # assign beam to temp_list
        beam = temp_list
    return beam
