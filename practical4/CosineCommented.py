__author__ = 'user'
# bits from http://stackoverflow.com/questions/15173225/how-to-calculate-cosine-similarity-given-2-sentence-strings-python
# load_docs, process_docs and compute_vector by MK
import math
from collections import Counter

vector_dict = {}                                       #Dict that will hold tf-idf matrix

dd1 = "On April 25 and 26, 1986, the worst nuclear accident in Chernobyl history unfolded in what is now northern Ukraine as a reactor at a nuclear power plant exploded and burned. Shrouded in secrecy, the incident was a watershed moment in both the Cold War and the history of nuclear power. Chernobyl More than 30 years on, scientists estimate the zone around the former plant will not be habitable for up to 20,000 years. The disaster took place near the city of Chernobyl in the former USSR, which invested heavily in nuclear power Chernobyl after World War II. Starting in 1977, Chernobyl Soviet scientists installed four RBMK nuclear reactors at the power plant, Chernobyl which is located just south of what is now Ukraine’s border with Belarus."
dd2 = "On April 25, 1986, routine maintenance was scheduled at V.I. Lenin Nuclear Power Station’s fourth reactor, and workers planned to use the downtime to test whether the reactor could still be cooled if the plant Chernobyl lost power. During the test, however, workers violated safety protocols Chernobyl and power surged inside the plant. Despite attempts to Chernobyl shut down the reactor entirely, another power surge caused a chain reaction of explosions inside. Finally, the nuclear core itself was exposed, Chernobyl spewing radioactive material into the atmosphere. Chernobyl Firefighters attempted to put out a series of blazes at the plant, and eventually helicopters dumped sand and other materials in an attempt to squelch the fires and contain the contamination. Despite the death of two people Chernobyl in the explosions, the hospitalization of workers and firefighters, and the danger from fallout and fire, no one in the surrounding Chernobyl areas—including the nearby city of Pripyat, which was built in the 1970s to house workers Chernobyl at the plant—was evacuated until about 36 hours after the disaster Chernobyl began."

dd3 = "The radioactive substance cesium-137 takes many years to break down with an estimated half-life of 30 years. It still exists in the earth in the areas affected by the Chernobyl accident, including large parts of Norway and Sweden. The substance is taken up from the soil by plants and fungi, which in turn are eaten by sheep, reindeer and other grazing animals. In the wake of the 1986 accident, Chernobyl cesium-137 spread over much of northern and central Scandinavia. The weather Chernobyl conditions were such that Norway and Sweden were two of the countries worst hit outside the Soviet Union. In Sweden, the areas around Uppsala, Gävle and Västerbotten were hardest hit, while in Norway the area between Trondheim and Bodø along with mountainous areas further south suffered, mainly because of rainfall."


#Just loads in all the documents
def load_docs():
 print("Loading docs...")
 doc1=('d1', 'LSI tutorials and fast tracks')
 doc2=('d2', 'books on semantic analysis')
 doc3=('d3', 'learning latent semantic indexing')
 doc4=('d4', 'advances in structures and advances in indexing')
 doc5=('d5', 'analysis of latent structures')
 return [doc1, doc2,doc3,doc4,doc5]

#Computes TF for words in each doc, DF for all features in all docs; finally whole Tf-IDF matrix
def process_docs(all_dcs):
 stop_words = [ 'of', 'and', 'on','in' ]
 all_words = []                                         #list to collect all unique words in each docs
 counts_dict = {}                                       #dict to collect doc data, word-counts and word-lists
 for doc in all_dcs:
    words = [x.lower() for x in doc[1].split() if x not in stop_words]
    words_counted = Counter(words)                      #counts words in a doc
    unique_words = list(words_counted.keys())           #list of the unique words in the doc
    counts_dict[doc[0]] = words_counted                 #make dict entry {'d1' : {'a': 1, 'b':6}}
    all_words = all_words + unique_words                #collect all unique words from each doc; bit hacky
 n = len(counts_dict)                                   #number of documents is no of entries in dict
 df_counts = Counter(all_words)                         #DF of all unique words from each doc, counted
 compute_vector_len(counts_dict, n, df_counts)          #computes TF-IDF for all words in all docs


#computes TF-IDF for all words in all docs
def compute_vector_len(doc_dict, no, df_counts):
  global vector_dict
  for doc_name in doc_dict:                              #for each doc
    doc_words = doc_dict[doc_name].keys()                #get all the unique words in the doc
    wd_tfidf_scores = {}
    for wd in list(set(doc_words)):                      #for each word in the doc
        wds_cts = doc_dict[doc_name]                     #get the word-counts-dict for the doc
        wd_tf_idf = wds_cts[wd] * math.log(no / df_counts[wd], 10)   #compute TF-IDF
        wd_tfidf_scores[wd] = round(wd_tf_idf, 4)        #store Tf-IDf scores with word
    vector_dict[doc_name] = wd_tfidf_scores              #store all Tf-IDf scores for words with doc_name


def get_cosine(text1, text2):
     vec1 = vector_dict[text1]
     vec2 = vector_dict[text2]
     intersection = set(vec1.keys()) & set(vec2.keys())
     #NB strictly, this is not really correct, needs vector of all features with zeros
     numerator = sum([vec1[x] * vec2[x] for x in intersection])
     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     if not denominator:
        return 0.0
     else:
        return round(float(numerator) / denominator, 3)

#RUN THE DEFINED FNS

all_docs = load_docs()
process_docs(all_docs)
vector_dict['q'] = {'semantic' : 1, 'latent' : 1, 'indexing' : 1}

for keys,values in vector_dict.items(): print(keys, values)

text1 = 'd3'
text2 = 'q'
cosine = get_cosine(text1, text2)
print('Cosine:', cosine)
