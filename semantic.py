import spacy
nlp = spacy.load('en_core_web_sm')

# Using en_core_web_sm I got a Warning when comparing words in the first three blocks of code:
# UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, 
# which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors.
# You can always add your own word vectors, or use one of the larger models instead if available.
# similarity = nlp(sentence).similarity(model_sentence)


word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Cat and monkey seem to be similar because they are both animals;
# Monkey and banana have a high similarity as well. So we can assume that the model already puts together that monkeys eat bananas 
# Another interesting fact is that cat does not have any significant similarity with banana. So, the model does not explicitly seem to recognise transitive relationships in its calculation
# A similar example could be car, carrot and rabbit. 

word1 = nlp("cat")
word2 = nlp("rabbit")
word3 = nlp("carrot")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Interestingly enough cat and rabbit seem to be more similar that cat and monkey - could be the size factor or the fact that both can be pets. 
# Carrots are slightly more linked to rabbits than bananas to monkeys.
# Finnaly cats seem to be more linked to carrots than to bananas. 

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
        
# Using en_core_web_sm would return lower similarities in the code below than when using 'en_core_web_md'. Model seems more limited
        
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)