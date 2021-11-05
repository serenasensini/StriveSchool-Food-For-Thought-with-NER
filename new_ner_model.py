import spacy
from spacy.util import minibatch, compounding
import random
from spacy.training import Example

nlp = spacy.load("en_core_web_lg")

with open("food.txt") as file:
    dataset = file.read()

doc = nlp(dataset)
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

#
# """
#
# STEP 1 - TRAIN DATA
#
# """
#
# # Prepare training data
#
words = ["ketchup", "pasta", "carrot", "pizza",
         "garlic", "tomato sauce", "basil", "carbonara",
         "eggs", "cheek fat", "pancakes", "parmigiana", "eggplant",
         "fettucine", "heavy cream", "polenta", "risotto", "espresso",
         "arrosticini", "spaghetti", "fiorentina steak", "pecorino",
         "maccherone", "nutella", "amaro", "pistachio", "coca-cola",
         "wine", "pastiera", "watermelon", "cappuccino", "ice cream",
         "soup", "lemon", "chocolate", "pineapple"]

train_data = []

with open("food.txt") as file:
    dataset = file.readlines()
    for sentence in dataset:
        print("######")
        print("sentence: ", sentence)
        print("######")
        sentence = sentence.lower()
        entities = []
        for word in words:
            word = word.lower()
            if word in sentence:
                start_index = sentence.index(word)
                end_index = len(word) + start_index
                print("word: ", word)
                print("----------------")
                print("start index:", start_index)
                print("end index:", end_index)
                pos = (start_index, end_index, "FOOD")
                entities.append(pos)
        element = (sentence.rstrip('\n'), {"entities": entities})

        train_data.append(element)
        print('----------------')
        print("element:", element)

#
# """
#
# STEP 2 - TRAINING MODEL
#
# """
#

ner = nlp.get_pipe('ner')

# Adding new label
ner.add_label("FOOD")

# Resume training
optimizer = nlp.resume_training()
move_names = list(ner.move_names)

# List of pipes you want to train
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

# List of pipes which should remain unaffected in training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Begin training by disabling other pipeline components
with nlp.disable_pipes(*other_pipes):
    for iteration in range(30):
        print("Iteration #" + str(iteration))

        random.shuffle(train_data)
        losses = {}
        for batch in spacy.util.minibatch(train_data, size=compounding(4.0, 32.0, 1.001)):
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update model
                nlp.update([example], losses=losses, drop=0.1)
        print("Losses", losses)

#
# """
#
# STEP 3 - TEST THE TRAINED MODEL
#
# """
#
test_text = "I ate an hamburger yesterday, it was so good!"
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent)
