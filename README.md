# NephoNeural
Create neural-based datasets for NephoVis

## What is NephoNeural?

NephoNeural is a library with which you can easily generate datasets for [NephoVis](https://github.com/QLVL/NephoVis), a vector semantics research tool. The main idea is that you have a collection of corpus examples of a specific polysemous type/lemma (e.g. *bank*). You create vector representations ("embeddings") for all examples using a neural model, apply dimension reduction to all vectors, and plot the reduced coordinates using NephoVis. Then you can inspect visually whether there are any specific patterns of linguistic interest that can be discerned, and use the built-in tools to find out what context words contributed to the vector representations the most.

NephoNeural currently only support [transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) models, notably in the [HuggingFace transformers implementation](https://huggingface.co/docs/transformers/index). Supporting static word embeddings does not make sense in this context, since static embeddings of course do not change depending on the context (which goes against the very idea of this type of research).

NephoNeural handles the following aspects of the data creation process:
- Keeping a corpus of all input sentences per type;
- Finding out the correspondence between transformer word pieces and regular words;
- Creating the embeddings of all token examples from word pieces;
- Applying dimension reduction to all token vectors;
- Keeping a list of all context word pieces which have high attention values (from the perspective of the target type);
- Keeping a list of the lemmas of all relevant context word pieces;
- Creating the embeddings of all relevant context word pieces;
- Keeping a collection of all models (unique combinations of parameters);
- Computing the euclidean distances between models;
- Applying dimension reduction to all model similarity values;
- Applying medoid clustering to all model distances;
- Attaching variables to each token (e.g. intra-/extralinguistic variables you might want to attach to each token);
- Exporting all of the above information to a NephoVis-compatible format immediately

## Installing NephoNeural

anthevec is not available on any Python package manager (yet). To use it, simply copy the `nepho_nn` folder from this repository to your Python script's directory (preferably using `git clone`). From there, you can simply import the libraries like you would for any other package. More information on what libraries to import is given below.

Note that the [`anthevec`](https://github.com/AntheSevenants/anthevec) library, on which NephoNeural depends heavily, is added as a submodule in this repository. To fetch the appropriate version, simply run `git submodule foreach git pull origin main` in NephoNeural's directory.

## Using NephoNeural

### Creating a corpus and adding sentences

```python
from nepho_nn.corpus import Corpus
```

#### Creating corpus

NephoNeural collects all corpus example sentences of a specific type in a `Corpus` object.

| parameter | type    | description                                      | example |
| --------- | ------- | ------------------------------------------------ | -------| 
| `lemma` | str  | the type/lemma of interest around which all your corpus examples are centred | `"bank"` |

```python
bank_corpus = Corpus("bank")
```

#### Adding a sentence

After you have created your corpus, you can populate it with sentences using the `add_sentence` method.

| parameter | type    | description                                      | example |
| --------- | ------- | ------------------------------------------------ | -------| 
| `sentence` | list(str)  | the **tokenised** sentence as a list of words  | `[ "I", "withdrew", "money", "at", "the", "bank" ]` |
| `token_index` | int | the zero-based index of your focus token (= the instantiation of your type) | `5` |
| `token_id` | str | a unique identifier you give to this specific token example | `"bank/brown/A/579"`  |
| `file`=`""` (optional) | str | the filename from where this example was extracted; currently unused | `"brown.txt"` |

```python
bank_corpus.add_sentence([ "I", "withdrew", "money", "at", "the", "bank" ], 5, "bank/brown/A/579", "brown.txt")
```

#### Retrieving a sentence

You can retrieve a sentence from a `Corpus` object using `get_sentence`.

| parameter | type    | description                                      | example |
| --------- | ------- | ------------------------------------------------ | -------| 
| `sentence_id` | int | the index of the sentence you want to retrieve | `2` |

There is no real straightforward way to "know" which sentence belongs to which id. However, this method is used in the background, so you should generally never come into contact with it.

```python
bank_corpus.get_sentence(2)
```

### Creating a type

```python
from nepho_nn.type import Type
```

Once all your sentences for a specific type are added to that type's corpus, you can hand over the corpus to a `Type` object. This object will create the vectors, keep track of context words, do dimensionality reduction, generate medoids ... In short, it does everything for you.

There are quite a number of parameters needed to initialise a `Type`. The table below is a reference, but a more elaborate explanation is given below.

| parameter | type    | description                                      | example |
| --------- | ------- | ------------------------------------------------ | -------| 
| `lemma` | str | the type/lemma of interest around which all your corpus examples are centred | `"bank"` |
| `sentences` | list(dict) | the collection of sentences from the `Corpus` object | `bank_corpus.sentences` |
| `bert_model` | [transformers.model](https://huggingface.co/docs/transformers/main_classes/model) | a HuggingFace transformers model, initialised with hidden state output (and optionally attention output) | `RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=config)`|
| `tokenizer` | [transformers.tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer) | a HuggingFace transformers **fast** tokenizer | `RobertaTokenizerFast.from_pretrained(MODEL_NAME)`|
| `nlp` | [spacy.lang](https://spacy.io/api/language)| a spaCy tokenizer, used for word tokenisation | `spacy.load("nl_core_news_sm")` |
| `dimension_reduction_techniques` | list(nepho_nn.DimensionReductionTechnique) | a list of dimension reduction techniques which will be applied to the token vectors | `[ DimTsne("tsne30"), DimMds("mds") ]` |
| `medoid_clusters` | int | the number of medoids to be found (0 = disabled) | `3` |
| `layer_indices` | list(int) | a list of the indices for which models are generated; ranges from 1-12, embedding layer `0` cannot be used | `[ 9, 10, 11, 12 ]` |
| `attention_head_indices`=`[None]` (optional)|list(int)|a list of which heads should be used for attention weighting; ranges 0-11| `[ 0, 1, 2, 3, 4, 5 ]` |
| `collect_context_words`=`True` (optional)|bool|whether to collect the relevant context words for each token|`False`|

This is what an instantiation would look like in practice: 
```python
bank_type = Type(bank_corpus.lemma,
                 bank_corpus.sentences,
                 bert_model,
                 tokenizer,
                 nlp,
                 [ DimTsne("tsne30"), DimMds("mds") ],
                 3,
                 [ 9, 10, 11, 12 ],
                 attention_head_indices=[ 0, 1, 2, 3, 4, 5 ],
                 collect_context_words=True)
```
- We create a type for the *bank* lemma by using the `bank_corpus` object we made earlier. We can pass its `lemma` and `sentences` properties first.
- We supply the BERT model, its tokenizer and the spaCy tokenizer. To initialise these, refer to the [anthevec documentation](https://github.com/AntheSevenants/anthevec#prerequisites). Only the "Prerequisites" are relevant.
- We supply a list of dimension reduction techniques. Two techniques, `DimTsne` and `DimMds` (for tSNE and MDS respectively) are included. You can easily program your own dimension reduction techniques if needed (see below).
- We declare that we want to find cluster centres ("medoids").
- We define that we want to create models for layers 9, 10, 11 and 12.
- We want to look at attenion head indices 0, 1, 2, 3, 4 and 5.
- We will collect context words.

Every combination of layer and attention head will be treated as a separate model. This means that you will always end up with n x m models (with n = number of layers, m = number of attention heads). If attention head weighting is disabled, you end up with as many models as you have layers defined.

Attention head weighting is a procedure that uses the attention weights from a specific head to compose a vector for a word consisting of multiple word pieces. To make this more clear, let's look at the example from the schematic below:
![Schematic of attention weighting illustrated using the word 'banks'. There are two vectors: the vector 'bank' and the vector '-s'. There are weights for both vectors. Without attention weighting, both vectors receive a weight of 0.5. With attention weighting, 'bank' is given a higher weight, 0.7, and 's' 0.3.](https://user-images.githubusercontent.com/84721952/164240727-a65377d3-0989-4b8a-b415-834a224c3ea1.png)
We see that without attention weighting, a regular mean is used. This means that both *bank* and *#s* receive the same importance in the construction of the final vector. However, with attention weighting, the average attention weights from all word pieces in the input are taken into account. The attention values for word pieces not part of the word are discarded, and the remaining attention values are normalised to sum to one. The normalised values are then used as a means of creating a weighted average, which should influence the final vector composition. Attention weighting is done for each head you define in `attention_head_indices` separately.

#### Attaching variables to a type

You can attach variables to your tokens. Variables could be extra- or intralinguistic information which you think might influence the vector values of a token. Variables are separated from the corpus, which makes it possible to compile both data sources separately, or use a corpus without variables.

To attach variables to a type, use the `attach_variables` method:

| parameter | type    | description                                      | example |
| --------- | ------- | ------------------------------------------------ | -------| 
| variables | `list(dict)` | a list of dictionaries containing token information | see below for how to structure this dict | 

```python
with open("variables.json", "rt") as reader:
    variables = json.loads(reader.read())
    bank_type.attach_variables(variables)
````

This is an example `variables.json` file. You can see that the file is simply a list of objects. The only compulsory property for each object is `_id`. This property's value should correspond with the id of the token it belongs to (= the `token_id` property you defined for the token in `Corpus.add_sentence()`). Of course, all tokens should share  the same properties.
```json
[
  {
    "_id": "bank/brown/A/579",
    "sense": "financial",
    "genre": "A",
    "country": "UK"
  }
]
```
Note: it is possible to not have variables for a specific token. Tokens without associated variables will show up as "NA" in NephoVis. Of course, having missing data is never a good thing...

### Dimension reduction techniques

#### Using the built-in dimension reduction techniques

```python
from nepho_nn.dim_mds import DimMds
from nepho_nn.dim_tsne import DimTsne
```

By default, NephoNeural comes with two dimension reduction techniques built in: MDS and tSNE. You can initialise a dimension reduction technique by importing it, and then simply initialising it.

| parameter | type    | description                                      | example |
| --------- | ------- | ------------------------------------------------ | -------| 
| `name` | str | the name of the dimension reduction technique as it will show up in NephoVis; you can use this to differentiate between the same technique with different parameters | `"tsne30"` |
| `settings`=`{}` (optional) | dict |  a dict of parameters you want to set for the dimension reduction; to find out what parameters are supported, please look at the [source code](https://github.com/AntheSevenants/NephoNeural/tree/main/nepho_nn) | `{ "perplexity": 15 }` |

```python
dim_red_mds = DimMds("mds")
dim_red_tsne = DimTsne("tsne30", { "perplexity": 30 })
```

#### Creating a new dimension reduction technique

You can also define your own dimension reduction technique by extending the [`DimensionalityReductionTechnique` class](https://github.com/AntheSevenants/NephoNeural/blob/main/nepho_nn/dimension_reduction_technique.py). You only need to implement one method: `reduce`.

```python
from nepho_nn.dimension_reduction_technique import DimensionReductionTechnique

class MyDimensionReductionTechnique(DimensionReductionTechnique):
    def reduce(self, data):
        # Do something with the data
        
        return reduced_data
```

| parameter | type    | description                                      | example |
| --------- | ------- | ------------------------------------------------ | -------| 
| `data` | [numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html) (2D) | the vector matrix/distance matrix to reduce, as a two-dimensional numpy matrix  | / |

By default, `reduce` is used to reduce both token vector matrices and model distance matrices. If you want to apply a different technique for the model distance matrices, you can define another method, `reduce_model`. You can look at [the implementation of MDS](https://github.com/AntheSevenants/NephoNeural/blob/main/nepho_nn/dim_mds.py) as an example.

## Exporting types for use with NephoVis

You can quickly export the `Type` objects you created to a file format compatible with NephoVis. For this, we use `TypeExporter`.

```python
from nepho_nn.type_exporter import TypeExporter
```

| parameter | type    | description                                      | example |
| --------- | ------- | ------------------------------------------------ | -------| 
| `output_dir` | str | the path to the root of the NephoVis dataset | `"../NephoVis/tokenclouds_nn/data/"` |
| `types` | list(nepho_nn.Type) |  a list of all types which should be included in the dataset | `[ Type(...), Type(...) ]` |
| `skip_novar_tokens`=`True` (optional) | bool | whether to include tokens in the dataset for which no variables are found|`False`|

```python
type_exporter = TypeExporter("../NephoVis/tokenclouds_nn/data/",
                             [ bank_type, sound_type ],
                             skip_novar_tokens=False)
```

To export all types, use `export()`. It takes no arguments.

```python
type_exporter.export()
```

## Future work

- provide more examples
- add option to mask SOS/EOS attention