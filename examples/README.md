# torchMoji examples

## Initialization
[create_twitter_vocab.py](create_twitter_vocab.py)
Create a new vocabulary from a tsv file.

[tokenize_dataset.py](tokenize_dataset.py)
Tokenize a given dataset using the prebuilt vocabulary.

[vocab_extension.py](vocab_extension.py)
Extend the given vocabulary using dataset-specific words.

[dataset_split.py](dataset_split.py)
Split a given dataset into training, validation and testing.

## Use pretrained model/architecture
[score_texts_emojis.py](score_texts_emojis.py)
Use torchMoji to score texts for emoji distribution.

[text_emojize.py](text_emojize.py)
Use torchMoji to output emoji visualization from a single text input (mapped from `emoji_overview.png`)

```sh
python examples/text_emojize.py --text "I love mom's cooking\!"
# => I love mom's cooking! 😋 😍 💓 💛 ❤
```

[encode_texts.py](encode_texts.py)
Use torchMoji to encode the text into 2304-dimensional feature vectors for further modeling/analysis.

## Transfer learning
[finetune_youtube_last.py](finetune_youtube_last.py)
Finetune the model on the SS-Youtube dataset using the 'last' method.

[finetune_insults_chain-thaw.py](finetune_insults_chain-thaw.py)
Finetune the model on the Kaggle insults dataset (from blog post) using the 'chain-thaw' method.

[finetune_semeval_class-avg_f1.py](finetune_semeval_class-avg_f1.py)
Finetune the model on the SemeEval emotion dataset using the 'full' method and evaluate using the class average F1 metric.
