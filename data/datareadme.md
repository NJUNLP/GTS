## Data Format Descprition
The datasets are formatted in JSON and organized as one aspect term corresponding to multiple opinion terms:

```
[{
	"id": "id of sentence 1",
	"sentence": "context of sentence 1",
	"triples": [{
		"uid": "id of first aspect term of sentence 1",
		"target_tags": "the first aspect term with BIO scheme",
		"opinion_tags": "the corresponding multiple opinion terms of the aspect term with BIO scheme",
		"sentiment": "the corresponding sentiment polarity of the aspect term"
	},{
        the second aspect term of sentence 1
        ...
    }]
}, {
	sentence 2
    ...
}]
```

So you can fetch one aspect term, its corresponding multiple opinion terms, and the corresponding sentiment polarity to form multiple opinion triplets of this aspect term.