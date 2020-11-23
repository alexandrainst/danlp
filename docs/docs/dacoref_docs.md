## Documentation about the coreference resource: Dacoref

 To get an overview of different datasets, please go to the general [dataset docs](datasets.md). This is extra documentation for the coreference resources named Dacoref. In the general dataset docs, there is also a small snippet to show how to load this resource with the DaNLP package. The resource can also be downloaded directly using the link below: 

[Download dacoref](http://danlp-downloads.alexandra.dk/datasets/dacoref.zip) 

This documentation provides details about how the resource has been constructed.

The work is conducted by Maria Jung Barrett.

#### LICENCE
This resource is copyrighted material, licensed under the GNU Public License version 2.

#### Dacoref

This Danish coreference annotation contains parts of the Copenhagen Dependency Treebank (Kromann and Lynge, 2004). Please cite this work when using the coreference resource. For the Universal Dependencies conversion included in the file, please cite Johannsen et al. (2015).

Size:
64.076 tokens
3.403 sentences
341 documents

It was originally annotated as part of the Copenhagen Dependency Treebank (CDT) project but never finished. The incomplete annotation can be downloaded from the [project github](https://github.com/mbkromann/copenhagen-dependency-treebank).

The [CDT documentation](https://github.com/mbkromann/copenhagen-dependency-treebank/blob/master/manual/cdt-manual.pdf) contains description of the coreference classes as well as inter-annotator agreement and confusion matrices.

For this resource, we used the annotation files from the annotator "Lotte" along with the UD syntax which is an automatic conversion og the CDT syntax annotation by Johansen et al. (2015). We provide the sentence ID from the UD resource as well as the document ID from CDT. The document ID has been prepended with a two letter domain code compatible with the domain codes of the Ontonotes corpus. This is a manually mapping of the sources listed in the CDT. Only nw (newswire), mz (magazine), and bn (broadcast news) were present:
* 299 nw documents
* 41 mz documents
* 1 bn

For the CDT, only the core node of each span was annotated and one annotator manually propagated the label to the entire span. A few systematic errors were corrected in this process, the most important being that plural pronouns "we" and "they" can be coreferent with company names if they refer to the employee group of this company. 

For this resource we have merged the following labels to form uniquely numbered clusters: coref, coref-evol, coref-iden, coref-iden.sb, coref-var, and ref.
Coref-res and coref-res.prg are also included as clusters but not merged with any other label, nor each other.

Some notes about the annotation, but see also the CDT documentation:
If conjunctions of entities are only referred to as a group, they are marked as one span. (e.g. if "Lise, Lone og Birthe" are only referred to as a group, e.g. by the plural pronoun "de"), "Line, Lone og Birthe" is marked as one span.
The spans are generally as long as possible. Example: Det sidste gik ud over politikerne, da de i sin tid præsenterede [det første forslag til den milliard-dyre vandmiljøplan].

Singletons are not annotated. 
The annotation does not label attributative noun phrases that are connected through copula verbs such as to be. Name-initual appositive constructions are part of the same mention as the name.
Generic pronouns (mainly "man" and "du") are not clustered unless they are part of a cluster, e.g. with a reflexive or possesive pronoun.

Furthermore, the resource has been augmented with Qcodes from Wiktionary. This was a semi-automatic process conducted in the Spring of 2020 with the Wikidata entries available at that time.
First, all tokens (not just named entities) were used to search using the Wikidata API. 
Given the entire list of matches with description for each token, one annotator decided which QID match was correct for each instance. It was decided in each case whether, e.g., "Østre landsret" refers to building, governmental administrative unit in Denmark or the legal process happening there. 
This was checked by another annotator who also manually added the Qcode to the correct span in the text. Both were native speakers of Danish. Furthermore, this process also included adding a generic QID for words that matched in the categories below but did not exist as a specific Wikidata entry. This can e.g. be used to decide which properties an entity may have or whether a name refers to a feminine or masculine entity.

In total, 7173 tokens were annotated with a Qcode. 2193 unique Qcodes were used.

The file can be opened by a conll reader that accepts an arbitrary number of fields, e.g. conllu

```python
import conllu
conlist = conllu.parse(open('CDT_coref.conll').read(), fields=["id", "form", "lemma", "upos", 'xpos', 'feats', 'head', 'deprel','deps', 'misc', 'coref_id', 'coref_rel', 'doc_id', 'qid'])
```



#### 🎓 References

Johannsen, A., Alonso, H. M., & Plank, B. (2015). Universal dependencies for danish. In International Workshop on Treebanks and Linguistic Theories (TLT14) (p. 157).

M.T. Kromann and S.K. Lynge. Danish Dependency Treebank v. 1.0. Department of Computational Linguistics, Copenhagen Business School., 2004. https://github.com/mbkromann/copenhagen-dependency-treebank

Weischedel, R., Palmer, M., Marcus, M., Hovy, E., Pradhan, S., Ramshaw, L., ... & El-Bachouti, M. (2013). Ontonotes release 5.0 ldc2013t19. Linguistic Data Consortium, Philadelphia, PA, 23. https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf

#### Generic QIDs

Family name Q101352
Unisex nickname Q49614
Female given name Q11879590
Male given name Q12308941
Unisex name Q3409032
Artist name Q483501
Magazine Q41298
Hotel Q27686
Work of art Q838948
governmental administrative unit in Denmark	Q21268738
Municipal Police Q1758690
Road Q34442
Cohousing Q1107167
Postal address Q319608
Museum Q33506
Security (tradeable financial asset) Q169489
Geographic location Q2221906
Radio program Q1555508
Tv program Q15416
Product / goods Q2424752
Department within organisation Q2366457
Organization Q43229
Sports venue Q1076486
Dish Q746549 (only one instance)
Event Q1656682
Fleet Q189524
University Q3918
Disease Q12136
Coast Q93352
Ship Q11446
Award Q618779
Automobile model Q3231690
Project (also Inquiry)  Q170584
Hospital Q16917
Amusement ride Q1144661
Sports team Q12973014
Building Q41176
Bill (proposed law) Q686822
Restaurant Q11707
People / ethnic group Q2472587
Educational institution Q2385804
Shop Q213441
Publication Q732577
legislation Q49371
Night club Q622425
Newspaper Q11032
Prison Q40357
Army Q37726
