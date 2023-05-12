Hi Laurits (cc Felix, Dennis, Mark),

I hope all is good with you and your newborn! 

I’m writing because I’ve been toying with my model for cultural loss and right now I might concentrate in capturing just the behavior of one trait in one node at a time. 

For that, I’ve created a simple conceptual model ([see notes in the github markdown](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Embers%20(quest%20for%20fire)%20model%2C%20trait%20decay%20and%20recovery.md "https://github.com/andreuandreu/cultural_loss/blob/master/notes/Embers%20(quest%20for%20fire)%20model%2C%20trait%20decay%20and%20recovery.md")).

In the model, the knowledge of a trait decays log-normal until it goes beyond a threshold, where I consider the knowledge to be lost.

I wanted the model to inspire the understanding of a particular case study.

The case is the apparent lack of use by paleolithic Nearthentals in southern-france that Denis is studying (see papers).

**Investigating variability in the frequency of fire use in the archaeological record of Late Pleistocene Europe**

[https://link.springer.com/article/10.1007/s12520-022-01526-1](https://link.springer.com/article/10.1007/s12520-022-01526-1 "https://link.springer.com/article/10.1007/s12520-022-01526-1")

**Reconstructing Late Pleistocene paleoclimate at the scale of human behavior: an example from the Neandertal occupation of La Ferrassie (France)**

[https://www.nature.com/articles/s41598-020-80777-1](https://www.nature.com/articles/s41598-020-80777-1 "https://www.nature.com/articles/s41598-020-80777-1")

The hypothesis on that paper (following Sandgathe et al. [2011a](https://link.springer.com/article/10.1007/s12520-022-01526-1#ref-CR122 "https://link.springer.com/article/10.1007/s12520-022-01526-1#ref-CR122"), [b](https://link.springer.com/article/10.1007/s12520-022-01526-1#ref-CR123 "https://link.springer.com/article/10.1007/s12520-022-01526-1#ref-CR123")) is that Neanderthals can not make fire at will.

Then, if there is no natural fire available in the environment because it is too cold & wet, at some point they could not procure natural fire sources.

Thus, they discontinued its use. 

The other hypothesis (Sorensen and Scherjon ([2019](https://link.springer.com/article/10.1007/s12520-022-01526-1#ref-CR130 "https://link.springer.com/article/10.1007/s12520-022-01526-1#ref-CR130"); Henry [2017](https://link.springer.com/article/10.1007/s12520-022-01526-1#ref-CR64 "https://link.springer.com/article/10.1007/s12520-022-01526-1#ref-CR64")) is that, during colder periods, the costs of maintaining fire (e.g., gathering fuel) exceeded its benefits. 

My model tries to capture that possible event (no noticeable presence of fire in caves with hominid presence) in the case of the decay of cultural memory and lack of access to naturally occurring fire. Once the knowledge is lost, it is not recovered. 

(see this article for reference for the log-normal decay [https://doi.org/10.1038/s41562-018-0474-5](https://doi.org/10.1038/s41562-018-0474-5 "https://doi.org/10.1038/s41562-018-0474-5") The universal decay of collective memory and attention, Canadia 2019)

Then, Laurits, I wanted to ask, would it be possible to have an estimate for the frequency of forest fires in that time period (116—29kaBP)?

I guess it is a difficult metric to have, as it would be a combination of how frequent dry lighting is, plus how dry and warm the land is. 

I do not know if there is any model available for current meteorological conditions, let alone, for paleolithic ones! 

But it might be correlated to the frequency of storms in your models, that’s why I’m asking. 

I know there are some models on the probability of lighting, plus there is the probability of rain, so maybe there is there shall be a way to probability of dry lighting. 

Still, since my model has 4 parameters to play with (_knowledge threshold, decay rate, frequency of occurrence_ and _variance of the frequency_), a paleoclimatic estimation on the frequency of forest fires plus it’s variance (if they can be obtained at all), would already constrain two of the parameters 😃

Then, choosing some reasonable upper-lower limits on _decay rate_ and _knowledge threshold_ and swipe through the parameter range should be relatively straightforward.

This would provide estimates for tipping-points on _decay_ and _knowledge_ that can be informative, assuming the model is useful. 

That’s my optimistic side, on the other hand, the range of storm-induced forest fires might be too wide (from once a year to once a millennia…). 

Which is one concern Dennis shares. He points that there might be sub-millennia warm periods with plenty of forest fires, in which the rediscovery of wildfire harvesting can happen. 

Using the model I describe to shed light, in this case, might be quite useless. 

The model I outline now assumes that fire can not easily be made a cultural practice again once the loss threshold is crossed.                    

I’m thinking of another model that might account for that.

That would be adding an _Invention rate_ that depends both on the natural occurrence of the conditions to pick up the fire again + an _acquisition/acceptance rate._ i.e. given the fire, how likely it is that the social group adopts it back to their cultural repertoire? Assuming it is no longer in their cultural memory.

However, that adds two/three more parameters, with probably no empirical constraint.

As interesting as it can be, I’m not eager to dive into innovation rates across history and cultures if it can be avoided.

The parameter should be a kind of _acquisition threshold_, beyond which the trait is acquired, the mean acceptance rate, and its variance. 

[See the last part of the notes for more details.](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Embers%20(quest%20for%20fire)%20model%2C%20trait%20decay%20and%20recovery.md "https://github.com/andreuandreu/cultural_loss/blob/master/notes/Embers%20(quest%20for%20fire)%20model%2C%20trait%20decay%20and%20recovery.md")

Anyway, this might end not being doable at this stage of data/climatic modeling, but it is worth a check 😃

Thanks in advance and best wishes,

Andreu