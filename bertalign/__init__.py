"""
Bertalign initialization
"""

__author__ = "Jason (bfsujason@163.com)"
__version__ = "1.1.0"

#from bertalign.encoder import Encoder
from bertalign.encoder import SonarTextEncoder

# See other cross-lingual embedding models at
# https://www.sbert.net/docs/pretrained_models.html

model_name = "cointegrated/SONAR_200_text_encoder"
#model = Encoder(model_name)
model = SonarTextEncoder(model_name)

def transform(sents, num_overlaps, lang):
    return model.transform(sents, num_overlaps, lang=lang)

from bertalign.aligner import Bertalign
