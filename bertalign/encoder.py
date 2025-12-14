import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps

class Encoder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def transform(self, sents, num_overlaps):
        overlaps = []
        for line in yield_overlaps(sents, num_overlaps):
            overlaps.append(line)

        sent_vecs = self.model.encode(overlaps)
        embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
        sent_vecs.resize(num_overlaps, len(sents), embedding_dim)

        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs)
        len_vecs.resize(num_overlaps, len(sents))

        return sent_vecs, len_vecs

# ISO 639-1 (plus a couple practical aliases) -> SONAR/NLLB (FLORES-200) code
ISO_TO_SONAR = {
    # A
    "sq": "als_Latn",   # Albanian (Tosk) :contentReference[oaicite:1]{index=1}
    "ar": "arb_Arab",   # Modern Standard Arabic :contentReference[oaicite:2]{index=2}
    "hy": "hye_Armn",   # Armenian :contentReference[oaicite:3]{index=3}

    # B
    "eu": "eus_Latn",   # Basque :contentReference[oaicite:4]{index=4}
    "bn": "ben_Beng",   # Bengali :contentReference[oaicite:5]{index=5}
    "br": "bre_Latn",   # Breton :contentReference[oaicite:6]{index=6}
    "bg": "bul_Cyrl",   # Bulgarian :contentReference[oaicite:7]{index=7}

    # C
    "hr": "hrv_Latn",   # Croatian :contentReference[oaicite:8]{index=8}
    "cs": "ces_Latn",   # Czech :contentReference[oaicite:9]{index=9}

    # D
    "da": "dan_Latn",   # Danish :contentReference[oaicite:10]{index=10}
    "nl": "nld_Latn",   # Dutch :contentReference[oaicite:11]{index=11}

    # E
    "en": "eng_Latn",   # English :contentReference[oaicite:12]{index=12}

    # F
    "fi": "fin_Latn",   # Finnish :contentReference[oaicite:13]{index=13}
    "fr": "fra_Latn",   # French :contentReference[oaicite:14]{index=14}

    # G
    "ka": "kat_Geor",   # Georgian :contentReference[oaicite:15]{index=15}
    "de": "deu_Latn",   # German :contentReference[oaicite:16]{index=16}
    "el": "ell_Grek",   # Greek :contentReference[oaicite:17]{index=17}

    # H
    "hi": "hin_Deva",   # Hindi :contentReference[oaicite:18]{index=18}
    "hu": "hun_Latn",   # Hungarian :contentReference[oaicite:19]{index=19}

    # I
    "id": "ind_Latn",   # Indonesian :contentReference[oaicite:20]{index=20}
    "ga": "gle_Latn",   # Irish :contentReference[oaicite:21]{index=21}
    "it": "ita_Latn",   # Italian :contentReference[oaicite:22]{index=22}

    # J
    "ja": "jpn_Jpan",   # Japanese :contentReference[oaicite:23]{index=23}

    # K (Kurmanji / Kurdish)
    "kmr": "kmr_Latn",  # Kurmanji (recommended key if you can) :contentReference[oaicite:24]{index=24}
    "ku":  "kmr_Latn",  # fallback if your lang detector outputs "ku" (ambiguous Kurdish)

    # L
    "la": "lat_Latn",   # Latin :contentReference[oaicite:25]{index=25}
    "lv": "lav_Latn",   # Latvian :contentReference[oaicite:26]{index=26}
    "lt": "lit_Latn",   # Lithuanian :contentReference[oaicite:27]{index=27}

    # M
    "zh": "zho_Hans",   # Mandarin Chinese (default = Simplified) :contentReference[oaicite:28]{index=28}
    # If you need Traditional Chinese, use: "zho_Hant"
    "mr": "mar_Deva",   # Marathi :contentReference[oaicite:29]{index=29}

    # P
    "fa": "pes_Arab",   # Persian :contentReference[oaicite:30]{index=30}
    "pl": "pol_Latn",   # Polish :contentReference[oaicite:31]{index=31}
    "pt": "por_Latn",   # Portuguese :contentReference[oaicite:32]{index=32}
    "pa": "pan_Guru",   # Punjabi (default = Eastern Punjabi, Gurmukhi) :contentReference[oaicite:33]{index=33}
    # If you need Western Punjabi (Shahmukhi), use: "pnb_Arab"

    # R
    "ro": "ron_Latn",   # Romanian :contentReference[oaicite:34]{index=34}
    "ru": "rus_Cyrl",   # Russian :contentReference[oaicite:35]{index=35}

    # S
    "es": "spa_Latn",   # Spanish :contentReference[oaicite:36]{index=36}
    "sw": "swh_Latn",   # Swahili :contentReference[oaicite:37]{index=37}
    "sv": "swe_Latn",   # Swedish :contentReference[oaicite:38]{index=38}

    # T
    "tr": "tur_Latn",   # Turkish :contentReference[oaicite:39]{index=39}

    # U
    "uk": "ukr_Cyrl",   # Ukrainian :contentReference[oaicite:40]{index=40}
    "ur": "urd_Arab",   # Urdu :contentReference[oaicite:41]{index=41}

    # W
    "cy": "cym_Latn",   # Welsh :contentReference[oaicite:42]{index=42}
}


class SonarTextEncoder:
    def __init__(self, model_name, device=None, batch_size=64, normalize=False):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.normalize = normalize

        self.encoder = M2M100Encoder.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.inference_mode()
    def _encode_batch_mean_pool(self, texts, lang):
        # SONAR は src_lang を指定（例: cym_Latn） :contentReference[oaicite:2]{index=2}
        self.tokenizer.src_lang = lang
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        seq_embs = self.encoder(**batch).last_hidden_state  # [B, T, H]
        mask = batch["attention_mask"].unsqueeze(-1)        # [B, T, 1]
        mean_emb = (seq_embs * mask).sum(1) / mask.sum(1).clamp(min=1)

        if self.normalize:
            mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)

        return mean_emb.detach().cpu().numpy()

    def transform(self, sents, num_overlaps, lang):
        overlaps = [line for line in yield_overlaps(sents, num_overlaps)]

        # ISO -> SONAR lang code
        sonar_lang = ISO_TO_SONAR.get(lang, None)
        if sonar_lang is None:
            raise ValueError(f"SONAR lang code not found for lang='{lang}'. Add it to ISO_TO_SONAR.")

        # batched encoding
        all_vecs = []
        for i in range(0, len(overlaps), self.batch_size):
            chunk = overlaps[i:i+self.batch_size]
            all_vecs.append(self._encode_batch_mean_pool(chunk, sonar_lang))
        sent_vecs = np.concatenate(all_vecs, axis=0)  # [num_overlaps*len(sents), H]

        embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
        sent_vecs = sent_vecs.reshape(num_overlaps, len(sents), embedding_dim)

        len_vecs = np.array([len(line.encode("utf-8")) for line in overlaps], dtype=np.int32)
        len_vecs = len_vecs.reshape(num_overlaps, len(sents))

        return sent_vecs, len_vecs