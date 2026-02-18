import re
import spacy


NLP_LANGS = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "it": "it_core_news_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm"
}


class Segmenter:
    def __init__(self, src_lang: str, tgt_lang: str):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        src_nlp = spacy.load(NLP_LANGS[src_lang], exclude=["parser"])
        tgt_nlp = spacy.load(NLP_LANGS[tgt_lang], exclude=["parser"])
        src_nlp.enable_pipe("senter")
        tgt_nlp.enable_pipe("senter")

        self.nlp = {
            src_lang: src_nlp,
            tgt_lang: tgt_nlp,
        }

    def split_words(self, text: str, lang: str) -> list[str]:
        """ """
        # -- Break sentences into words
        words = [t.text for t in self.nlp[lang](text)]
        return words

    def split_sents(self, text: str, lang: str) -> list[str]:
        """ """
        doc = self.nlp[lang](text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        return sents
    
    def split_sents_punct(self, text: str) -> list[str]:
        """
        Sentence splitter based on terminal punctuation, with guardrails for:
        - NBSP / weird whitespace normalization (for splitting only)
        - spaced ellipses ". . ." and double dots ".."
        - decimal/version dots between digits (3.14, 1.0.2)
        - common abbreviations / initials that end in '.' (Dr., U.S., e.g.)
        - post-pass merge for lonely quote/paren closers/openers (», «, ", ', ), ])
        """
        if not text:
            return []

        # -------------------------
        # Normalize (splitting only)
        # -------------------------
        t = (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\u00A0", " ")  # NBSP
            .strip()
        )

        # Normalize spaced ellipsis: ". . ." -> "..."
        t = re.sub(r"\.\s*\.\s*\.", "...", t)
        # Normalize double dot -> ellipsis (common OCR-ish artifact)
        t = re.sub(r"(?<!\.)\.\.(?!\.)", "...", t)

        # -------------------------
        # Protect things that look like "a period but not a sentence end"
        # -------------------------
        DOT = "\uE000"  # private-use char unlikely to appear in text

        # Protect decimals / version-like dots between digits: 3.14, 1.0.2
        t = re.sub(r"(?<=\d)\.(?=\d)", DOT, t)

        # Protect common abbreviations (both EN/FR-ish); keep this list small+high precision
        # NOTE: This is intentionally conservative; you can expand as needed.
        abbr = [
            r"Mr\.", r"Mrs\.", r"Ms\.", r"Dr\.", r"Prof\.", r"Sr\.", r"Jr\.",
            r"St\.", r"No\.", r"Inc\.", r"Ltd\.", r"Co\.", r"vs\.", r"etc\.",
            r"e\.g\.", r"i\.e\.", r"U\.S\.", r"U\.K\.", r"E\.U\.",
            r"M\.", r"Mme\.", r"Mlle\.", r"p\.\s*ex\.", r"c\.-à-d\.", r"n°\."
        ]
        # Replace the final dot in each abbreviation with DOT
        for a in abbr:
            t = re.sub(a, lambda m: m.group(0)[:-1] + DOT, t)

        # Protect initials like "J. R. R. Tolkien" (replace dots after single letters)
        # (This is a heuristic; still conservative.)
        t = re.sub(r"\b([A-Za-zÀ-ÖØ-öø-ÿ])\.(?=\s*[A-Za-zÀ-ÖØ-öø-ÿ]\.)", r"\1" + DOT, t)

        # -------------------------
        # Regex split by terminal punctuation, allowing closers after it
        # -------------------------
        TERMINAL = r"(?:\.\.\.|…|[.!?])"
        CLOSERS = r"""["'”»)\]\}]+"""  # things that can trail a sentence

        sent_re = re.compile(
            rf".*?{TERMINAL}(?:\s*{CLOSERS})?(?=\s+|$)",
            re.DOTALL
        )

        parts: list[str] = []
        last_end = 0
        for m in sent_re.finditer(t):
            parts.append(m.group(0).strip())
            last_end = m.end()

        tail = t[last_end:].strip()
        if tail:
            parts.append(tail)

        # Restore protected dots
        parts = [p.replace(DOT, ".") for p in parts]
        parts = [p for p in parts if p]

        # -------------------------
        # Post-pass merge: lonely openers/closers
        # -------------------------
        # If the model puts « or » (or similar) alone, merge into neighbor.
        OPENERS_ONLY = re.compile(r'^[\s"\'“«(\[\{—–-]+$')
        CLOSERS_ONLY = re.compile(r'^[\s"\'”»)\]\}]+$')

        merged: list[str] = []
        i = 0
        while i < len(parts):
            s = parts[i].strip()

            # If this chunk is only closers, attach to previous if possible
            if CLOSERS_ONLY.match(s):
                if merged:
                    merged[-1] = (merged[-1].rstrip() + " " + s).strip()
                else:
                    merged.append(s)
                i += 1
                continue

            # If this chunk is only openers, attach to next if possible
            if OPENERS_ONLY.match(s):
                if i + 1 < len(parts):
                    parts[i + 1] = (s + " " + parts[i + 1].lstrip()).strip()
                else:
                    merged.append(s)
                i += 1
                continue

            merged.append(s)
            i += 1

        return merged

    def split_pars(self, text: str) -> list[str]:
        """ """
        # -- Normalize windows newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # -- Split on blank lines (one or more)
        pars = re.split(r"\n\s*\n+", text)

        return [p.strip() for p in pars if p.strip()]

    def split_par_sents(self, text: str) -> list[list[str]]:
        """ """
        pars = self.split_pars(text)

        par_sents = []

        for par in pars:
            sents = self.split_sents_punct(par)
            par_sents.append(sents)

        return par_sents
    
    def split_par_sent_words(self, text: str, lang: str) -> list[list[list[str]]]:
        """ """
        par_sents = self.split_par_sents(text)

        par_sent_words = []

        for par in par_sents:
            # -- Split each sentence into words
            sent_words = []
            for sent in par:
                words = self.split_words(sent, lang)
                sent_words.append(words)
            # -- Store all split sentences in paragraph
            par_sent_words.append(sent_words)

        return par_sent_words

    def split_pages(self, text: str, max_chars: int=2500) -> list[str]:
        """ """
        # -------------------------
        # Normalize whitespace / split into pars[sents]
        # -------------------------
        text = self.normalize_text(text)
        par_sents = self.split_par_sents(text)

        # -------------------------
        # Build each page until max_chars
        # -------------------------
        pages: list[str] = []
        cur_sents: list[str] = []
        cur_len = 0

        for par in par_sents:
            # -- Keep paragraphs on the same page
            page_break = False

            for sent in par:
                sent_len = len(sent) + 1    # space/newline

                # -- At > max_chars, build page and append
                if cur_len + sent_len > max_chars:
                    page_break = True
                
                # -- Add sentence to page
                cur_sents.append(sent)
                cur_len += sent_len

            # -- Create page and move to next one
            if page_break:
                pages.append(" ".join(cur_sents).strip())
                cur_sents = []
                cur_len = 0
            # -- Preserve paragraph break inside page
            else:
                cur_sents.append("\n\n")
                cur_len += 2

        last_page = " ".join(cur_sents).strip()
        if last_page:
            pages.append(last_page)

        return pages

    def normalize_text(self, text: str):
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split into paragraphs on blank lines
        paras = re.split(r"\n\s*\n+", text.strip())

        # De-wrap each paragraph: replace remaining newlines with spaces
        paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paras if p.strip()]

        # Re-join paragraphs with a single newline (already spaced by ParagraphGrid)
        return "\n\n".join(paras)