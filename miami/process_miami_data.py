#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# This file processes the Miami dataset into CS and monolingual test sets
import re
import os
import glob
import pylangacq
import librosa
import yaml
import json
import soundfile as sf
import string
from tqdm import tqdm
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer

DETOKENIZER = TreebankWordDetokenizer()

ONE_SECOND = 16000

# their language mapping to our language tags
LANG_MAP = {
    "s:spa": "spa",
    "s:eng": "eng",
    "s:eng&spa": "unknown",
    "s:eng&spag": "unknown",
    "s:spa&eng": "unknown",
    "s:spa+eng": "first spa, second eng",
    "s:eng+spa": "first eng, second spa",
    "s:eng&spa+eng": "unknown+extra",
    "s:ita": "italian",
    "s:fra": "french",
}

MAP_FOR_WORD_PREDS = {
    "first spa, second eng": "spa",
    "first eng, second spa": "spa",
    "eng": "eng",
    "spa": "spa",
    "italian": "italian",
    "french": "french",
}


##### simple string cleaning functions #####
def remove_punct(s: str) -> str:
    return s.translate(str.maketrans("", "", string.punctuation))


def verify_text(text: str):
    illegal_chars = ["[", "]", "(", ")", "/", "+", "&"]
    for char in illegal_chars:
        if char in text:
            raise Exception("had illegal char", char, text)


def remove_leading_spaces_punct(sent: list) -> str:
    new_sent = DETOKENIZER.detokenize(sent)
    # doesn't get second sentence
    if " ." in new_sent:
        new_sent = new_sent.replace(" .", ".")
    if " ?" in new_sent:
        new_sent = new_sent.replace(" ?", "?")
    if " ," in new_sent:
        new_sent = new_sent.replace(" ,", ",")
    return new_sent


def clean_underscores(sent: str) -> str:
    if "o_k" in sent:  # don't want to remove for o_k
        sent = sent.replace("o_k", "ok")

    new_sent = sent.replace("_", " ")
    return new_sent


def clean_up_common_markup_errors(sent: str) -> str:
    all_chars_to_replace = [
        "(.)",
        "(..)",
        "+//",
        "<",
        ">",
        "+/.",
        "+/?",
        "/",
        "...",
        "..",
        "++",
        "+/",
        "xxx",
        "+",
        '+"',
        "+,",
        "[",
        "]",
        "“",
    ]
    for char_phrase in all_chars_to_replace:
        sent = sent.replace(char_phrase, "")

    if '".' in sent:
        sent.replace('".', ".")
    if ":." in sent:
        sent.replace(":.", ".")

    if "@s:eng&spa" in sent:
        sent = sent.replace("@s:eng&spa", "")

    if re.search('".*"', sent) is None:
        sent = sent.replace('"', "")
    return sent


def clean_translation(text: str) -> str:
    text = clean_word_text(text)
    text = [word for word in re.sub(r"\([^)]*\)", "", text).split(" ") if word != ""]
    text = remove_leading_spaces_punct(text)
    return text


def clean_word_text(transcript):
    transcript = clean_up_common_markup_errors(transcript)
    detokenized_transcript = remove_leading_spaces_punct(transcript.split(" "))
    clean_sent = clean_underscores(detokenized_transcript)

    if len(transcript) and transcript[0] == ",":
        transcript = transcript[1:]

    return clean_sent


def make_transcript_manually(raw_utt: str) -> str:
    """
    Some of the utterances have disfluencies, which the pylangacq software excludes from the transcript
        Thus, we have to manually take the raw utterance transcription in CHAT form to keep them
    """
    filter_raw_utt = raw_utt.replace("<", "").replace(">", "")
    only_words = filter_raw_utt.split(" ")[:-1]  # last one is timing
    new_words = []
    for word in only_words:
        # words to replace
        if "@" in word:
            word = word[: word.find("@")]  # annotation for code-switching

        if not len(word):
            continue

        if word[0] == "[" or word[-1] == "]":
            continue  # don't need the markup

        if word[0] == "&":
            continue  # don't need partial starts

        word = clean_up_common_markup_errors(word)
        if len(word) == 0:
            continue

        if "+//." in word:
            word = word.replace("+//.", ".")
        if '".' in word:
            word = word.replace('".', ".")

        if "(" in word or ")" in word:
            # NOTE: this is whether we remove parantheticals
            word = re.sub(r"\([^)]*\)", "", word)

        new_words.append(word)

    detokenized_sent = remove_leading_spaces_punct(new_words)
    clean_sent = clean_underscores(detokenized_sent)
    return clean_sent


def gather_cs_statistics_and_words(utterance, raw_utt: str, transcript: str, file_lang: list, cur_lang: str):
    # for tagging each word, use a list of most common words
    common_spanish_words = []
    with open("common_words/spa.txt", "r") as fin:
        for line in fin:
            common_spanish_words.append(line.strip())

    common_english_words = []
    with open("common_words/eng.txt", "r") as fin:
        for line in fin:
            common_english_words.append(line.strip())

    # words like internet, etc.
    common_spanish_words = list(
        set(common_spanish_words)
        - set(common_english_words).intersection(set(common_spanish_words))
    )

    def get_lang_id(input_word): # parse the CHAT language id
        word = input_word.split("@")[1]
        word = (
            word.replace(">", "")
            .replace("[/]", "")
            .replace('"', "")
            .replace("”", "")
            .replace(".", "")
            .replace("]", "")
            .replace(",", "")
        )
        return LANG_MAP[word]

    eng = (
        utterance.tiers["%eng"] if "%eng" in utterance.tiers else None
    )  # English translation
    word_to_lang_map = [
        (word.split("@")[0], get_lang_id(word))
        for word in raw_utt.split(" ")
        if "@" in word
    ]
    is_cs = any(
        ["unknown" not in lang for (_, lang) in word_to_lang_map]
    )  # any not unknown is code-switched
    is_cs_any = len(word_to_lang_map)
    num_words = len(
        [
            item
            for item in raw_utt.split(" ")[:-1]
            if ("[" not in item and "." not in item)
        ]
    )
    cs_percent = 0 if not is_cs else len(word_to_lang_map) / num_words

    # refine the process of the main language, really only used for statistical purposes
    # this just flips the main language if the CS percent is greater than 0.5
    if (eng is None and cur_lang == "spa" and cs_percent > 0.5 and len(transcript.split(" ")) >= 3):
        cur_lang = "eng"
    if (eng is not None and cur_lang == "eng" and cs_percent > 0.5 and len(transcript.split(" ")) >= 3):
        cur_lang = "spa"

    # lets try to get word level tags for CS data. We have to do this manually parsing the sentence
    clean_transcript = remove_punct(transcript)
    cs_words = [
        remove_punct(clean_word_text(word))
        for (word, lang) in word_to_lang_map
        if "unknown" not in lang
    ]
    cs_words_lang = [
        lang for (word, lang) in word_to_lang_map if "unknown" not in lang
    ]
    for idx, cs_word in enumerate(cs_words):
        if cs_word not in clean_transcript:
            # try to clean up the word to see if it's in the clean transcript
            if cs_word.replace("(", "").replace(")", "") in clean_transcript:
                cs_words[idx] = cs_word.replace("(", "").replace(")", "")
            elif cs_word.split("(")[0] in clean_transcript:
                cs_words[idx] = cs_word.split("(")[0]

    tagged_words = ""
    main_lang, embedded_lang = file_lang[0], file_lang[-1]
    if "[- spa]" in raw_utt or "[-spa]" in raw_utt:
        main_lang, embedded_lang = "spa", "eng"
    if "[- eng]" in raw_utt or "[-eng]" in raw_utt:
        main_lang, embedded_lang = "eng", "spa"

    for word in clean_transcript.split(" "):
        if word in cs_words: # first see if they were annotated
            index = cs_words.index(word)
            annote_lang = MAP_FOR_WORD_PREDS[cs_words_lang[index]]
            tagged_words += f"{word}={embedded_lang} "
        else: # try to rely on the backup common words if they're not annotated
            if word in common_spanish_words:
                tagged_words += f"{word}=spa "
            elif word in common_english_words:
                tagged_words += f"{word}=eng "
            else:
                tagged_words += f"{word}={main_lang} "

    tagged_words = tagged_words.strip()
    return tagged_words, eng, cs_percent, is_cs, is_cs_any
        


def write_out(final_path, all_segments, all_transcripts, all_translations):
    with open(os.path.join(final_path, f"miami.yaml"), "w") as fout:
        fout.write(yaml.dump(all_segments, allow_unicode=True))

    with open(os.path.join(final_path, f"miami.transcript"), "w") as fout:
        for line in all_transcripts:
            fout.write(line)
            fout.write("\n")

    with open(os.path.join(final_path, f"miami.translation"), "w") as fout:
        for line in all_translations:
            fout.write(line)
            fout.write("\n")

    with open(os.path.join(final_path, f"miami.jsonl"), "w") as fout:
        for segment in all_segments:
            fout.write(json.dumps(segment))
            fout.write("\n")


def prepare_miami_data():
    all_segments = []
    all_transcripts = []
    all_translations = []

    final_path = "output/miami/all"
    if not os.path.isdir(final_path):
        os.makedirs(os.path.join(final_path, "clips"))

    chat_file_location = "data/miami/beta"  # beta has the most up to date
    for chat_file_path in tqdm(
        glob.glob(os.path.join(chat_file_location, "*.cha")), leave=True
    ):
        clip_name = chat_file_path.split("/")[-1].replace(".cha", "")
        cur_reader = pylangacq.read_chat(chat_file_path)
        all_words = cur_reader.words(by_utterances=True)
        assert len(cur_reader._files) == 1
        file_lang = cur_reader._files[0].header["Languages"]

        # get wav data
        wav_path = chat_file_path.replace("beta", "audio").replace("cha", "wav")
        wav_data, sampling_rate = librosa.load(
            wav_path, sr=ONE_SECOND
        )  # already at 16khz/16bit/mono
        assert sampling_rate == ONE_SECOND

        for idx, utterance in enumerate(cur_reader.utterances()):
            word_utterance = all_words[idx]
            transcript = " ".join(word_utterance)
            if not len(transcript):
                continue
            transcript = clean_word_text(transcript)
            raw_utt = utterance.tiers[utterance.participant]

            # the main language can be overriden if marked that way
            if "[- eng]" in raw_utt or "[-eng]" in raw_utt:
                cur_lang = "eng"
            elif "[- spa]" in raw_utt or "[-spa]" in raw_utt:
                cur_lang = "spa"
            else:
                cur_lang = file_lang[0]

            ## Check if we really want to keep cleaning this utterance ##
            if "www" in raw_utt:
                continue  # means untranscribed text, skip
            if word_utterance == ["."]:
                continue  # we don't want empty lines

            if "[" in raw_utt:  # some markup to deal with
                # see https://talkbank.org/manuals/CHAT.pdf for details
                markings = re.findall("\[.*?\]", raw_utt)
                for mark in markings:
                    if mark in [
                        "[!]",
                        "[?]",
                        "[!!]",
                        "[*]",
                        "[/-]",
                        "[//]",
                        '["]',
                    ] or mark in ["[- spa]", "[-spa]", "[-eng]", "[- eng]"]:
                        """
                        Markup definitions that we can skip/remove for ST purposes:
                            [!] means stressing
                            [!!] means constrastive stressing
                            [?] means uncertainty in transcription, but best guess
                            [=! ...] is some kind of para-linguistic communication, laugh, yell, etc.
                            [# ...] indicates duration of previous <> tag
                            [*] means the word is incorrect semantically/grammatically, typically followed by the [* correct_word]
                            [/-] is for false starts but still spoken
                            [//] for abandended and retracing speech

                        """
                        continue
                    elif "[=!" in mark or "[= !" in mark or "[*" in mark:  # see above
                        continue
                    elif mark in ["[/]", "[//]", "[///]"]:
                        # indicates trailing or correction while speaking, pylangacq gets rid of them, do it manually
                        if raw_utt is None:
                            continue
                        transcript = make_transcript_manually(raw_utt)
                        break
                    else:
                        raise Exception(f"Encountered new mark {mark}")

            time_marks = utterance.time_marks
            if time_marks is None:
                continue  # don't know why there are no time marks, but skip.
                # Happens appx 3 times outside of maria18.cha where there are ~20 instances

            # get the audio clip and validate it
            start_time, end_time = time_marks
            start_time_s, end_time_s = start_time / 1000, end_time / 1000
            duration_s = end_time_s - start_time_s
            wav_clip = wav_data[
                int(start_time_s * ONE_SECOND) : int(end_time_s * ONE_SECOND)
            ]
            if int(end_time_s * ONE_SECOND) < wav_data.shape[0]:
                # sometimes audio may go beyond the file length, which we allow
                error_str = f"Wav Clip:{wav_clip.shape[0]} vs duration:{duration_s * ONE_SECOND}"
                assert (duration_s * ONE_SECOND - wav_clip.shape[0]) < 1, error_str

            cur_clip_name = clip_name + "_p" + str(idx)
            clip_path = os.path.join("clips", cur_clip_name + ".wav")
            sf.write(os.path.join(final_path, clip_path), wav_clip, ONE_SECOND)

            # for LID and statistics, gather the lang id for each word
            tagged_words, eng, cs_percent, is_cs, is_cs_any = gather_cs_statistics_and_words(utterance, raw_utt, transcript, file_lang, cur_lang)
            speakers = utterance.participant # just in case it's needed someday for speaker ID

            all_segments.append(
                {
                    "wav": clip_path,
                    "offset": start_time_s,
                    "duration": duration_s,
                    "cs_percent": cs_percent,
                    "speaker_id": speakers,
                    "code_switched": is_cs,
                    "main_lang": cur_lang,
                    "code_switched_any": is_cs_any,
                    "tagged_words": tagged_words,
                }
            )
            translation = clean_translation(eng) if eng is not None else ""
            assert transcript is not None
            
            # validate the sentences
            verify_text(transcript)
            verify_text(translation)
            all_transcripts.append(transcript)
            all_translations.append(translation)
            assert len(all_transcripts) == len(all_segments) == len(all_translations)
        assert len(all_transcripts) == len(all_segments) == len(all_translations)
    assert len(all_transcripts) == len(all_segments) == len(all_translations)
    write_out(final_path, all_segments, all_transcripts, all_translations)


if __name__ == "__main__":
    prepare_miami_data()
