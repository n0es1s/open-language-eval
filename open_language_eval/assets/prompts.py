SYSTEM_PROMPT = """You are an annotator for the quality of machine translation. Your task is to identify
errors and assess the quality of the translation.

\n
Based on the source segment and machine translation surrounded with triple backticks provided by the user, identify
error types in the translation and classify them. The categories of errors are: accuracy
(addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar,
inconsistency, punctuation, register, spelling),
style (awkward expression, inappropriate for context, inconsistent use), other, or no-error.\n

When returning the error categories, use the following categories: accuracy, fluency, style, terminology, non-translation, other, or no-error.

"""

DETAILED_CATEGORIES = """
/nAddition: This error occurs when extra content not in the original text
leads to repetition, unnecessary details, or redundancy, distorting the message
and potentially confusing readers or diverging from the original intent.
/nMistranslation: This error involves inaccurate translation or interpretation,
often due to poor word choice, leading to a message that strays from
the original content's meaning and intent.
/nOmission: This error occurs when essential elements from the original text are
missing in the translation, resulting in incomplete meaning and loss of
critical information or nuances needed for full understanding.
/nUntranslated text: This error refers to parts of the source language that remain
in the translation without being converted, resulting in an incomplete or inaccurate
translation.
/nCharacter encoding: This error occurs when the translation contains incorrect character encoding.
/nGrammar: This error involves incorrect grammar, such as tense, verb form, pronouns,
agreement, articles, or gender, disrupting fluency and coherence and risking
misunderstandings or credibility loss.
/nInconsistency: It refers to variations in style or structure that undermine
the fluency and readability of the translated text.
/nPunctuation: This error stems from incorrect punctuation, prepositions, quotation
marks, or hyphenation, disrupting clarity and reading flow, and potentially
causing misunderstandings.
/nRegister: This error occurs when the translation fails to maintain the appropriate level of formality or casualness, leading to awkward or inappropriate language usage.
/nSpelling: This error occurs when the translation contains incorrect spelling.
/nSource issue: It refers to any problematic elements originating from the source
text (i.e., ambiguities, grammatical errors, of unclear phrasing) that hinder
accurate translation and lead to misunderstandings.
/nIncorrect word order: This error occurs when the translation fails to keep the
original structure, order, or phrasing, which can alter the meaning, clarity,
or emphasis, leading to awkward or confusing text.
/nTerminology: This error occurs when a term or word choice is contextually inappropriate
or inconsistent, leading to misaligned meaning or intent and potentially causing
confusion or lack of clarity, especially with technical or specialized terms.
/nNo-error: This category denotes a flawless translation, accurately conveying the
source text's meaning, tone, nuances, consistency, and style with clarity, cultural
appropriateness, and grammatical accuracy in the target language.
"""

USER_PROMPT = """{source_language} source:\n
```{source_segment}```\n
{target_language} translation:\n
```{target_segment}```\n
"""

SEVERITY_PROMPT_GEMMA = """
\nEach error is classified as one of three categories: critical, major, and minor.
Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what
the text is trying to say is still understandable. Minor errors are technically errors,
but do not disrupt the flow or hinder comprehension."""

SEVERITY_PROMPT_DR4 = """
\nEvaluate the severity of each error on a scale from 1 to 4 according to the given rubric.
\nScale 1: The error slightly changes in wording with has no impact on message clarity or intent.
\nScale 2: The error makes some alteration of wording, but the overall message and intent remain
mostly clear.
\nScale 3: The error has noticeable impact on comprehension and may slightly distort the intended
message.
\nScale 4: The error substantially distorts the message, making the translation unfaithful and
potentially misleading."""

SEVERITY_PROMPT_DQ4 = """
Evaluate the severity of each error on a scale from 1 to 4, where 1 starts on "minimal error
with no impact on clarity", goes to "minor alterations" and "noticeably impact comprehension",
up to 4, indicating "significant error substantially distort the message"."""

TRANSLATION_PROMPT = """
You are a professional translator. Translate the following text from {source_language} to {target_language}.

Requirements:
- Maintain the original meaning and tone
- Preserve any formatting (line breaks, punctuation)
- Keep proper nouns unchanged unless they have standard translations
- Provide only the translation without explanations

Text to translate:
{text}

Translation:"""
