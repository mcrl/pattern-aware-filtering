
import fasttext

model = fasttext.load_model("pattern_aware_filtering/utils/lid.176.ftz")
def detect_en(content: str) -> bool:
    """
    Keep the text only if ftlangdetect identifies it as English with a score >= 0.65 # refinedweb criteria
    """
    text = content.replace("\n", " ")

    result = model.predict(text, k=1)
    return (result[0][0] == "__label__en") and (result[1][0] >= 0.65)