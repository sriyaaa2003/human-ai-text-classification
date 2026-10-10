"""Download the two public corpora used by the study into data/.

HC3 (Guo et al. 2023): human vs ChatGPT answers, five domains.   https://huggingface.co/datasets/Hello-SimpleAI/HC3
Liang et al. 2023: TOEFL, US student, college-admission and CS224N essays, with GPT-3 / GPT-4 rewrites.
                   https://github.com/Weixin-Liang/ChatGPT-Detector-Bias
"""
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
HC3 = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
LIANG = "https://raw.githubusercontent.com/Weixin-Liang/ChatGPT-Detector-Bias/main/Data_and_Results/"
ESSAYS = [
    "Human_Data/TOEFL_real_91",
    "Human_Data/TOEFL_gpt4polished_91",
    "Human_Data/HewlettStudentEssay_real_88",
    "Human_Data/HewlettStudentEssay_GPTsimplify_88",
    "Human_Data/CS224N_real_145",
    "Human_Data/CollegeEssay_real_70",
    "GPT_Data/CS224N_gpt3_145",
    "GPT_Data/CS224N_gpt3PromptEng_145",
    "GPT_Data/CollegeEssay_gpt3_31",
    "GPT_Data/CollegeEssay_gpt3PromptEng_31",
]


def fetch(url: str, dest: Path) -> None:
    if not dest.exists():
        print("downloading", dest.name)
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    fetch(HC3, DATA / "hc3_all.jsonl")
    for path in ESSAYS:
        fetch(LIANG + path + "/data.json", DATA / f"{Path(path).name}.json")
