import datasets
import json

dl = datasets.DownloadManager()
configs_file = dl.download('https://huggingface.co/datasets/RealTimeData/bbc_alltime/raw/main/configs.txt')

with open(configs_file, encoding="utf-8") as f:
    _TIMES = f.read().splitlines()

_CITATION = """\
@misc{li2023estimating,
      title={Estimating Contamination via Perplexity: Quantifying Memorisation in Language Model Evaluation}, 
      author={Yucheng Li},
      year={2023},
      eprint={2309.10677},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
This dataset contains BBC News articles from 2017 to 2022. The articles are arraged by month. Access the specific month by using the format "YYYY-MM" as config. Such as load_dataset("RealTimeData/bbc_alltime", "2021-1").
"""

_HOMEPAGE = "https://github.com/liyucheng09/Contamination_Detector"

class Bbc_alltimes(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=time, version=datasets.Version("1.0.0"), description=f"BBC News articles published in the priod of {time}"
        )
        for time in _TIMES
    ]

    def _info(self):
        features = datasets.Features(
            {
                "title": datasets.Value("string"),
                "published_date": datasets.Value("string"),
                "authors": datasets.Value("string"),
                "description": datasets.Value("string"),
                "section": datasets.Value("string"),
                "content": datasets.Value("string"),
                "link": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.name == "all":
            times = _TIMES[:-1]
            files = dl_manager.download([f"articles/{time}.json" for time in _TIMES ])
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"files": files},
                )
            ]
        else:
            time = self.config.name
            _URL = f"articles/{time}.json"
            file = dl_manager.download(_URL)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"files": file},
                )
            ]

    def _generate_examples(self, files):
        """Yields examples."""
        if self.config.name == "all":
            assert isinstance(files, list)
            for file in files:
                time = file.strip('.json')
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)
                length = len(data['title'])
                for i in range(length):
                    yield f'{time}-{i}', {
                        "title": data['title'][i],
                        "published_date": data['published_date'][i],
                        "authors": data['authors'][i],
                        "description": data['description'][i],
                        "section": data['section'][i],
                        "content": data['content'][i],
                        "link": data['link'][i],
                    }
        else:
            assert isinstance(files, str)
            time = self.config.name
            with open(files, encoding="utf-8") as f:
                data = json.load(f)
            length = len(data['title'])
            for i in range(length):
                yield f'{time}-{i}', {
                    "title": data['title'][i],
                    "published_date": data['published_date'][i],
                    "authors": data['authors'][i],
                    "description": data['description'][i],
                    "section": data['section'][i],
                    "content": data['content'][i],
                    "link": data['link'][i],
                }