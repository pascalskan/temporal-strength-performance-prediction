from pathlib import Path


class ProjectPaths:
    @staticmethod
    def project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def data_dir(cls) -> Path:
        return cls.project_root() / "data"

    @classmethod
    def results_dir(cls) -> Path:
        path = cls.project_root() / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def dataset_path(cls, filename: str = "openpowerlifting.csv") -> Path:
        return cls.data_dir() / filename

    @classmethod
    def retrospective_group_dir(cls, group_name: str) -> Path:
        path = cls.results_dir() / group_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def retrospective_evaluation_dir(cls, group_name: str) -> Path:
        path = cls.retrospective_group_dir(group_name) / "evaluation"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def retrospective_traditional_dir(cls, group_name: str) -> Path:
        path = cls.retrospective_group_dir(group_name) / "traditional"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def retrospective_importance_dir(cls, group_name: str) -> Path:
        path = (
            cls.retrospective_group_dir(group_name)
            / "ml_engineered"
            / "feature_importance"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def forward_group_dir(cls, group_name: str) -> Path:
        return cls.retrospective_group_dir(group_name)

    @classmethod
    def forward_results_dir(cls, group_name: str) -> Path:
        path = cls.forward_group_dir(group_name) / "forward"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def comparison_plots_dir(cls, group_name: str) -> Path:
        path = cls.results_dir() / group_name / "comparison_plots"
        path.mkdir(parents=True, exist_ok=True)
        return path
