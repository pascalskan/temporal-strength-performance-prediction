from pathlib import Path


class ProjectPaths:
    """
    Filesystem layout for the project.

    Results are scoped by the dataset that produced them, so that a run over
    the deterministic fixture cannot be mistaken for, or overwrite, a run over
    the production dataset. Both previously resolved to results/<cohort>,
    because the cohort name ("raw", "equipped") is chosen independently of the
    dataset file, which meant `--test` wrote 905-row fixture output into the
    production path and stamped it as the raw cohort.

    Declaring the scope is mandatory: results_dir() raises until a caller has
    said which dataset is in play. Prefer use_dataset(), which resolves the
    input path and sets the scope together so the two cannot drift apart.
    """

    _dataset_scope = None
    _dataset_digest = None

    @staticmethod
    def project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def data_dir(cls) -> Path:
        return cls.project_root() / "data"

    @classmethod
    def set_dataset_scope(cls, label: str) -> None:
        """Declare which dataset subsequent results belong to."""
        if not label:
            raise ValueError("Dataset scope must be a non-empty label.")
        cls._dataset_scope = label

    @classmethod
    def dataset_scope(cls) -> str:
        return cls._dataset_scope

    @classmethod
    def set_dataset_digest(cls, digest: str) -> None:
        """Record the content digest of the input, for run metadata."""
        cls._dataset_digest = digest

    @classmethod
    def dataset_digest(cls):
        return cls._dataset_digest

    @classmethod
    def clear_dataset_scope(cls) -> None:
        """Reset the scope and digest. Intended for tests."""
        cls._dataset_scope = None
        cls._dataset_digest = None

    @classmethod
    def use_dataset(cls, filename: str) -> Path:
        """
        Resolve a dataset path and scope results to it in one step.

        Returns the path to the dataset file.
        """
        cls.set_dataset_scope(Path(filename).stem)
        return cls.dataset_path(filename)

    @classmethod
    def results_dir(cls) -> Path:
        if cls._dataset_scope is None:
            raise RuntimeError(
                "No dataset scope declared, so results have no provenance and "
                "would be written to a shared, ambiguous location. Call "
                "ProjectPaths.use_dataset(<filename>) before producing results."
            )
        path = cls.project_root() / "results" / cls._dataset_scope
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
