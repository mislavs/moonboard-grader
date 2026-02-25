"""
Registry for services keyed by hold setup and wall angle.
"""

from pathlib import Path
from typing import Optional, TypeVar

from fastapi import HTTPException, status

from .generator_service import GeneratorService
from .predictor_service import PredictorService
from .problem_service import ProblemService

ServiceKey = tuple[str, int]
ServiceType = TypeVar(
    "ServiceType",
    PredictorService,
    ProblemService,
    GeneratorService,
)


class ServiceRegistry:
    """Stores and resolves backend services for board setup/angle combos."""

    def __init__(self) -> None:
        self._predictors: dict[ServiceKey, PredictorService] = {}
        self._problem_services: dict[ServiceKey, ProblemService] = {}
        self._generators: dict[ServiceKey, GeneratorService] = {}
        self._analytics_paths: dict[ServiceKey, Path] = {}
        self._available_keys: set[ServiceKey] = set()
        self._default_key: Optional[ServiceKey] = None

    @staticmethod
    def _build_key(setup_id: str, angle: int) -> ServiceKey:
        return setup_id, angle

    def register_combo(self, setup_id: str, angle: int) -> None:
        """Register a valid setup/angle combo even without all services."""
        self._available_keys.add(self._build_key(setup_id, angle))

    def register_predictor(
        self, setup_id: str, angle: int, service: PredictorService
    ) -> None:
        key = self._build_key(setup_id, angle)
        self._available_keys.add(key)
        self._predictors[key] = service

    def register_problem_service(
        self, setup_id: str, angle: int, service: ProblemService
    ) -> None:
        key = self._build_key(setup_id, angle)
        self._available_keys.add(key)
        self._problem_services[key] = service

    def register_generator(
        self, setup_id: str, angle: int, service: GeneratorService
    ) -> None:
        key = self._build_key(setup_id, angle)
        self._available_keys.add(key)
        self._generators[key] = service

    def register_analytics_path(self, setup_id: str, angle: int, path: Path) -> None:
        key = self._build_key(setup_id, angle)
        self._available_keys.add(key)
        self._analytics_paths[key] = path

    def set_default(self, setup_id: str, angle: int) -> None:
        key = self._build_key(setup_id, angle)
        self._available_keys.add(key)
        self._default_key = key

    def _resolve_key(self, setup_id: Optional[str], angle: Optional[int]) -> ServiceKey:
        if setup_id is None and angle is None:
            if self._default_key is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service registry default configuration is not set",
                )
            return self._default_key

        if setup_id is None or angle is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both hold_setup and angle must be provided together",
            )

        key = self._build_key(setup_id, angle)
        if key not in self._available_keys:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Configuration not found for setup '{setup_id}' at angle {angle}",
            )
        return key

    def _get_required_service(
        self,
        services: dict[ServiceKey, ServiceType],
        service_name: str,
        setup_id: Optional[str],
        angle: Optional[int],
    ) -> ServiceType:
        key = self._resolve_key(setup_id, angle)
        service = services.get(key)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"{service_name} unavailable for setup '{key[0]}' "
                    f"at angle {key[1]}"
                ),
            )
        return service

    def get_predictor(
        self, setup_id: Optional[str] = None, angle: Optional[int] = None
    ) -> PredictorService:
        return self._get_required_service(
            self._predictors,
            "Prediction service",
            setup_id,
            angle,
        )

    def get_loaded_predictor(
        self, setup_id: Optional[str] = None, angle: Optional[int] = None
    ) -> PredictorService:
        predictor_service = self.get_predictor(setup_id, angle)
        if not predictor_service.is_loaded:
            key = self._resolve_key(setup_id, angle)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not loaded for setup '{key[0]}' at angle {key[1]}",
            )
        return predictor_service

    def get_problem_service(
        self, setup_id: Optional[str] = None, angle: Optional[int] = None
    ) -> ProblemService:
        return self._get_required_service(
            self._problem_services,
            "Problem service",
            setup_id,
            angle,
        )

    def get_generator(
        self, setup_id: Optional[str] = None, angle: Optional[int] = None
    ) -> GeneratorService:
        return self._get_required_service(
            self._generators,
            "Generation service",
            setup_id,
            angle,
        )

    def get_loaded_generator(
        self, setup_id: Optional[str] = None, angle: Optional[int] = None
    ) -> GeneratorService:
        generator_service = self.get_generator(setup_id, angle)
        if not generator_service.is_loaded:
            key = self._resolve_key(setup_id, angle)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Generator model not loaded for setup '{key[0]}' "
                    f"at angle {key[1]}"
                ),
            )
        return generator_service

    def get_analytics_path(
        self, setup_id: Optional[str] = None, angle: Optional[int] = None
    ) -> Optional[Path]:
        key = self._resolve_key(setup_id, angle)
        return self._analytics_paths.get(key)

