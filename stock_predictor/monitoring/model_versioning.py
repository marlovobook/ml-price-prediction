"""
Model versioning system with rollback capabilities.
Provides comprehensive model lifecycle management and version control.
"""

import os
import json
import shutil
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from ..utils.logging_config import get_logger
from ..utils.exceptions import StockPredictorError, ModelTrainingError
from ..interfaces import IMLModel, ModelConfiguration


@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str
    model_type: str
    pattern_length: int
    created_at: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_set: List[str] = field(default_factory=list)
    file_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = False
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelVersionManager:
    """
    Model versioning system with rollback capabilities.
    Manages model lifecycle, versioning, and deployment.
    """
    
    def __init__(self, base_path: str = "models", max_versions: int = 10):
        """
        Initialize model version manager.
        
        Args:
            base_path: Base directory for storing models
            max_versions: Maximum number of versions to keep per model type
        """
        self.base_path = Path(base_path)
        self.max_versions = max_versions
        self.logger = get_logger("ModelVersionManager")
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Version registry
        self._version_registry: Dict[str, List[ModelVersion]] = {}
        self._active_versions: Dict[str, str] = {}  # model_key -> version_id
        
        # Load existing versions
        self._load_version_registry()
    
    def save_model(
        self,
        model: IMLModel,
        model_config: ModelConfiguration,
        performance_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a model with versioning.
        
        Args:
            model: The trained model to save
            model_config: Model configuration
            performance_metrics: Optional performance metrics
            metadata: Optional additional metadata
            
        Returns:
            Version ID of the saved model
        """
        try:
            # Generate version ID
            version_id = self._generate_version_id(model_config)
            model_key = self._get_model_key(model_config.model_type, model_config.pattern_length)
            
            # Create version directory
            version_dir = self.base_path / model_key / version_id
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model file
            model_file = version_dir / "model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Save configuration
            config_file = version_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(asdict(model_config), f, indent=2, default=str)
            
            # Create version metadata
            model_version = ModelVersion(
                version_id=version_id,
                model_type=model_config.model_type,
                pattern_length=model_config.pattern_length,
                created_at=datetime.now(),
                performance_metrics=performance_metrics or {},
                hyperparameters=model_config.hyperparameters,
                feature_set=model_config.feature_set,
                file_path=str(model_file),
                metadata=metadata or {}
            )
            
            # Save version metadata
            version_file = version_dir / "version.json"
            with open(version_file, 'w') as f:
                json.dump(model_version.to_dict(), f, indent=2)
            
            # Update registry
            if model_key not in self._version_registry:
                self._version_registry[model_key] = []
            
            self._version_registry[model_key].append(model_version)
            
            # Clean up old versions if needed
            self._cleanup_old_versions(model_key)
            
            # Save registry
            self._save_version_registry()
            
            self.logger.info(f"Saved model version {version_id} for {model_key}")
            return version_id
            
        except Exception as e:
            raise ModelTrainingError(
                f"Failed to save model version: {str(e)}",
                error_code="MODEL_SAVE_ERROR",
                details={"model_type": model_config.model_type, "pattern_length": model_config.pattern_length}
            )
    
    def load_model(self, model_type: str, pattern_length: int, version_id: Optional[str] = None) -> tuple[IMLModel, ModelVersion]:
        """
        Load a model by type, pattern length, and optional version.
        
        Args:
            model_type: Type of model to load
            pattern_length: Pattern length for the model
            version_id: Specific version ID (None for active version)
            
        Returns:
            Tuple of (model, version_metadata)
        """
        try:
            model_key = self._get_model_key(model_type, pattern_length)
            
            if version_id is None:
                # Load active version
                if model_key not in self._active_versions:
                    raise ModelTrainingError(
                        f"No active version found for {model_key}",
                        error_code="NO_ACTIVE_VERSION"
                    )
                version_id = self._active_versions[model_key]
            
            # Find version
            version = self._find_version(model_key, version_id)
            if version is None:
                raise ModelTrainingError(
                    f"Version {version_id} not found for {model_key}",
                    error_code="VERSION_NOT_FOUND"
                )
            
            # Load model
            model_file = Path(version.file_path)
            if not model_file.exists():
                raise ModelTrainingError(
                    f"Model file not found: {model_file}",
                    error_code="MODEL_FILE_NOT_FOUND"
                )
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            self.logger.info(f"Loaded model version {version_id} for {model_key}")
            return model, version
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(
                f"Failed to load model: {str(e)}",
                error_code="MODEL_LOAD_ERROR",
                details={"model_type": model_type, "pattern_length": pattern_length, "version_id": version_id}
            )
    
    def set_active_version(self, model_type: str, pattern_length: int, version_id: str) -> None:
        """
        Set the active version for a model type and pattern length.
        
        Args:
            model_type: Type of model
            pattern_length: Pattern length
            version_id: Version ID to set as active
        """
        model_key = self._get_model_key(model_type, pattern_length)
        
        # Verify version exists
        version = self._find_version(model_key, version_id)
        if version is None:
            raise ModelTrainingError(
                f"Version {version_id} not found for {model_key}",
                error_code="VERSION_NOT_FOUND"
            )
        
        # Update active versions
        old_active = self._active_versions.get(model_key)
        self._active_versions[model_key] = version_id
        
        # Update version metadata
        for v in self._version_registry.get(model_key, []):
            v.is_active = (v.version_id == version_id)
        
        self._save_version_registry()
        
        self.logger.info(f"Set active version for {model_key}: {version_id} (was: {old_active})")
    
    def rollback_to_version(self, model_type: str, pattern_length: int, version_id: str) -> None:
        """
        Rollback to a previous model version.
        
        Args:
            model_type: Type of model
            pattern_length: Pattern length
            version_id: Version ID to rollback to
        """
        model_key = self._get_model_key(model_type, pattern_length)
        current_active = self._active_versions.get(model_key)
        
        # Set the specified version as active
        self.set_active_version(model_type, pattern_length, version_id)
        
        self.logger.info(f"Rolled back {model_key} from {current_active} to {version_id}")
    
    def list_versions(self, model_type: str, pattern_length: int) -> List[ModelVersion]:
        """
        List all versions for a model type and pattern length.
        
        Args:
            model_type: Type of model
            pattern_length: Pattern length
            
        Returns:
            List of ModelVersion objects sorted by creation date (newest first)
        """
        model_key = self._get_model_key(model_type, pattern_length)
        versions = self._version_registry.get(model_key, [])
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def get_active_version(self, model_type: str, pattern_length: int) -> Optional[ModelVersion]:
        """
        Get the active version for a model type and pattern length.
        
        Args:
            model_type: Type of model
            pattern_length: Pattern length
            
        Returns:
            Active ModelVersion or None if no active version
        """
        model_key = self._get_model_key(model_type, pattern_length)
        active_version_id = self._active_versions.get(model_key)
        
        if active_version_id:
            return self._find_version(model_key, active_version_id)
        
        return None
    
    def delete_version(self, model_type: str, pattern_length: int, version_id: str) -> None:
        """
        Delete a specific model version.
        
        Args:
            model_type: Type of model
            pattern_length: Pattern length
            version_id: Version ID to delete
        """
        model_key = self._get_model_key(model_type, pattern_length)
        
        # Check if it's the active version
        if self._active_versions.get(model_key) == version_id:
            raise ModelTrainingError(
                f"Cannot delete active version {version_id}",
                error_code="CANNOT_DELETE_ACTIVE_VERSION"
            )
        
        # Find and remove version
        versions = self._version_registry.get(model_key, [])
        version_to_delete = None
        
        for i, version in enumerate(versions):
            if version.version_id == version_id:
                version_to_delete = versions.pop(i)
                break
        
        if version_to_delete is None:
            raise ModelTrainingError(
                f"Version {version_id} not found for {model_key}",
                error_code="VERSION_NOT_FOUND"
            )
        
        # Delete files
        version_dir = self.base_path / model_key / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        self._save_version_registry()
        
        self.logger.info(f"Deleted version {version_id} for {model_key}")
    
    def get_version_comparison(self, model_type: str, pattern_length: int, version_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple versions of a model.
        
        Args:
            model_type: Type of model
            pattern_length: Pattern length
            version_ids: List of version IDs to compare
            
        Returns:
            Comparison data including performance metrics and metadata
        """
        model_key = self._get_model_key(model_type, pattern_length)
        comparison = {
            "model_key": model_key,
            "versions": {},
            "performance_comparison": {}
        }
        
        for version_id in version_ids:
            version = self._find_version(model_key, version_id)
            if version:
                comparison["versions"][version_id] = version.to_dict()
        
        # Compare performance metrics
        metric_names = set()
        for version_id in version_ids:
            version = self._find_version(model_key, version_id)
            if version:
                metric_names.update(version.performance_metrics.keys())
        
        for metric_name in metric_names:
            comparison["performance_comparison"][metric_name] = {}
            for version_id in version_ids:
                version = self._find_version(model_key, version_id)
                if version and metric_name in version.performance_metrics:
                    comparison["performance_comparison"][metric_name][version_id] = version.performance_metrics[metric_name]
        
        return comparison
    
    def _generate_version_id(self, model_config: ModelConfiguration) -> str:
        """Generate a unique version ID based on configuration and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(json.dumps(asdict(model_config), sort_keys=True).encode()).hexdigest()[:8]
        return f"v{timestamp}_{config_hash}"
    
    def _get_model_key(self, model_type: str, pattern_length: int) -> str:
        """Generate a model key from type and pattern length."""
        return f"{model_type}_p{pattern_length}"
    
    def _find_version(self, model_key: str, version_id: str) -> Optional[ModelVersion]:
        """Find a version by model key and version ID."""
        versions = self._version_registry.get(model_key, [])
        for version in versions:
            if version.version_id == version_id:
                return version
        return None
    
    def _cleanup_old_versions(self, model_key: str) -> None:
        """Clean up old versions if we exceed the maximum."""
        versions = self._version_registry.get(model_key, [])
        if len(versions) <= self.max_versions:
            return
        
        # Sort by creation date and keep the newest versions
        versions.sort(key=lambda v: v.created_at, reverse=True)
        versions_to_keep = versions[:self.max_versions]
        versions_to_delete = versions[self.max_versions:]
        
        # Delete old versions
        for version in versions_to_delete:
            try:
                version_dir = self.base_path / model_key / version.version_id
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                self.logger.info(f"Cleaned up old version {version.version_id} for {model_key}")
            except Exception as e:
                self.logger.error(f"Failed to clean up version {version.version_id}: {str(e)}")
        
        # Update registry
        self._version_registry[model_key] = versions_to_keep
    
    def _load_version_registry(self) -> None:
        """Load version registry from disk."""
        registry_file = self.base_path / "version_registry.json"
        
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            for model_key, versions_data in registry_data.get("versions", {}).items():
                versions = [ModelVersion.from_dict(v) for v in versions_data]
                self._version_registry[model_key] = versions
            
            self._active_versions = registry_data.get("active_versions", {})
            
            self.logger.info(f"Loaded version registry with {len(self._version_registry)} model types")
            
        except Exception as e:
            self.logger.error(f"Failed to load version registry: {str(e)}")
    
    def _save_version_registry(self) -> None:
        """Save version registry to disk."""
        registry_file = self.base_path / "version_registry.json"
        
        try:
            registry_data = {
                "versions": {
                    model_key: [v.to_dict() for v in versions]
                    for model_key, versions in self._version_registry.items()
                },
                "active_versions": self._active_versions
            }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save version registry: {str(e)}")