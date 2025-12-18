"""
Configuration Management System for VectorBT Visualization Enhancement.

This module provides dynamic configuration management, user preference persistence,
and configuration templates for different analysis types.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict, fields
import logging
from datetime import datetime

from .portfolio_config import PlotConfig, PortfolioConfig
from ..utils.exceptions import DataValidationError


class ConfigurationManager:
    """
    Manages dynamic configuration updates, user preferences, and configuration templates.
    
    This class provides functionality for:
    - Saving and loading user preferences
    - Runtime configuration updates with validation
    - Configuration templates for different analysis types
    - Configuration versioning and migration
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the Configuration Manager.
        
        Args:
            config_dir: Directory to store configuration files. 
                       Defaults to ~/.vectorbt_viz_config
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up configuration directory
        if config_dir is None:
            config_dir = os.path.expanduser("~/.vectorbt_viz_config")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration file paths
        self.user_prefs_file = self.config_dir / "user_preferences.json"
        self.templates_file = self.config_dir / "templates.json"
        self.history_file = self.config_dir / "config_history.json"
        
        # Initialize default templates
        self._initialize_default_templates()
    
    def save_user_preferences(
        self, 
        plot_config: PlotConfig, 
        portfolio_config: Optional[PortfolioConfig] = None,
        preference_name: str = "default"
    ) -> None:
        """
        Save user preferences to persistent storage.
        
        Args:
            plot_config: PlotConfig to save as user preference
            portfolio_config: Optional PortfolioConfig to save
            preference_name: Name for this preference set
            
        Raises:
            DataValidationError: If configuration is invalid
        """
        try:
            # Validate configurations
            plot_config.validate()
            if portfolio_config:
                portfolio_config.validate()
            
            # Load existing preferences
            preferences = self._load_preferences_file()
            
            # Create preference entry
            preference_data = {
                'plot_config': asdict(plot_config),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            if portfolio_config:
                preference_data['portfolio_config'] = asdict(portfolio_config)
            
            # Save preference
            preferences[preference_name] = preference_data
            
            # Write to file
            with open(self.user_prefs_file, 'w') as f:
                json.dump(preferences, f, indent=2)
            
            self.logger.info(f"Saved user preference: {preference_name}")
            
        except Exception as e:
            raise DataValidationError(f"Failed to save user preferences: {str(e)}")
    
    def load_user_preferences(
        self, 
        preference_name: str = "default"
    ) -> Dict[str, Union[PlotConfig, PortfolioConfig]]:
        """
        Load user preferences from persistent storage.
        
        Args:
            preference_name: Name of preference set to load
            
        Returns:
            Dictionary containing 'plot_config' and optionally 'portfolio_config'
            
        Raises:
            DataValidationError: If preference not found or invalid
        """
        try:
            preferences = self._load_preferences_file()
            
            if preference_name not in preferences:
                raise DataValidationError(f"Preference '{preference_name}' not found")
            
            pref_data = preferences[preference_name]
            
            # Reconstruct PlotConfig
            plot_config = PlotConfig(**pref_data['plot_config'])
            
            result = {'plot_config': plot_config}
            
            # Reconstruct PortfolioConfig if present
            if 'portfolio_config' in pref_data:
                portfolio_config = PortfolioConfig(**pref_data['portfolio_config'])
                result['portfolio_config'] = portfolio_config
            
            self.logger.info(f"Loaded user preference: {preference_name}")
            return result
            
        except Exception as e:
            raise DataValidationError(f"Failed to load user preferences: {str(e)}")
    
    def list_user_preferences(self) -> List[Dict[str, Any]]:
        """
        List all saved user preferences.
        
        Returns:
            List of preference metadata dictionaries
        """
        try:
            preferences = self._load_preferences_file()
            
            result = []
            for name, data in preferences.items():
                metadata = {
                    'name': name,
                    'created_at': data.get('created_at'),
                    'updated_at': data.get('updated_at'),
                    'has_portfolio_config': 'portfolio_config' in data
                }
                result.append(metadata)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to list preferences: {str(e)}")
            return []
    
    def delete_user_preference(self, preference_name: str) -> bool:
        """
        Delete a user preference.
        
        Args:
            preference_name: Name of preference to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            preferences = self._load_preferences_file()
            
            if preference_name in preferences:
                del preferences[preference_name]
                
                with open(self.user_prefs_file, 'w') as f:
                    json.dump(preferences, f, indent=2)
                
                self.logger.info(f"Deleted user preference: {preference_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete preference: {str(e)}")
            return False
    
    def update_configuration_runtime(
        self, 
        config: Union[PlotConfig, PortfolioConfig], 
        updates: Dict[str, Any],
        validate: bool = True
    ) -> Union[PlotConfig, PortfolioConfig]:
        """
        Update configuration at runtime with validation.
        
        Args:
            config: Configuration object to update
            updates: Dictionary of field updates
            validate: Whether to validate after updates
            
        Returns:
            Updated configuration object
            
        Raises:
            DataValidationError: If updates are invalid
        """
        try:
            # Create a copy to avoid modifying original
            import copy
            updated_config = copy.deepcopy(config)
            
            # Get valid field names for the configuration type
            valid_fields = {field.name for field in fields(type(config))}
            
            # Apply updates
            for field_name, value in updates.items():
                if field_name not in valid_fields:
                    raise DataValidationError(
                        f"Invalid field '{field_name}' for {type(config).__name__}"
                    )
                
                setattr(updated_config, field_name, value)
            
            # Validate if requested
            if validate:
                updated_config.validate()
            
            self.logger.info(f"Updated {type(config).__name__} with {len(updates)} changes")
            return updated_config
            
        except Exception as e:
            raise DataValidationError(f"Failed to update configuration: {str(e)}")
    
    def create_configuration_template(
        self,
        template_name: str,
        plot_config: PlotConfig,
        portfolio_config: Optional[PortfolioConfig] = None,
        description: str = "",
        analysis_type: str = "general"
    ) -> None:
        """
        Create a configuration template for specific analysis types.
        
        Args:
            template_name: Name for the template
            plot_config: PlotConfig for the template
            portfolio_config: Optional PortfolioConfig for the template
            description: Description of the template's purpose
            analysis_type: Type of analysis this template is for
            
        Raises:
            DataValidationError: If template creation fails
        """
        try:
            # Validate configurations
            plot_config.validate()
            if portfolio_config:
                portfolio_config.validate()
            
            # Load existing templates
            templates = self._load_templates_file()
            
            # Create template entry
            template_data = {
                'plot_config': asdict(plot_config),
                'description': description,
                'analysis_type': analysis_type,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            if portfolio_config:
                template_data['portfolio_config'] = asdict(portfolio_config)
            
            # Save template
            templates[template_name] = template_data
            
            # Write to file
            with open(self.templates_file, 'w') as f:
                json.dump(templates, f, indent=2)
            
            self.logger.info(f"Created configuration template: {template_name}")
            
        except Exception as e:
            raise DataValidationError(f"Failed to create template: {str(e)}")
    
    def load_configuration_template(
        self, 
        template_name: str
    ) -> Dict[str, Union[PlotConfig, PortfolioConfig, str]]:
        """
        Load a configuration template.
        
        Args:
            template_name: Name of template to load
            
        Returns:
            Dictionary containing configuration objects and metadata
            
        Raises:
            DataValidationError: If template not found or invalid
        """
        try:
            templates = self._load_templates_file()
            
            if template_name not in templates:
                raise DataValidationError(f"Template '{template_name}' not found")
            
            template_data = templates[template_name]
            
            # Reconstruct PlotConfig
            plot_config = PlotConfig(**template_data['plot_config'])
            
            result = {
                'plot_config': plot_config,
                'description': template_data.get('description', ''),
                'analysis_type': template_data.get('analysis_type', 'general'),
                'version': template_data.get('version', '1.0')
            }
            
            # Reconstruct PortfolioConfig if present
            if 'portfolio_config' in template_data:
                portfolio_config = PortfolioConfig(**template_data['portfolio_config'])
                result['portfolio_config'] = portfolio_config
            
            self.logger.info(f"Loaded configuration template: {template_name}")
            return result
            
        except Exception as e:
            raise DataValidationError(f"Failed to load template: {str(e)}")
    
    def list_configuration_templates(
        self, 
        analysis_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available configuration templates.
        
        Args:
            analysis_type: Filter by analysis type (optional)
            
        Returns:
            List of template metadata dictionaries
        """
        try:
            templates = self._load_templates_file()
            
            result = []
            for name, data in templates.items():
                # Filter by analysis type if specified
                if analysis_type and data.get('analysis_type') != analysis_type:
                    continue
                
                metadata = {
                    'name': name,
                    'description': data.get('description', ''),
                    'analysis_type': data.get('analysis_type', 'general'),
                    'created_at': data.get('created_at'),
                    'version': data.get('version', '1.0'),
                    'has_portfolio_config': 'portfolio_config' in data
                }
                result.append(metadata)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to list templates: {str(e)}")
            return []
    
    def get_configuration_preview(
        self, 
        config: Union[PlotConfig, PortfolioConfig]
    ) -> Dict[str, Any]:
        """
        Generate a preview of configuration settings.
        
        Args:
            config: Configuration to preview
            
        Returns:
            Dictionary with configuration preview information
        """
        try:
            config_dict = asdict(config)
            config_type = type(config).__name__
            
            # Categorize settings for better preview
            if isinstance(config, PlotConfig):
                preview = {
                    'type': config_type,
                    'dimensions': {
                        'width': config.width,
                        'height': config.height
                    },
                    'theme': {
                        'template': config.template,
                        'theme': config.theme,
                        'color_scheme': config.color_scheme
                    },
                    'display': {
                        'show_trades': config.show_trades,
                        'show_metrics': config.show_metrics,
                        'show_drawdown': config.show_drawdown
                    },
                    'export': {
                        'formats': config.export_formats,
                        'dpi': config.export_dpi
                    }
                }
            else:  # PortfolioConfig
                preview = {
                    'type': config_type,
                    'capital': {
                        'init_cash': config.init_cash,
                        'size_strategy': config.size_strategy,
                        'size_value': config.size_value
                    },
                    'costs': {
                        'fees': config.fees,
                        'slippage': config.slippage
                    },
                    'risk_management': {
                        'stop_loss': config.stop_loss,
                        'take_profit': config.take_profit
                    }
                }
            
            return preview
            
        except Exception as e:
            self.logger.error(f"Failed to generate preview: {str(e)}")
            return {'type': type(config).__name__, 'error': str(e)}
    
    def _load_preferences_file(self) -> Dict[str, Any]:
        """Load preferences from file, creating empty dict if file doesn't exist."""
        if self.user_prefs_file.exists():
            try:
                with open(self.user_prefs_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load preferences file: {str(e)}")
        
        return {}
    
    def _load_templates_file(self) -> Dict[str, Any]:
        """Load templates from file, creating defaults if file doesn't exist."""
        if self.templates_file.exists():
            try:
                with open(self.templates_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load templates file: {str(e)}")
        
        return self._get_default_templates()
    
    def _initialize_default_templates(self) -> None:
        """Initialize default configuration templates."""
        if not self.templates_file.exists():
            try:
                default_templates = self._get_default_templates()
                with open(self.templates_file, 'w') as f:
                    json.dump(default_templates, f, indent=2)
                
                self.logger.info("Initialized default configuration templates")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize default templates: {str(e)}")
    
    def _get_default_templates(self) -> Dict[str, Any]:
        """Get default configuration templates."""
        return {
            'trading_analysis': {
                'plot_config': asdict(PlotConfig.create_preset('trading')),
                'description': 'Comprehensive trading analysis with all indicators',
                'analysis_type': 'trading',
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'research_analysis': {
                'plot_config': asdict(PlotConfig.create_preset('research')),
                'description': 'Clean research-focused visualization',
                'analysis_type': 'research',
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'presentation': {
                'plot_config': asdict(PlotConfig.create_preset('presentation')),
                'description': 'High-quality presentation-ready plots',
                'analysis_type': 'presentation',
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'dashboard': {
                'plot_config': asdict(PlotConfig.create_preset('dashboard')),
                'description': 'Compact dashboard-friendly visualization',
                'analysis_type': 'dashboard',
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }


class ConfigurationValidator:
    """
    Validates configuration changes and provides suggestions for improvements.
    """
    
    @staticmethod
    def validate_configuration_compatibility(
        plot_config: PlotConfig, 
        portfolio_config: PortfolioConfig
    ) -> Dict[str, List[str]]:
        """
        Validate compatibility between plot and portfolio configurations.
        
        Args:
            plot_config: PlotConfig to validate
            portfolio_config: PortfolioConfig to validate
            
        Returns:
            Dictionary with 'warnings' and 'errors' lists
        """
        warnings = []
        errors = []
        
        try:
            # Validate individual configurations
            plot_config.validate()
            portfolio_config.validate()
            
            # Check compatibility issues
            if plot_config.show_trades and not plot_config.show_positions:
                warnings.append("Showing trades without positions may be confusing")
            
            if plot_config.show_benchmark and not plot_config.benchmark_symbol:
                errors.append("Benchmark display enabled but no benchmark symbol specified")
            
            if portfolio_config.size_strategy == 'percent_equity' and portfolio_config.size_value > 0.5:
                warnings.append("High equity percentage (>50%) may increase risk")
            
            if portfolio_config.fees > 0.01:  # 1%
                warnings.append("High transaction fees may significantly impact returns")
            
            if plot_config.export_dpi > 300 and 'png' in plot_config.export_formats:
                warnings.append("High DPI PNG exports may result in large file sizes")
            
        except Exception as e:
            errors.append(f"Configuration validation failed: {str(e)}")
        
        return {'warnings': warnings, 'errors': errors}
    
    @staticmethod
    def suggest_configuration_improvements(
        config: Union[PlotConfig, PortfolioConfig]
    ) -> List[str]:
        """
        Suggest improvements for a configuration.
        
        Args:
            config: Configuration to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        if isinstance(config, PlotConfig):
            # Plot configuration suggestions
            if config.width < 800:
                suggestions.append("Consider increasing width to at least 800px for better readability")
            
            if config.marker_size < 6:
                suggestions.append("Small marker size may make trades hard to see")
            
            if not config.show_metrics:
                suggestions.append("Enable metrics display for better performance analysis")
            
            if config.theme == 'default' and config.template == 'plotly_white':
                suggestions.append("Consider using a themed preset for better visual consistency")
            
        elif isinstance(config, PortfolioConfig):
            # Portfolio configuration suggestions
            if config.init_cash < 1000:
                suggestions.append("Low initial capital may not provide realistic trading simulation")
            
            if config.fees == 0:
                suggestions.append("Consider adding realistic transaction fees for accurate backtesting")
            
            if config.stop_loss is None:
                suggestions.append("Consider adding stop-loss for risk management")
            
            if config.size_strategy == 'fixed_amount' and config.size_value > config.init_cash * 0.1:
                suggestions.append("Large fixed position size may lead to over-concentration")
        
        return suggestions